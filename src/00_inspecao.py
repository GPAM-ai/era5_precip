#!/usr/bin/env python
"""
Inspeção inicial dos dados do Hackathon WorCAP 2026.
Roda antes de qualquer modelagem. Responde:
  1. o que tem no pacote (arquivos, variaveis, dimensoes, periodo)
  2. unidade da precipitacao (mm/dia? metros?)
  3. orientacao da grade (latitude crescente? campo espelhado?)
  4. formato do sample_submission

Uso:
    python 00_inspecao.py dados_brutos
"""

import sys
import os
import glob

import numpy as np

DADOS = sys.argv[1] if len(sys.argv) > 1 else "."
SAIDA = "inspecao_saida"
os.makedirs(SAIDA, exist_ok=True)

sep = lambda t: print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------- 1. arquivos
sep("1. ARQUIVOS NO PACOTE")

arquivos = sorted(glob.glob(os.path.join(DADOS, "**", "*"), recursive=True))
for f in arquivos:
    if os.path.isfile(f):
        tam = os.path.getsize(f) / 1e6
        print(f"  {tam:10.1f} MB  {os.path.relpath(f, DADOS)}")

nc = [f for f in arquivos if f.endswith((".nc", ".nc4", ".grib", ".zarr"))]
csv = [f for f in arquivos if f.endswith(".csv")]
print(f"\n  {len(nc)} arquivos de dados em grade, {len(csv)} CSVs")


# ------------------------------------------------------- 2. conteudo netcdf
sep("2. CONTEUDO DOS ARQUIVOS EM GRADE")

try:
    import xarray as xr
except ImportError:
    print("  !! xarray nao instalado no ambiente. Instale antes de continuar.")
    sys.exit(1)

ds = None
for caminho in nc:
    print(f"\n--- {os.path.relpath(caminho, DADOS)}")
    try:
        d = xr.open_dataset(caminho)
    except Exception as e:
        print(f"    erro ao abrir: {e}")
        continue

    print(f"    dims: {dict(d.sizes)}")
    print(f"    vars: {list(d.data_vars)}")
    for v in d.data_vars:
        attrs = d[v].attrs
        unidade = attrs.get("units", "SEM UNIDADE DECLARADA")
        nome = attrs.get("long_name", "")
        print(f"      - {v:20s} [{unidade}]  {nome}  dtype={d[v].dtype}")

    # periodo coberto
    for cand in ("time", "valid_time", "month", "date"):
        if cand in d.coords:
            t = d[cand].values
            print(f"    {cand}: {len(t)} passos, de {t[0]} a {t[-1]}")
            break

    if ds is None and any("t" in v.lower() or "prec" in v.lower() for v in d.data_vars):
        ds = d
        ds_path = caminho

if ds is None and nc:
    ds = xr.open_dataset(nc[0])
    ds_path = nc[0]


# ------------------------------------------------- 3. unidade da precipitacao
sep("3. UNIDADE DA PRECIPITACAO")

if ds is None:
    print("  nenhum dataset aberto; pule esta secao")
else:
    # tenta achar a variavel de precipitacao
    cands = [v for v in ds.data_vars
             if v.lower() in ("tp", "precip", "precipitation", "tp_mm_day", "pr")]
    if not cands:
        cands = [v for v in ds.data_vars if "p" in v.lower()][:1]

    if not cands:
        print("  variavel de precipitacao nao identificada automaticamente.")
    else:
        var = cands[0]
        print(f"  variavel: {var}")
        amostra = ds[var].isel({d: slice(0, 24) for d in ds[var].dims
                                if d in ("time", "valid_time")}).values
        amostra = amostra[np.isfinite(amostra)]

        print(f"    media   : {amostra.mean():.6g}")
        print(f"    mediana : {np.median(amostra):.6g}")
        print(f"    p99     : {np.percentile(amostra, 99):.6g}")
        print(f"    max     : {amostra.max():.6g}")

        m = amostra.mean()
        if 1 < m < 6:
            print("\n  >> compativel com mm/dia. Nada a converter.")
        elif m < 0.01:
            print("\n  >> parece metros/dia (ERA5 nativo). MULTIPLIQUE POR 1000.")
        elif m > 30:
            print("\n  >> parece acumulado mensal em mm. DIVIDA pelos dias do mes.")
        else:
            print("\n  >> inconclusivo, confira manualmente.")


# ------------------------------------------------- 4. orientacao da grade
sep("4. ORIENTACAO DA GRADE")

if ds is not None:
    latname = "latitude" if "latitude" in ds.coords else "lat"
    lonname = "longitude" if "longitude" in ds.coords else "lon"

    if latname in ds.coords:
        lat = ds[latname].values
        lon = ds[lonname].values
        print(f"  {latname}: {lat[0]:.2f} .. {lat[-1]:.2f}  ({len(lat)} pontos, "
              f"passo {abs(lat[1]-lat[0]):.2f})")
        print(f"  {lonname}: {lon[0]:.2f} .. {lon[-1]:.2f}  ({len(lon)} pontos, "
              f"passo {abs(lon[1]-lon[0]):.2f})")
        print(f"  latitude {'CRESCENTE (sul->norte)' if lat[0] < lat[-1] else 'DECRESCENTE (norte->sul)'}")

        # plot de sanidade: climatologia de janeiro
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            var = cands[0]
            tname = "time" if "time" in ds.coords else "valid_time"
            clim_jan = ds[var].sel({tname: ds[tname].dt.month == 1}).mean(tname)

            fig, ax = plt.subplots(figsize=(7, 8))
            clim_jan.plot(ax=ax, cmap="Blues",
                          y=latname, x=lonname)  # xarray respeita a coordenada
            ax.set_title("Climatologia de JANEIRO\n"
                         "Amazonia deve aparecer umida ao NORTE; "
                         "Atacama seco na faixa oeste ~20S")
            fig.savefig(f"{SAIDA}/sanidade_clim_janeiro.png", dpi=110,
                        bbox_inches="tight")
            print(f"\n  figura salva em {SAIDA}/sanidade_clim_janeiro.png")
            print("  ABRA A FIGURA. Se estiver de cabeca para baixo, "
                  "o resto do pipeline esta errado.")
        except Exception as e:
            print(f"  nao consegui plotar: {e}")


# ------------------------------------------------- 5. sample_submission
sep("5. SAMPLE_SUBMISSION")

sub = [f for f in csv if "sample" in os.path.basename(f).lower()
       or "submission" in os.path.basename(f).lower()]

if not sub:
    print("  sample_submission nao encontrado no diretorio.")
else:
    import pandas as pd
    s = pd.read_csv(sub[0])
    print(f"  arquivo : {os.path.relpath(sub[0], DADOS)}")
    print(f"  shape   : {s.shape}")
    print(f"  colunas : {list(s.columns)}")
    print("\n  primeiras linhas:")
    print(s.head(3).to_string(index=False))
    print("\n  ultimas linhas:")
    print(s.tail(3).to_string(index=False))

    idcol = s.columns[0]
    # decompoe o id ano_mes_lat_lon
    try:
        partes = s[idcol].str.split("_", expand=True)
        anos = sorted(partes[0].unique())
        print(f"\n  anos presentes : {anos}")
        print(f"  meses por ano  : {partes[1].nunique()}")
        print(f"  pontos unicos  : {(partes[2] + '_' + partes[3]).nunique()}")
        esperado = len(anos) * partes[1].nunique() * (partes[2] + '_' + partes[3]).nunique()
        print(f"  linhas esperadas: {esperado}  |  reais: {len(s)}")
        if "2024" in anos:
            print("\n  >> 2024 esta na submissao: o CSV cobre publico E privado.")
    except Exception as e:
        print(f"  nao consegui decompor o id: {e}")


# ------------------------------------------------- 6. o que ainda falta saber
sep("6. PERGUNTAS QUE A INSPECAO PRECISA TER RESPONDIDO")

print("""
  [ ] as covariaveis de 2023 e 2024 estao no pacote?
  [ ] a precipitacao OBSERVADA de 2023 esta no pacote?
      (se sim -> persistencia de anomalia entra como feature)
      (se nao -> usar umidade/convergencia como substituto)
  [ ] geopotencial vem em quantos niveis?
  [ ] existe alguma variavel a mais alem das 8 anunciadas?
  [ ] latitude crescente confere com o anunciado no slide?
""")