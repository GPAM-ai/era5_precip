#!/usr/bin/env python
"""
Etapa 09 - Indices climaticos externos (NOAA/PSL).

POR QUE:
  O pacote da competicao nao traz temperatura da superficie do mar. Ate
  aqui o ENOS entrou indiretamente, via pressao e geopotencial em 850 hPa,
  e o dominio para em -90 de longitude - pega so a borda do Pacifico.
  Indices de TSM dao o estado do ENOS e do Atlantico tropical de forma
  direta, que eh o que governa a chuva na America do Sul em escala mensal.

REGULAMENTO:
  Secao 2.6 permite dados externos de dominio publico, gratuitos e
  igualmente acessiveis a todos. Os indices do NOAA/PSL sao exatamente
  isso. Fonte citada no README.

REGRA DE VAZAMENTO, aplicada de forma rigida:
  Para prever o mes M+1, so entram indices observados ate o mes M. Os
  arquivos da NOAA vao ate 2026, entao o valor de M+1 EXISTE no arquivo -
  e nunca eh usado. A serie eh indexada pelo mes-feature, como os PCs.

INDICES:
  Nino 1+2   Pacifico leste, costa do Peru. Responde mais rapido, mais
             proximo do dominio.
  Nino 3.4   o indice padrao do ENOS.
  Nino 4     Pacifico central. Distingue El Nino canonico de Modoki.
  ONI        media movel de 3 meses do 3.4; suaviza ruido mensal.
  TNA, TSA   Atlantico tropical norte e sul. A diferenca eh o dipolo do
             Atlantico, que governa Nordeste e posicao da ZCIT.
  SOI        Tahiti - Darwin; a contraparte atmosferica do ENOS.
  PDO        Pacifico norte decadal; modula a resposta ao ENOS.

VALIDACAO:
  Correlacao do Nino 3.4 baixado com o pc_slp_2 e o idx_pacifico que
  extraimos em 03. Se os sinais e magnitudes baterem, o alinhamento
  temporal esta certo e a pegada indireta que tinhamos era real.

SAIDAS:
  dados_processados/indices_noaa.parquet     series alinhadas por mes-feature
  dados_processados/X_treino_sst.npy         X original + bloco de indices
  dados_processados/X_teste_sst.npy
  dados_processados/features_sst.json

Uso:
    python src/09_indices_noaa.py
"""

import io
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

SAIDA = "dados_processados"
os.makedirs(f"{SAIDA}/indices_brutos", exist_ok=True)

# candidatos de URL por indice, em ordem de preferencia. A NOAA mantem os
# arquivos no formato antigo em /data/correlation/ e no novo em
# /data/timeseries/month/data/. Tenta um, cai para o outro.
BASE_OLD = "https://psl.noaa.gov/data/correlation/"
BASE_NEW = "https://psl.noaa.gov/data/timeseries/month/data/"

INDICES = {
    "nino12": [BASE_OLD + "nina1.anom.data", BASE_NEW + "nino12.long.anom.data"],
    "nino34": [BASE_OLD + "nina34.anom.data", BASE_NEW + "nino34.long.anom.data"],
    "nino4":  [BASE_OLD + "nina4.anom.data",  BASE_NEW + "nino4.long.anom.data"],
    "oni":    [BASE_OLD + "oni.data",         BASE_NEW + "oni.data"],
    "tna":    [BASE_OLD + "tna.data",         BASE_NEW + "tna.data"],
    "tsa":    [BASE_OLD + "tsa.data",         BASE_NEW + "tsa.data"],
    "soi":    [BASE_OLD + "soi.data",         BASE_NEW + "soi.data"],
    "pdo":    [BASE_OLD + "pdo.data",         BASE_NEW + "pdo.data"],
}

LAGS = (1, 2, 3, 6)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------------------
sep("1. DOWNLOAD")


def baixar(url, timeout=30):
    """Baixa com verificacao de certificado; se falhar (proxy do cluster
    com CA propria, ou pacote de CAs desatualizado), tenta sem verificar.
    Aceitavel aqui: sao arquivos publicos de texto da NOAA, sem
    credencial envolvida."""
    import ssl
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", errors="replace")


def parse_psl(texto):
    """Formato PSL: linha 'ano1 anoN', depois linhas 'ano v1..v12',
    depois uma linha com o valor de missing, depois atribuicao."""
    linhas = [l for l in texto.splitlines() if l.strip()]
    tok = linhas[0].split()
    a0, a1 = int(float(tok[0])), int(float(tok[1]))

    registros, i = [], 1
    while i < len(linhas):
        t = linhas[i].split()
        if len(t) == 13:
            try:
                ano = int(float(t[0]))
                if a0 <= ano <= a1:
                    registros.append([ano] + [float(v) for v in t[1:]])
                    i += 1
                    continue
            except ValueError:
                pass
        break

    # valor de missing: primeira linha depois dos dados que eh um numero so
    missing = None
    for l in linhas[i:i + 3]:
        t = l.split()
        if len(t) == 1:
            try:
                missing = float(t[0])
                break
            except ValueError:
                pass

    df = pd.DataFrame(registros, columns=["ano"] + list(range(1, 13)))
    longo = df.melt(id_vars="ano", var_name="mes", value_name="valor")
    longo["data"] = pd.to_datetime(
        longo.ano.astype(str) + "-" + longo.mes.astype(str).str.zfill(2) + "-01")
    s = longo.set_index("data").valor.sort_index()

    if missing is not None:
        s = s.where(~np.isclose(s, missing))
    # alguns arquivos usam -99.9, -999 sem declarar
    s = s.where(s > -90)
    return s.astype("float32")


series = {}
fontes = {}

# O firewall do cluster bloqueia psl.noaa.gov ("Blocked site"). Os
# arquivos sao baixados fora e colocados em dados_processados/indices_brutos/.
# Tenta download primeiro, para o caso de rodar em ambiente com acesso;
# se falhar, le do disco.
for nome, urls in INDICES.items():
    arq = f"{SAIDA}/indices_brutos/{nome}.data"
    texto = None

    for url in urls:
        try:
            t = baixar(url)
            if t.lstrip().startswith("<"):
                raise ValueError("resposta eh HTML (proxy ou pagina de erro)")
            texto, fontes[nome] = t, url
            with open(arq, "w") as f:
                f.write(texto)
            print(f"  {nome:8s} baixado de {url}")
            break
        except Exception:
            pass

    if texto is None and os.path.exists(arq):
        with open(arq) as f:
            texto = f.read()
        if texto.lstrip().startswith("<"):
            print(f"  !! {nome}: arquivo em disco eh HTML. Refazer o download.")
            continue
        fontes[nome] = urls[1] + "  (baixado externamente)"
        print(f"  {nome:8s} lido do disco")

    if texto is None:
        print(f"  !! {nome}: sem download e sem arquivo em {arq}")
        continue

    s = parse_psl(texto)
    if s.notna().sum() < 500:
        print(f"  !! {nome}: so {s.notna().sum()} valores validos, descartado")
        continue
    series[nome] = s
    print(f"  {'':8s} {s.index[0]:%Y-%m} a {s.dropna().index[-1]:%Y-%m}  "
          f"({s.notna().sum()} valores)")

if not series:
    raise SystemExit("\n  nenhum indice disponivel. Baixar no Mac e subir por scp.")

print(f"\n  {len(series)} de {len(INDICES)} indices obtidos")


# ---------------------------------------------------------------------------
sep("2. ALINHAMENTO POR MES-FEATURE")

# a serie unificada de meses-feature vai de jan/1940 a nov/2024 (ver 03)
pcs = pd.read_parquet(f"{SAIDA}/pcs.parquet")
meses = pcs.index
print(f"  meses-feature: {meses[0]:%Y-%m} a {meses[-1]:%Y-%m} ({len(meses)})")

tab = pd.DataFrame(index=meses)
for nome, s in series.items():
    tab[nome] = s.reindex(meses)

print(f"\n  {'indice':8s} {'NaN':>6s} {'primeiro NaN':>14s}  {'ultimo valor':>13s}")
for c in tab.columns:
    nan = int(tab[c].isna().sum())
    prim = tab[c][tab[c].isna()].index
    prim = f"{prim[0]:%Y-%m}" if len(prim) else "-"
    ult = tab[c].dropna().index[-1]
    print(f"  {c:8s} {nan:6d} {prim:>14s}  {ult:%Y-%m}")

# derivados fisicos
if "tna" in tab and "tsa" in tab:
    tab["dipolo_atl"] = tab["tna"] - tab["tsa"]
    print("\n  dipolo_atl = TNA - TSA (gradiente meridional do Atlantico)")
if "nino34" in tab:
    tab["nino34_3m"] = tab["nino34"].rolling(3, min_periods=1).mean()
if "nino12" in tab and "nino4" in tab:
    tab["nino_grad"] = tab["nino12"] - tab["nino4"]
    print("  nino_grad = Nino1+2 - Nino4 (leste vs centro, canonico vs Modoki)")

# preenche NaN no INICIO da serie (indices que comecam depois de 1940)
# com zero = anomalia neutra. NaN no FIM seria vazamento mascarado - aborta.
fim = tab.index[-1]
for c in tab.columns:
    if pd.isna(tab.loc[fim, c]):
        raise SystemExit(f"\n  !! {c} tem NaN em {fim:%Y-%m}, o ultimo mes-feature. "
                         "Sem esse valor nao ha como prever dez/2024.")
tab = tab.fillna(0.0)

base = list(tab.columns)
print(f"\n  {len(base)} series base")

# MODO ROBUSTO (diagnostico em 10): TNA e TSA absolutos estao fora da
# distribuicao de treino em 2023/24 (58-75% das linhas alem do p99; TSA
# 21% alem do maximo historico). O dipolo TNA-TSA cancela o aquecimento
# uniforme e fica dentro do visto (0% fora). Mantem-se o gradiente,
# descartam-se os absolutos.
ROBUSTO = os.environ.get("ROBUSTO", "1") == "1"
if ROBUSTO:
    descartar = [c for c in ("tna", "tsa") if c in tab]
    tab = tab.drop(columns=descartar)
    base = list(tab.columns)
    print(f"  modo robusto: descartados {descartar}, mantido dipolo_atl")
    print(f"  {len(base)} series base apos filtro")


# ---------------------------------------------------------------------------
sep("3. VALIDACAO CONTRA OS MODOS EXTRAIDOS EM 03")

print("""  Se o Nino 3.4 baixado for real e estiver alinhado, deve
  correlacionar com os modos de ENOS que extraimos da pressao. O sinal
  esperado eh NEGATIVO com idx_pacifico (El Nino = pressao baixa no
  Pacifico leste) e com pc_slp_2 (que teve separacao -1.88 entre
  El Nino e La Nina em 03).
""")

comum = tab.index.intersection(pcs.index)
for ref in ("idx_pacifico", "pc_slp_2"):
    if ref in pcs:
        for c in ("nino34", "nino12", "oni", "soi"):
            if c in tab:
                r = np.corrcoef(tab.loc[comum, c], pcs.loc[comum, ref])[0, 1]
                print(f"  corr({c:7s}, {ref:13s}) = {r:+.3f}")
        print()

if "nino34" in tab and "soi" in tab:
    r = np.corrcoef(tab.nino34, tab.soi)[0, 1]
    print(f"  corr(nino34, soi) = {r:+.3f}   (esperado fortemente negativo,")
    print("  ~ -0.6 a -0.8: SOI eh a contraparte atmosferica do ENOS)")

# posicao de 2023/2024 no espaco dos indices
print("\n  Nino 3.4 nos meses-feature do teste (deve mostrar El Nino forte")
print("  em 2023 virando para La Nina no 2o semestre de 2024):")
te = tab.loc["2022-12":"2024-11", "nino34"]
for d in te.index[::3]:
    print(f"    {d:%Y-%m}: {te[d]:+.2f}")


# ---------------------------------------------------------------------------
sep("4. DEFASAGENS E TENDENCIAS")

extras = {}
for lag in LAGS:
    for c in base:
        extras[f"{c}_lag{lag}"] = tab[c].shift(lag)
for c in base:
    extras[f"{c}_tend"] = tab[c] - tab[c].shift(3)

tab = pd.concat([tab, pd.DataFrame(extras, index=tab.index)], axis=1)
tab = tab.fillna(0.0).astype("float32")     # NaN so no inicio (1940)
tab.index.name = "mes_feature"
tab.to_parquet(f"{SAIDA}/indices_noaa.parquet")
print(f"  tabela: {tab.shape}  salva em indices_noaa.parquet")


# ---------------------------------------------------------------------------
sep("5. MATRIZES AUMENTADAS")

with open(f"{SAIDA}/features.json") as f:
    spec = json.load(f)
nomes = spec["nomes"]
meta_tr = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

# treino: linhas em blocos por passo de tempo, indice de tempo 3..994
X = np.load(f"{SAIDA}/X_treino.npy")
n_pt = int(spec["n_linhas_treino"]) // 992
assert n_pt * 992 == len(X), "numero de pontos nao bate"
idx_t = np.arange(3, 3 + 992)
meses_tr = pcs.index[idx_t]
bloco_tr = tab.loc[meses_tr].values                  # (992, n_ind)
Xs = np.hstack([X, np.repeat(bloco_tr, n_pt, axis=0)]).astype("float32")
del X
print(f"  X_treino_sst: {Xs.shape}  ({Xs.nbytes/1e9:.2f} GB)")
np.save(f"{SAIDA}/X_treino_sst.npy", Xs)
del Xs

# teste: mes-feature = mes-alvo do id menos um
Xte = np.load(f"{SAIDA}/X_teste.npy")
data_alvo = pd.to_datetime(meta_te["ano"].astype(str) + "-"
                           + meta_te["mes_alvo"].astype(str).str.zfill(2) + "-01")
mes_feat_te = data_alvo - pd.DateOffset(months=1)
bloco_te = tab.loc[mes_feat_te].values
Xtes = np.hstack([Xte, bloco_te]).astype("float32")
del Xte
print(f"  X_teste_sst : {Xtes.shape}")
assert not np.isnan(Xtes).any()
np.save(f"{SAIDA}/X_teste_sst.npy", Xtes)
del Xtes

nomes_sst = nomes + list(tab.columns)
with open(f"{SAIDA}/features_sst.json", "w") as f:
    json.dump({"nomes": nomes_sst, "n_original": len(nomes),
               "indices_noaa": list(tab.columns), "fontes": fontes,
               "lags": LAGS,
               "citacao": "NOAA/PSL Monthly Climate Indices, "
                          "https://psl.noaa.gov/data/timeseries/month/"},
              f, indent=2)

print(f"""
  features: {len(nomes)} originais + {len(tab.columns)} de indices = {len(nomes_sst)}

  Proximo passo: 06_submissao.py com X_treino_sst.npy / X_teste_sst.npy
  (variavel de ambiente USAR_SST=1, ou editar os caminhos).""")