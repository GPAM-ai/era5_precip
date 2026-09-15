#!/usr/bin/env python
"""
Etapa 01d - Climatologia final: mistura + tendencia encolhida.

O QUE 01c ESTABELECEU:
  - a tendencia local com encolhimento pequeno AJUDA, e o ganho aparece
    nos tres cortes com o mesmo sinal (k=0.2: 1.8323/1.8451/1.8804 contra
    1.8393/1.8551/1.8839 em k=0). Consistencia entre periodos
    independentes eh o que distingue efeito real de ruido.
  - k grande destroi: k=1.0 leva o RMSE a 2.03. A inclinacao crua eh
    quase toda ruido amostral; aproveita-se a direcao, nao a amplitude.
  - a mistura entre janela curta e longa reproduziu o que ja estava em
    producao, sem ganho adicional.
  - 01c testou as duas coisas SEPARADAMENTE e gerou a submissao so com a
    mistura - por isso o arquivo de 01c nao trazia o ganho.

POR QUE O CORTE 2012 PESA MAIS:
  Os tres cortes discordam: 2002 prefere janelas longas, 2012 e 2007
  preferem 20-24 anos. O corte 2012 avalia 2013-2022, o periodo com mais
  deriva termica acumulada - eh o analogo mais proximo de prever 2023/24.
  A busca aqui usa media ponderada: peso 2 em 2012, 1.5 em 2007, 1 em 2002.

FORMA FINAL:
    previsao(ano, mes, ponto) =
        (1-w) * clim_longa[mes, ponto]
      +   w   * clim_curta[mes, ponto]
      +   k   * inclinacao[mes, ponto] * (ano - ano_central)

Uso:
    python src/01d_climatologia_final.py
"""

import numpy as np
import pandas as pd
import xarray as xr

DADOS = "/prj/cptec/alex.campos/satrain/previsao_precipitacao_america_sul/dados"
SAIDA = "dados_processados"
SUBS = "saidas"

CORTES = {2012: ((2013, 2022), 2.0),
          2007: ((2008, 2017), 1.5),
          2002: ((2003, 2012), 1.0)}

GRADE_N = (20, 24, 28)
GRADE_W = (0.4, 0.5, 0.6, 0.7)
GRADE_K = (0.0, 0.10, 0.15, 0.20, 0.25)


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
print(f"serie: {tp.sizes['time']} meses")

_cache = {}


def clim(ini, fim):
    ch = (ini, fim)
    if ch not in _cache:
        _cache[ch] = (tp.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
                        .groupby("time.month").mean("time")
                        .transpose("month", "lat", "lon").values)
    return _cache[ch]


def tendencia(ini, fim):
    """Inclinacao por (mes, ponto) e ano central da janela."""
    ch = ("t", ini, fim)
    if ch not in _cache:
        sub = tp.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
        incls = []
        for m in range(1, 13):
            sm = sub.sel(time=sub.time.dt.month == m)
            x = sm.time.dt.year.values.astype("float64")
            xc = x - x.mean()
            y = sm.values
            incls.append(np.tensordot(xc, y - y.mean(0), axes=(0, 0))
                         / (xc ** 2).sum())
        _cache[ch] = (np.stack(incls), float(x.mean()))
    return _cache[ch]


def rmse(nivel, incl, centro, k, ini, fim):
    erros = []
    for ano in range(ini, fim + 1):
        p = nivel + k * incl * (ano - centro)
        obs = tp.sel(time=slice(f"{ano}-01-01", f"{ano}-12-31"))
        m = obs.time.dt.month.values
        erros.append(((obs.values - p[m - 1]) ** 2).ravel())
    return float(np.sqrt(np.mean(np.concatenate(erros))))


# ---------------------------------------------------------------------------
sep("1. BUSCA CONJUNTA (janela, mistura, tendencia)")

print("  Media ponderada dos cortes: 2012 pesa 2.0, 2007 pesa 1.5, 2002 pesa 1.0\n")
print(f"  {'n':>4s} {'w':>5s} {'k':>5s}" + "".join(f"{c:>10d}" for c in CORTES)
      + f"{'ponderada':>12s}")

resultados = []
for n in GRADE_N:
    for w in GRADE_W:
        for k in GRADE_K:
            vals, pesos = [], []
            for corte, ((a, b), peso) in CORTES.items():
                ini = max(1940, corte - n + 1)
                nivel = (1 - w) * clim(1940, corte) + w * clim(ini, corte)
                incl, centro = tendencia(ini, corte)
                vals.append(rmse(nivel, incl, centro, k, a, b))
                pesos.append(peso)
            pond = float(np.average(vals, weights=pesos))
            resultados.append((pond, n, w, k, vals))

resultados.sort()
for pond, n, w, k, vals in resultados[:12]:
    print(f"  {n:4d} {w:5.2f} {k:5.2f}" + "".join(f"{v:10.4f}" for v in vals)
          + f"{pond:12.4f}")

pond, n_ot, w_ot, k_ot, vals_ot = resultados[0]
print(f"\n  >> melhor: janela {n_ot} anos, w={w_ot:.2f}, k={k_ot:.2f}")
print(f"     ponderada {pond:.4f}, corte 2012 {vals_ot[0]:.4f}")
print(f"     configuracao em producao: 1.8406 no corte 2012")
print(f"     ganho no corte 2012: {100*(1.8406 - vals_ot[0])/1.8406:+.2f}%")


# ---------------------------------------------------------------------------
sep("2. ESTABILIDADE DA ESCOLHA")

print("""  Se o otimo for uma agulha, provavelmente eh ruido. Se for um
  plato, eh propriedade do problema. Vizinhanca do melhor ponto:
""")

melhor = {(n, w, k): p for p, n, w, k, _ in resultados}
print(f"  {'k':>6s}" + "".join(f"{w:>10.2f}" for w in GRADE_W))
for k in GRADE_K:
    linha = [melhor.get((n_ot, w, k), np.nan) for w in GRADE_W]
    marca = "  <--" if k == k_ot else ""
    print(f"  {k:6.2f}" + "".join(f"{v:10.4f}" for v in linha) + marca)

viz = [p for p, n, w, k, _ in resultados
       if n == n_ot and abs(w - w_ot) <= 0.1 and abs(k - k_ot) <= 0.05]
print(f"\n  vizinhanca imediata: {min(viz):.4f} a {max(viz):.4f} "
      f"(amplitude {max(viz)-min(viz):.4f})")
if max(viz) - min(viz) < 0.003:
    print("  >> plato: a escolha eh robusta.")
else:
    print("  >> superficie inclinada: usar o valor mais conservador de k.")


# ---------------------------------------------------------------------------
sep("3. CONFIGURACAO FINAL E SANIDADE")

ini_ot = 2022 - n_ot + 1
nivel = (1 - w_ot) * clim(1940, 2022) + w_ot * clim(ini_ot, 2022)
incl, centro = tendencia(ini_ot, 2022)

print(f"  janela curta : {ini_ot}-2022 ({n_ot} anos)")
print(f"  mistura      : {w_ot:.2f} curta + {1-w_ot:.2f} longa (1940-2022)")
print(f"  tendencia    : k={k_ot:.2f}, centrada em {centro:.1f}")
print(f"  extrapolacao : {2023.5 - centro:.1f} anos alem do centro")

clim_da = xr.DataArray(nivel, dims=("month", "lat", "lon"),
                       coords={"month": np.arange(1, 13),
                               "lat": tp.lat, "lon": tp.lon})
incl_da = xr.DataArray(incl, dims=("month", "lat", "lon"),
                       coords={"month": np.arange(1, 13),
                               "lat": tp.lat, "lon": tp.lon})

locais = {"Manaus": (-3.1, -60.0), "Sao Paulo": (-23.5, -46.6),
          "Atacama": (-23.0, -69.0), "Belem": (-1.5, -48.5),
          "Porto Alegre": (-30.0, -51.2)}
print(f"\n  {'local':14s} {'jan':>7s} {'jul':>7s} {'tend jan':>10s}")
for nome, (la, lo) in locais.items():
    j = float(clim_da.sel(month=1, lat=la, lon=lo, method="nearest"))
    ju = float(clim_da.sel(month=7, lat=la, lon=lo, method="nearest"))
    t = float(incl_da.sel(month=1, lat=la, lon=lo, method="nearest"))
    print(f"  {nome:14s} {j:7.2f} {ju:7.2f} {t*10:10.3f}")
print("  (tendencia em mm/dia por decada, janeiro)")


# ---------------------------------------------------------------------------
sep("4. GERANDO A SUBMISSAO")

sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
partes = sub["id"].str.split("_", expand=True)
ano = partes[0].astype(int).values
mes = partes[1].astype(int).values
lat = partes[2].astype(float).values
lon = partes[3].astype(float).values

ilat = np.rint((lat - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon - float(tp.lon[0])) / 0.25).astype(int)
assert np.allclose(tp.lat.values[ilat], lat), "indice de lat errado"
assert np.allclose(tp.lon.values[ilon], lon), "indice de lon errado"

pred = (nivel[mes - 1, ilat, ilon]
        + k_ot * incl[mes - 1, ilat, ilon] * (ano - centro))

neg = int((pred < 0).sum())
if neg:
    print(f"  {neg:,} previsoes negativas ({100*neg/len(pred):.3f}%) "
          "truncadas em zero")
    print("  (a tendencia pode puxar pontos aridos abaixo de zero)")
    pred = np.maximum(pred, 0.0)

assert np.isfinite(pred).all()
print(f"  previsoes: media {pred.mean():.4f}, max {pred.max():.2f}")

sub["tp_mm_day"] = pred
caminho = f"{SUBS}/sub03_clim_tendencia.csv"
sub.to_csv(caminho, index=False, float_format="%.4f")

np.savez(f"{SAIDA}/climatologia_v3.npz", nivel=nivel, incl=incl,
         centro=centro, k=k_ot, n=n_ot, w=w_ot)

esperado = 1.8395 * vals_ot[0] / 1.8406
print(f"""
  salvo: {caminho}
  primeira linha: {sub.id.iloc[0]},{sub.tp_mm_day.iloc[0]:.4f}

  EXPECTATIVA: a versao anterior marcou 1.8395. Com ganho de
  {100*(1.8406 - vals_ot[0])/1.8406:.2f}% no corte 2012, espera-se algo em torno
  de {esperado:.4f}.

  ANOTE ESSE NUMERO antes de enviar. Se o placar vier proximo, o ganho
  se transferiu para 2023 e a metodologia de validacao esta calibrada.
  Se vier pior, a tendencia ajustada ate 2022 nao vale para 2023 - e
  isso tambem eh resultado, vai para o README.""")