#!/usr/bin/env python
"""
Etapa 01c - Refinamento da climatologia.

CORRIGE UM BUG DE 01b:
  La estava escrito  pred = nivel + k * inclinacao * ano
  onde `nivel` era o intercepto no ano ZERO. Multiplicar apenas a
  inclinacao por k desloca o nivel inteiro em (1-k)*inclinacao*ano, que
  para ano=2013 eh um numero enorme - dai os RMSE absurdos de 50 e 76.

  A forma correta encolhe o DESVIO em relacao ao centro da janela:
      pred = media + k * inclinacao * (ano - ano_central)
  Com k=0 recupera-se a climatologia simples; com k=1, a tendencia cheia.

O QUE 01b JA MOSTROU:
  - janela de 20 anos vence nos dois cortes (1.8374 e 1.8473), batendo a
    mistura em uso (1.8406). Curva em U bem definida.
  - tendencia cheia (k=1) piora muito: 2.01 contra 1.84 da base. As
    inclinacoes locais sao majoritariamente ruido amostral.

O QUE SE TESTA AQUI:
  1. busca fina do comprimento da janela, de 14 a 28 anos
  2. tendencia com encolhimento correto, k de 0 a 1
  3. mistura entre a janela curta otima e uma janela longa
  4. gera a submissao com a melhor configuracao

Uso:
    python src/01c_climatologia_refino.py
"""

import numpy as np
import pandas as pd
import xarray as xr

DADOS = "/prj/cptec/alex.campos/satrain/previsao_precipitacao_america_sul/dados"
SAIDA = "dados_processados"
SUBS = "saidas"

CORTES = {2012: (2013, 2022), 2007: (2008, 2017), 2002: (2003, 2012)}


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
print(f"serie: {tp.sizes['time']} meses, {tp.sizes['lat']}x{tp.sizes['lon']}")


def clim(ini, fim):
    return (tp.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
              .groupby("time.month").mean("time")
              .transpose("month", "lat", "lon").values)


def rmse(pred_mes, ini, fim):
    """pred_mes: (12, lat, lon) fixo, ou funcao ano -> (12, lat, lon)."""
    erros = []
    for ano in range(ini, fim + 1):
        obs = tp.sel(time=slice(f"{ano}-01-01", f"{ano}-12-31"))
        p = pred_mes(ano) if callable(pred_mes) else pred_mes
        m = obs.time.dt.month.values
        erros.append(((obs.values - p[m - 1]) ** 2).ravel())
    return float(np.sqrt(np.mean(np.concatenate(erros))))


# ---------------------------------------------------------------------------
sep("1. BUSCA FINA DO COMPRIMENTO DA JANELA")

print("  01b indicou 20 anos. Refina de 14 a 28 em tres cortes.\n")
print(f"  {'anos':>6s}" + "".join(f"{c:>12d}" for c in CORTES) + f"{'media':>12s}")

tabela = {}
for n in range(14, 29, 2):
    linha = []
    for corte, (a, b) in CORTES.items():
        c = clim(max(1940, corte - n + 1), corte)
        linha.append(rmse(c, a, b))
    tabela[n] = linha
    print(f"  {n:6d}" + "".join(f"{v:12.4f}" for v in linha)
          + f"{np.mean(linha):12.4f}")

n_otimo = min(tabela, key=lambda k: np.mean(tabela[k]))
print(f"\n  >> comprimento otimo: {n_otimo} anos "
      f"(media {np.mean(tabela[n_otimo]):.4f})")


# ---------------------------------------------------------------------------
sep("2. TENDENCIA LOCAL, COM ENCOLHIMENTO CORRETO")

print("""  pred(ano) = media + k * inclinacao * (ano - ano_central)

  k=0 recupera a climatologia simples. Se nenhum k>0 melhorar, a
  tendencia local eh ruido e a conclusao vai para o README.
""")

def ajuste_tendencia(ini, fim):
    """Devolve (media, inclinacao, ano_central) por mes e ponto."""
    sub = tp.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
    medias, incls = [], []
    for m in range(1, 13):
        sm = sub.sel(time=sub.time.dt.month == m)
        x = sm.time.dt.year.values.astype("float64")
        xc = x - x.mean()
        y = sm.values
        b = np.tensordot(xc, y - y.mean(0), axes=(0, 0)) / (xc ** 2).sum()
        medias.append(y.mean(0))
        incls.append(b)
    return np.stack(medias), np.stack(incls), float(x.mean())

print(f"  {'k':>6s}" + "".join(f"{c:>12d}" for c in CORTES) + f"{'media':>12s}")
for k in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
    linha = []
    for corte, (a, b) in CORTES.items():
        mu, bb, xc = ajuste_tendencia(max(1940, corte - n_otimo + 1), corte)
        linha.append(rmse(lambda ano: mu + k * bb * (ano - xc), a, b))
    print(f"  {k:6.2f}" + "".join(f"{v:12.4f}" for v in linha)
          + f"{np.mean(linha):12.4f}")


# ---------------------------------------------------------------------------
sep("3. MISTURA ENTRE JANELA CURTA E LONGA")

print(f"""  A janela de {n_otimo} anos responde rapido mas tem ruido amostral;
  a longa eh estavel mas defasada. Uma mistura pode ganhar das duas,
  como ja aconteceu em 01.
""")

print(f"  {'w curta':>9s}" + "".join(f"{c:>12d}" for c in CORTES) + f"{'media':>12s}")
melhor = (1e9, None)
for w in np.arange(0.0, 1.01, 0.1):
    linha = []
    for corte, (a, b) in CORTES.items():
        curta = clim(max(1940, corte - n_otimo + 1), corte)
        longa = clim(1940, corte)
        linha.append(rmse((1 - w) * longa + w * curta, a, b))
    m = np.mean(linha)
    if m < melhor[0]:
        melhor = (m, w)
    print(f"  {w:9.2f}" + "".join(f"{v:12.4f}" for v in linha) + f"{m:12.4f}")

print(f"\n  >> melhor mistura: w={melhor[1]:.2f} com media {melhor[0]:.4f}")
print(f"     (a configuracao atual em producao da 1.8406 no corte 2012)")


# ---------------------------------------------------------------------------
sep("4. GERANDO A SUBMISSAO COM A MELHOR CONFIGURACAO")

w = melhor[1]
curta = clim(2022 - n_otimo + 1, 2022)
longa = clim(1940, 2022)
final = (1 - w) * longa + w * curta

print(f"  janela curta: {2022 - n_otimo + 1}-2022 ({n_otimo} anos)")
print(f"  peso        : {w:.2f} curta + {1-w:.2f} longa")
print(f"  media global: {final.mean():.4f} mm/dia")

# sanidade fisica, a mesma de 01
clim_da = xr.DataArray(final, dims=("month", "lat", "lon"),
                       coords={"month": np.arange(1, 13),
                               "lat": tp.lat, "lon": tp.lon})
locais = {"Manaus": (-3.1, -60.0), "Sao Paulo": (-23.5, -46.6),
          "Atacama": (-23.0, -69.0), "Belem": (-1.5, -48.5)}
print(f"\n  {'local':12s} {'janeiro':>9s} {'julho':>9s}")
for nome, (la, lo) in locais.items():
    j = float(clim_da.sel(month=1, lat=la, lon=lo, method="nearest"))
    ju = float(clim_da.sel(month=7, lat=la, lon=lo, method="nearest"))
    print(f"  {nome:12s} {j:9.2f} {ju:9.2f}")

clim_da.to_netcdf(f"{SAIDA}/climatologia_tp_v2.nc")

sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
partes = sub["id"].str.split("_", expand=True)
mes = partes[1].astype(int).values
lat = partes[2].astype(float).values
lon = partes[3].astype(float).values

ilat = np.rint((lat - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon - float(tp.lon[0])) / 0.25).astype(int)
assert np.allclose(tp.lat.values[ilat], lat)
assert np.allclose(tp.lon.values[ilon], lon)

pred = final[mes - 1, ilat, ilon]
assert np.isfinite(pred).all() and (pred >= 0).all()

sub["tp_mm_day"] = pred
sub.to_csv(f"{SUBS}/sub02_climatologia_v2.csv", index=False,
           float_format="%.4f")
print(f"\n  salvo: {SUBS}/sub02_climatologia_v2.csv  ({len(sub):,} linhas)")
print(f"  primeira linha: {sub.id.iloc[0]},{sub.tp_mm_day.iloc[0]:.4f}")

print(f"""
  Expectativa: a versao anterior marcou 1.8395 no placar. O ganho
  esperado eh de ~{100*(1.8406 - melhor[0])/1.8406:.2f}%, entao algo em torno de
  {1.8395 * melhor[0] / 1.8406:.4f}. Se vier muito diferente, o ganho nao
  se transferiu para 2023.""")