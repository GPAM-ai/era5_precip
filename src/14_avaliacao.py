#!/usr/bin/env python
"""
Etapa 14 - Avaliacao do modelo final: o que o RMSE agregado nao mostra.

Roda o walk-forward (mesmo desenho de 06) guardando as previsoes fora
da amostra de cada fold, e responde:

  1. OVERFITTING   RMSE dentro do treino vs fora. A diferenca eh a
                   medida direta de quanto o modelo decorou.
  2. POR ANO       o ganho esta espalhado ou concentrado em poucos anos?
  3. POR REGIME    El Nino / neutro / La Nina - o modelo depende do ENOS?
  4. POR REGIAO    onde ajuda e onde nao ajuda.
  5. POR MES       habilidade sazonal.
  6. CALIBRACAO    anomalia prevista vs observada, por faixa. O alfa~1
                   sugere calibracao; aqui se ve se vale por faixa.
  7. EXTREMOS      RMSE nos meses/pontos mais umidos e mais secos.
  8. OUTRAS METRICAS  MAE, vies, correlacao, MSSS, e habilidade
                   categorica (acerta o tercil?).

Saidas: dados_processados/avaliacao_oof.parquet (previsoes fora da
amostra, para figuras) e as tabelas impressas.

Uso:
    FEATURES=mme3 python -u src/14_avaliacao.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

DADOS = "dados_brutos"
SAIDA = "dados_processados"
FEATURES = os.environ.get("FEATURES", "mme3")
sufixo = f"_{FEATURES}" if FEATURES else ""

N_JANELA, W_MIX, K_TEND = 24, 0.50, 0.15
CORTES = [1997, 2002, 2007, 2012, 2017]
N_AJUSTE = 2_500_000
SEMENTE = 42
PARAMS = dict(objective="l2", num_leaves=63, learning_rate=0.05,
              min_child_samples=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
              n_estimators=400, n_jobs=-1, verbose=-1, random_state=SEMENTE)

EL_NINO = [1998, 2003, 2007, 2010, 2016, 2019]
LA_NINA = [1999, 2000, 2001, 2008, 2011, 2012, 2018, 2021, 2022]

REGIOES = {
    "Amazonia norte":  ((-5, 5), (-72, -55)),
    "Amazonia sul":    ((-15, -5), (-72, -50)),
    "Nordeste":        ((-15, -3), (-45, -35)),
    "Centro-Oeste/SE": ((-25, -15), (-58, -40)),
    "Sul":             ((-34, -25), (-58, -48)),
    "Andes/Peru":      ((-18, 0), (-82, -68)),
    "Argentina/Pampa": ((-40, -30), (-66, -55)),
    "Patagonia":       ((-55, -40), (-75, -63)),
    "Oceano":          None,
}


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def rmse(a, b=None):
    return float(np.sqrt(np.mean((a if b is None else a - b) ** 2)))


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

X = np.load(f"{SAIDA}/X_treino{sufixo}.npy")
y_anom = np.load(f"{SAIDA}/y_treino.npy")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
anos = meta["ano_feature"].values
mes_alvo = meta["mes_alvo"].values
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)
print(f"  X: {X.shape}")

tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
lat_sub, lon_sub = tr_ds.lat.values, tr_ds.lon.values
tr_ds.close()
n_pt = n_lat * n_lon
stride = (len(tp.lat) - 1) // (n_lat - 1)

clim_g = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")
clim_arr = (clim_g.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
            .transpose("month", "lat", "lon").values.reshape(12, n_pt))
n_t = len(y_anom) // n_pt
ipt = np.tile(np.arange(n_pt, dtype="int32"), n_t)
y_bruto = y_anom + clim_arr[mes_alvo - 1, ipt]

lat_row = lat_sub[ipt // n_lon]
lon_row = lon_sub[ipt % n_lon]

# mascara de terra: usa a propria climatologia como proxy (oceano tem
# climatologia anual muito suave); melhor: pontos com |lat| < 60 e dentro
# do continente aproximado. Simplificacao: terra = qualquer regiao nomeada.
tp_sub = tp.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
tp_np = tp_sub.transpose("time", "lat", "lon").values.reshape(len(tp_sub.time), -1)
anos_tp = tp_sub.time.dt.year.values
meses_tp = tp_sub.time.dt.month.values


def alvo_do_fold(corte):
    ini = max(1940, corte - N_JANELA + 1)

    def media(a, b):
        out = np.zeros((12, n_pt))
        m = (anos_tp >= a) & (anos_tp <= b)
        for k in range(12):
            out[k] = tp_np[m & (meses_tp == k + 1)].mean(0)
        return out

    nivel = (1 - W_MIX) * media(1940, corte) + W_MIX * media(ini, corte)
    incl = np.zeros((12, n_pt))
    for k in range(12):
        sel = (anos_tp >= ini) & (anos_tp <= corte) & (meses_tp == k + 1)
        x = anos_tp[sel].astype("float64"); xc = x - x.mean()
        v = tp_np[sel]
        incl[k] = (xc @ (v - v.mean(0))) / (xc ** 2).sum()
    centro = float(anos_tp[(anos_tp >= ini) & (anos_tp <= corte)].mean())
    base = nivel[mes_alvo - 1, ipt] + K_TEND * incl[mes_alvo - 1, ipt] * (ano_alvo - centro)
    return (y_bruto - base).astype("float32"), base.astype("float32")


# ---------------------------------------------------------------------------
sep("2. WALK-FORWARD GUARDANDO PREVISOES FORA DA AMOSTRA")

rng = np.random.default_rng(SEMENTE)
oof = []          # (idx, y_anom_v3, pred, base)
dentro = []

for corte in CORTES:
    t0 = time.time()
    yf, base = alvo_do_fold(corte)
    m_fit = anos <= corte
    m_val = (anos >= corte + 1) & (anos <= corte + 5)
    idx_fit = np.where(m_fit)[0]
    if len(idx_fit) > N_AJUSTE:
        idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
    idx_val = np.where(m_val)[0]

    mod = lgb.LGBMRegressor(**PARAMS)
    mod.fit(X[idx_fit], yf[idx_fit])

    pv = mod.predict(X[idx_val]).astype("float32")
    pf = mod.predict(X[idx_fit[:500_000]]).astype("float32")    # amostra do treino

    r_in = rmse(yf[idx_fit[:500_000]], pf)
    r_in_c = rmse(yf[idx_fit[:500_000]])
    r_out = rmse(yf[idx_val], pv)
    r_out_c = rmse(yf[idx_val])
    dentro.append(dict(corte=corte, rmse_treino=r_in, ganho_treino=100 * (1 - r_in / r_in_c),
                       rmse_val=r_out, ganho_val=100 * (1 - r_out / r_out_c),
                       r_treino=float(np.corrcoef(pf, yf[idx_fit[:500_000]])[0, 1]),
                       r_val=float(np.corrcoef(pv, yf[idx_val])[0, 1])))
    oof.append(pd.DataFrame({"idx": idx_val, "y": yf[idx_val], "pred": pv,
                             "base": base[idx_val], "fold": corte}))
    print(f"  fold {corte}: treino ganho {dentro[-1]['ganho_treino']:+.2f}% "
          f"(r {dentro[-1]['r_treino']:.3f}) | validacao {dentro[-1]['ganho_val']:+.2f}% "
          f"(r {dentro[-1]['r_val']:.3f})  ({time.time()-t0:.0f}s)", flush=True)

oof = pd.concat(oof, ignore_index=True)
oof["ano"] = ano_alvo[oof.idx.values]
oof["mes"] = mes_alvo[oof.idx.values]
oof["lat"] = lat_row[oof.idx.values]
oof["lon"] = lon_row[oof.idx.values]
oof["clim"] = oof.base
oof["obs"] = oof.y + oof.base
oof["prev"] = oof.pred + oof.base
oof.to_parquet(f"{SAIDA}/avaliacao_oof.parquet")
print(f"\n  {len(oof):,} previsoes fora da amostra, {oof.ano.min()}-{oof.ano.max()}")


# ---------------------------------------------------------------------------
sep("3. OVERFITTING: DENTRO vs FORA DA AMOSTRA")

d = pd.DataFrame(dentro)
print(f"  {'fold':>6s} {'ganho treino':>13s} {'ganho valid':>12s} {'r treino':>9s} {'r valid':>8s} {'gap r':>7s}")
for _, r in d.iterrows():
    print(f"  {int(r.corte):6d} {r.ganho_treino:+12.2f}% {r.ganho_val:+11.2f}% "
          f"{r.r_treino:9.3f} {r.r_val:8.3f} {r.r_treino - r.r_val:7.3f}")
print(f"\n  media: treino r {d.r_treino.mean():.3f}, validacao r {d.r_val.mean():.3f}, "
      f"gap {d.r_treino.mean() - d.r_val.mean():.3f}")
print("""
  Leitura: gap de r entre 0.05 e 0.15 eh o normal para gradient boosting
  regularizado em dado ruidoso. Acima de 0.25 seria memorizacao. Gap
  perto de zero indicaria subajuste.""")


# ---------------------------------------------------------------------------
sep("4. HABILIDADE POR ANO")


def tab(grupo, chave):
    g = oof.groupby(chave)
    out = pd.DataFrame({
        "n": g.size(),
        "rmse_clim": g.apply(lambda s: rmse(s.y)),
        "rmse_mod": g.apply(lambda s: rmse(s.y, s.pred)),
        "r": g.apply(lambda s: np.corrcoef(s.pred, s.y)[0, 1]),
        "vies": g.apply(lambda s: float((s.pred - s.y).mean())),
    })
    out["ganho_%"] = 100 * (1 - out.rmse_mod / out.rmse_clim)
    return out


por_ano = tab(oof, "ano")
print(f"  {'ano':>5s} {'rmse clim':>10s} {'rmse mod':>9s} {'ganho':>8s} {'r':>7s} {'vies':>7s}")
for a, r in por_ano.iterrows():
    reg = "EN" if a in EL_NINO else ("LN" if a in LA_NINA else "  ")
    print(f"  {a:5d} {r.rmse_clim:10.3f} {r.rmse_mod:9.3f} {r['ganho_%']:+7.2f}% {r.r:7.3f} "
          f"{r.vies:+7.3f}  {reg}")
print(f"\n  anos com ganho positivo: {int((por_ano['ganho_%'] > 0).sum())} de {len(por_ano)}")
print(f"  ganho: media {por_ano['ganho_%'].mean():.2f}%, mediana {por_ano['ganho_%'].median():.2f}%, "
      f"min {por_ano['ganho_%'].min():.2f}%, max {por_ano['ganho_%'].max():.2f}%")
print("""
  Leitura: se o ganho e positivo em quase todos os anos e a mediana fica
  perto da media, a habilidade eh espalhada e nao depende de tres ou
  quatro anos excepcionais.""")


# ---------------------------------------------------------------------------
sep("5. POR REGIME DE ENOS")

oof["regime"] = np.where(oof.ano.isin(EL_NINO), "El Nino",
                         np.where(oof.ano.isin(LA_NINA), "La Nina", "neutro"))
por_reg = tab(oof, "regime")
print(f"  {'regime':10s} {'anos':>5s} {'rmse clim':>10s} {'rmse mod':>9s} {'ganho':>8s} {'r':>7s}")
for k, r in por_reg.iterrows():
    n_anos = oof[oof.regime == k].ano.nunique()
    print(f"  {k:10s} {n_anos:5d} {r.rmse_clim:10.3f} {r.rmse_mod:9.3f} {r['ganho_%']:+7.2f}% {r.r:7.3f}")
print("""
  Leitura: ganho maior em El Nino eh esperado (sinal forte). O que
  importa eh que La Nina e neutro tambem sejam positivos - 2024 eh
  transicao para La Nina.""")


# ---------------------------------------------------------------------------
sep("6. POR REGIAO")

def regiao_de(la, lo):
    for nome, caixa in REGIOES.items():
        if caixa is None:
            continue
        (la0, la1), (lo0, lo1) = caixa
        if la0 <= la <= la1 and lo0 <= lo <= lo1:
            return nome
    return "Oceano/outro"

oof["regiao"] = [regiao_de(a, b) for a, b in zip(oof.lat.values, oof.lon.values)]
por_regiao = tab(oof, "regiao").sort_values("ganho_%", ascending=False)
print(f"  {'regiao':18s} {'pontos':>7s} {'rmse clim':>10s} {'rmse mod':>9s} {'ganho':>8s} {'r':>7s} {'vies':>7s}")
for k, r in por_regiao.iterrows():
    print(f"  {k:18s} {int(r.n)//len(oof.ano.unique())//12:7d} {r.rmse_clim:10.3f} {r.rmse_mod:9.3f} "
          f"{r['ganho_%']:+7.2f}% {r.r:7.3f} {r.vies:+7.3f}")
print("""
  Leitura: o RMSE eh dominado por onde chove muito (Amazonia, ZCIT). Uma
  regiao com ganho pequeno mas RMSE alto pesa mais no total do que uma
  com ganho grande e RMSE baixo.""")


# ---------------------------------------------------------------------------
sep("7. POR MES CALENDARIO")

por_mes = tab(oof, "mes")
nomes_mes = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
print(f"  {'mes':>4s} {'rmse clim':>10s} {'rmse mod':>9s} {'ganho':>8s} {'r':>7s}")
for m, r in por_mes.iterrows():
    print(f"  {nomes_mes[m-1]:>4s} {r.rmse_clim:10.3f} {r.rmse_mod:9.3f} {r['ganho_%']:+7.2f}% {r.r:7.3f}")


# ---------------------------------------------------------------------------
sep("8. CALIBRACAO: PREVISTO vs OBSERVADO POR FAIXA")

print("""  Divide as previsoes de anomalia em decis e compara a media prevista
  com a media observada em cada decil. Modelo calibrado: os dois
  crescem juntos com inclinacao ~1. Inclinacao < 1 = superestima a
  amplitude; > 1 = subestima.
""")
oof["decil"] = pd.qcut(oof.pred, 10, labels=False)
cal = oof.groupby("decil").agg(prev=("pred", "mean"), obs=("y", "mean"), n=("y", "size"))
print(f"  {'decil':>6s} {'prev media':>11s} {'obs media':>10s} {'razao':>7s}")
for k, r in cal.iterrows():
    razao = r.obs / r.prev if abs(r.prev) > 0.05 else float("nan")
    print(f"  {k+1:6d} {r.prev:+11.3f} {r.obs:+10.3f} {razao:7.2f}")
incl = np.polyfit(cal.prev, cal.obs, 1)[0]
print(f"\n  inclinacao obs ~ prev: {incl:.3f}  (1.0 = perfeitamente calibrado)")


# ---------------------------------------------------------------------------
sep("9. EXTREMOS")

print("""  Os 10% de (ponto, mes) com anomalia observada mais positiva e mais
  negativa. Eh onde a previsao tem valor pratico - enchente e seca - e
  onde modelos de media tendem a falhar.
""")
q10, q90 = oof.y.quantile([0.10, 0.90])
for rotulo, m in (("10% mais secos", oof.y <= q10), ("10% mais umidos", oof.y >= q90),
                  ("80% do meio", (oof.y > q10) & (oof.y < q90))):
    s = oof[m]
    print(f"  {rotulo:16s} n={len(s):>8,}  rmse clim {rmse(s.y):.3f}  "
          f"rmse mod {rmse(s.y, s.pred):.3f}  ganho {100*(1-rmse(s.y, s.pred)/rmse(s.y)):+.1f}%  "
          f"vies {float((s.pred - s.y).mean()):+.3f}")
print("""
  Leitura: vies negativo nos umidos e positivo nos secos eh o padrao de
  todo modelo que minimiza MSE - ele encolhe extremos em direcao a media.
  O que se quer eh que o ganho seja positivo mesmo assim.""")


# ---------------------------------------------------------------------------
sep("10. OUTRAS METRICAS (agregado fora da amostra)")

y, p = oof.y.values, oof.pred.values
mse_c, mse_m = np.mean(y ** 2), np.mean((y - p) ** 2)
print(f"  RMSE climatologia : {np.sqrt(mse_c):.4f} mm/dia")
print(f"  RMSE modelo       : {np.sqrt(mse_m):.4f} mm/dia")
print(f"  MAE climatologia  : {np.mean(np.abs(y)):.4f}")
print(f"  MAE modelo        : {np.mean(np.abs(y - p)):.4f}")
print(f"  vies medio        : {np.mean(p - y):+.4f}")
print(f"  correlacao        : {np.corrcoef(p, y)[0, 1]:.4f}")
print(f"  MSSS              : {1 - mse_m / mse_c:.4f}   (1 - MSE_mod/MSE_clim; 0 = climatologia)")

# habilidade categorica: acerta o tercil?
t_obs = pd.qcut(oof.y, 3, labels=[0, 1, 2]).astype(int)
t_prev = pd.qcut(oof.pred, 3, labels=[0, 1, 2]).astype(int)
acerto = float((t_obs == t_prev).mean())
print(f"  acerto de tercil  : {100*acerto:.1f}%   (aleatorio = 33.3%)")
extremo_ok = float(((t_obs == 0) & (t_prev == 0)).sum() / (t_obs == 0).sum())
extremo_ok2 = float(((t_obs == 2) & (t_prev == 2)).sum() / (t_obs == 2).sum())
print(f"  acerto no tercil seco : {100*extremo_ok:.1f}%   umido: {100*extremo_ok2:.1f}%")

print("""
  MSSS eh a metrica padrao em verificacao de previsao sazonal (Murphy
  1988). Valores de 0.15 a 0.30 sao considerados bons para precipitacao
  a um mes; a maioria dos centros operacionais reporta abaixo de 0.2.""")

resumo = dict(rmse_clim=float(np.sqrt(mse_c)), rmse_mod=float(np.sqrt(mse_m)),
              msss=float(1 - mse_m / mse_c), r=float(np.corrcoef(p, y)[0, 1]),
              gap_r_overfit=float(d.r_treino.mean() - d.r_val.mean()),
              anos_com_ganho=int((por_ano["ganho_%"] > 0).sum()), n_anos=int(len(por_ano)),
              inclinacao_calibracao=float(incl), acerto_tercil=acerto)
with open(f"{SAIDA}/avaliacao_resumo.json", "w") as f:
    json.dump(resumo, f, indent=2)
print(f"\n  salvo: avaliacao_oof.parquet, avaliacao_resumo.json")
