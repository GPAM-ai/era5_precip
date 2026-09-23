#!/usr/bin/env python
"""
Etapa 16 - Alfa por mes calendario.

A calibracao global eh 1.016 (14_avaliacao), mas a habilidade varia com
a estacao: r de 0.58 em fevereiro e 0.42 em agosto. Um alfa unico eh o
compromisso entre os dois. Aqui o alfa eh ajustado por mes-alvo nas
previsoes fora da amostra (2.6M linhas de 1998-2022, ja calculadas) e
aplicado a submissao mais recente.

  alfa_m = argmin sum (y - a * pred)^2  =  <pred, y> / <pred, pred>
           para as linhas do mes m

Ganho esperado: pequeno (0.1-0.2% no CV). O que se verifica:
  - se os alfas por mes ficarem todos entre 0.9 e 1.1, nao vale a pena
    (ja esta calibrado) e o script diz isso;
  - se a estacao seca pedir alfa claramente menor, aplica.

Uso:
    python src/16_alfa_mensal.py saidas/sub07_mme3_norec_s5.csv
"""

import sys

import numpy as np
import pandas as pd

SAIDA = "dados_processados"
DADOS = "dados_brutos"

arq_sub = sys.argv[1] if len(sys.argv) > 1 else f"saidas/sub07_mme3_norec_s5.csv"


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


sep("1. ALFA POR MES NAS PREVISOES FORA DA AMOSTRA")

oof = pd.read_parquet(f"{SAIDA}/avaliacao_oof.parquet")
print(f"  {len(oof):,} previsoes, {oof.ano.min()}-{oof.ano.max()}")

alfa_g = float(np.dot(oof.pred, oof.y) / np.dot(oof.pred, oof.pred))
rmse_g = float(np.sqrt(np.mean((oof.y - alfa_g * oof.pred) ** 2)))
rmse_1 = float(np.sqrt(np.mean((oof.y - oof.pred) ** 2)))

alfas = {}
nomes = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
print(f"\n  alfa global: {alfa_g:.3f}\n")
print(f"  {'mes':>4s} {'alfa':>7s} {'r':>7s} {'rmse a=1':>9s} {'rmse a_m':>9s}")
for m in range(1, 13):
    s = oof[oof.mes == m]
    a = float(np.dot(s.pred, s.y) / np.dot(s.pred, s.pred))
    alfas[m] = a
    r = float(np.corrcoef(s.pred, s.y)[0, 1])
    r1 = float(np.sqrt(np.mean((s.y - s.pred) ** 2)))
    ra = float(np.sqrt(np.mean((s.y - a * s.pred) ** 2)))
    print(f"  {nomes[m-1]:>4s} {a:7.3f} {r:7.3f} {r1:9.4f} {ra:9.4f}")

# RMSE agregado com alfa por mes (ajustado in-sample; para ser honesto,
# valida cruzadamente por ano: alfa de cada mes ajustado sem o ano avaliado)
pred_m = oof.pred * oof.mes.map(alfas)
rmse_m = float(np.sqrt(np.mean((oof.y - pred_m) ** 2)))

anos = sorted(oof.ano.unique())
erros = []
for a in anos:
    tr, te = oof[oof.ano != a], oof[oof.ano == a]
    al = {m: float(np.dot(tr[tr.mes == m].pred, tr[tr.mes == m].y)
                   / np.dot(tr[tr.mes == m].pred, tr[tr.mes == m].pred)) for m in range(1, 13)}
    erros.append(((te.y - te.pred * te.mes.map(al)) ** 2).values)
rmse_cv = float(np.sqrt(np.mean(np.concatenate(erros))))

sep("2. VALE A PENA?")
print(f"  RMSE fora da amostra, alfa = 1          : {rmse_1:.4f}")
print(f"  RMSE fora da amostra, alfa global       : {rmse_g:.4f}")
print(f"  RMSE fora da amostra, alfa por mes      : {rmse_m:.4f}  (ajustado in-sample)")
print(f"  RMSE fora da amostra, alfa por mes (LOYO): {rmse_cv:.4f}  (honesto)")
ganho = 100 * (1 - rmse_cv / rmse_g)
print(f"\n  ganho do alfa mensal sobre o global, validado: {ganho:+.3f}%")

amplitude = max(alfas.values()) - min(alfas.values())
print(f"  amplitude dos alfas: {min(alfas.values()):.3f} a {max(alfas.values()):.3f}")

if ganho < 0.05:
    print("""
  >> ganho abaixo de 0.05%: dentro do ruido. O modelo ja modula a
     amplitude por mes internamente (recebe sin/cos do mes e a
     climatologia). Nao aplicar; registrar no README.""")
    sys.exit(0)

sep("3. APLICANDO A SUBMISSAO")

sub = pd.read_csv(arq_sub)
partes = sub["id"].str.split("_", expand=True)
mes = partes[1].astype(int).values

# a submissao tem clim + alfa_global * anom; recupera a anomalia e reaplica
# precisa da climatologia v3 por linha - reconstroi como em 06
import xarray as xr
v3 = np.load(f"{SAIDA}/climatologia_v3.npz")
tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp
lat_v = partes[2].astype(float).values
lon_v = partes[3].astype(float).values
ano_v = partes[0].astype(int).values
ilat = np.rint((lat_v - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon_v - float(tp.lon[0])) / 0.25).astype(int)
ipt = ilat * len(tp.lon) + ilon
nivel = v3["nivel"].reshape(12, -1); incl = v3["incl"].reshape(12, -1)
base = nivel[mes - 1, ipt] + float(v3["k"]) * incl[mes - 1, ipt] * (ano_v - float(v3["centro"]))

# alfa usado na submissao original: le do pkl mais recente
import pickle, glob, os
pk = sorted(glob.glob(f"{SAIDA}/modelo_final_mme3*.pkl"), key=os.path.getmtime)[-1]
alfa_usado = pickle.load(open(pk, "rb"))["alfa"]
print(f"  alfa da submissao original: {alfa_usado:.3f} ({os.path.basename(pk)})")

anom = (sub["tp_mm_day"].values - base) / alfa_usado
# onde a submissao foi truncada em zero, a anomalia recuperada esta errada;
# nesses pontos mantem o valor original
truncado = sub["tp_mm_day"].values <= 0
fator = np.array([alfas[m] for m in mes]) / 1.0
novo = base + anom * fator
novo = np.where(truncado, sub["tp_mm_day"].values, np.maximum(novo, 0))

out = arq_sub.replace(".csv", "_alfames.csv")
sub["tp_mm_day"] = novo
sub.to_csv(out, index=False, float_format="%.4f")
print(f"  salvo: {out}")
print(f"  diferenca media absoluta vs original: {np.mean(np.abs(novo - pd.read_csv(arq_sub).tp_mm_day.values)):.4f} mm/dia")
