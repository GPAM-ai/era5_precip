#!/usr/bin/env python
"""
Etapa 18 - Sanidade das previsoes do teste SEM rotulos.

O placar publico valida 2023. O segundo semestre de 2024 tem um
componente que nunca passou por validacao: a costura IC3 -> IC4 do
CanSIPS, que entra em julho de 2024. Se houver salto de escala ali, a
metade do conjunto privado paga e nada no publico avisa.

Tudo aqui usa so as previsoes e as features do teste. Nenhuma
observacao de 2023 ou 2024 eh lida.

Checa:
  1. Anomalia mensal media e desvio do CanSIPS em 2024 - salto em julho?
  2. Anomalia mensal media e desvio da PREVISAO FINAL - 2024 parece 2023?
  3. Distribuicao espacial: pontos com previsao fora de [0, 3x clim]?
  4. Comparacao das duas finais: onde diferem mais?

Uso:
    python src/18_sanidade_2024.py
"""

import json

import numpy as np
import pandas as pd

SAIDA = "dados_processados"
FINAIS = ["saidas/sub07_mme3_norec_t1200l63_s5_mos.csv",
          "saidas/sub07_mme3_norec_t1200l63_s5.csv"]


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


with open(f"{SAIDA}/features_mme3.json") as f:
    nomes = json.load(f)["nomes"]
Xte = np.load(f"{SAIDA}/X_teste_mme3.npy", mmap_mode="r")
meta = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")
ano, mes = meta["ano"].values, meta["mes_alvo"].values
rot = [f"{a}-{m:02d}" for a, m in zip(ano, mes)]
meses_u = sorted(set(rot))

# ---------------------------------------------------------------------------
sep("1. CONTINUIDADE DA COSTURA IC3 -> IC4 (CanSIPS)")

ic = nomes.index("cansips_L05_anom")
c = np.asarray(Xte[:, ic])
print(f"  {'mes':>8s} {'media':>8s} {'desvio':>8s} {'p5':>8s} {'p95':>8s}")
for m in meses_u:
    s = c[np.array(rot) == m]
    print(f"  {m:>8s} {s.mean():+8.3f} {s.std():8.3f} {np.percentile(s,5):+8.3f} "
          f"{np.percentile(s,95):+8.3f}{'   <-- IC4 comeca' if m == '2024-07' else ''}")
jun = c[np.array(rot) == "2024-06"]; jul = c[np.array(rot) == "2024-07"]
print(f"\n  desvio jun/2024: {jun.std():.3f}   jul/2024: {jul.std():.3f}   "
      f"razao {jul.std()/jun.std():.2f}")
print("  (razao entre 0.7 e 1.4 eh variacao sazonal normal; fora disso, salto de escala)")

# compara com os outros sistemas no mesmo mes, para ter referencia
for outro in ("seas5_L05_anom", "cfsv2_L05_anom", "ukmo_L05_anom"):
    if outro in nomes:
        o = np.asarray(Xte[:, nomes.index(outro)])
        oj, ol = o[np.array(rot) == "2024-06"], o[np.array(rot) == "2024-07"]
        print(f"  {outro:18s} razao jul/jun = {ol.std()/oj.std():.2f}")

# ---------------------------------------------------------------------------
sep("2. PREVISAO FINAL POR MES: 2023 vs 2024")

sub = pd.read_csv(FINAIS[0])
partes = sub["id"].str.split("_", expand=True)
ano_s, mes_s = partes[0].astype(int).values, partes[1].astype(int).values
v3 = np.load(f"{SAIDA}/climatologia_v3.npz")
import xarray as xr
DADOS = "dados_brutos"
tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp
ilat = np.rint((partes[2].astype(float).values - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((partes[3].astype(float).values - float(tp.lon[0])) / 0.25).astype(int)
ipt = ilat * len(tp.lon) + ilon
nivel = v3["nivel"].reshape(12, -1); incl = v3["incl"].reshape(12, -1)
base = nivel[mes_s - 1, ipt] + float(v3["k"]) * incl[mes_s - 1, ipt] * (ano_s - float(v3["centro"]))
anom = sub["tp_mm_day"].values - base

print(f"  {'mes':>8s} {'anom media':>11s} {'anom desvio':>12s} {'prev media':>11s} {'max':>7s}")
for a in (2023, 2024):
    for m in range(1, 13):
        s = (ano_s == a) & (mes_s == m)
        print(f"  {a}-{m:02d} {anom[s].mean():+11.3f} {anom[s].std():12.3f} "
              f"{sub.tp_mm_day.values[s].mean():11.3f} {sub.tp_mm_day.values[s].max():7.1f}")
    print()
a23, a24 = anom[ano_s == 2023], anom[ano_s == 2024]
print(f"  2023: anomalia media {a23.mean():+.3f}, desvio {a23.std():.3f}")
print(f"  2024: anomalia media {a24.mean():+.3f}, desvio {a24.std():.3f}")
print("  (desvios parecidos = amplitude consistente; 2024 muito maior = suspeito)")

# ---------------------------------------------------------------------------
sep("3. VALORES FORA DE ESCALA")

razao = sub["tp_mm_day"].values / np.maximum(base, 0.05)
print(f"  previsao > 3x climatologia : {int((razao > 3).sum()):>7,} linhas "
      f"({100*(razao>3).mean():.2f}%)")
print(f"  previsao > 5x climatologia : {int((razao > 5).sum()):>7,} linhas")
print(f"  previsao == 0 (truncada)   : {int((sub.tp_mm_day.values == 0).sum()):>7,} linhas")
print(f"  NaN                        : {int(sub.tp_mm_day.isna().sum()):>7,}")
print(f"  maximo absoluto            : {sub.tp_mm_day.max():.1f} mm/dia")
for a in (2023, 2024):
    r = razao[ano_s == a]
    print(f"  {a}: > 3x clim em {100*(r>3).mean():.2f}% das linhas")

# ---------------------------------------------------------------------------
sep("4. AS DUAS FINAIS")

s1 = pd.read_csv(FINAIS[0]).tp_mm_day.values
s2 = pd.read_csv(FINAIS[1]).tp_mm_day.values
d = s1 - s2
print(f"  diferenca media absoluta: {np.abs(d).mean():.4f} mm/dia")
print(f"  correlacao entre as duas: {np.corrcoef(s1, s2)[0,1]:.5f}")
for a in (2023, 2024):
    print(f"  {a}: |diff| media {np.abs(d[ano_s == a]).mean():.4f}, max {np.abs(d[ano_s == a]).max():.2f}")
print("""
  Leitura: se 2024 diferir muito mais que 2023, o MOS pontual esta
  fazendo algo em 2024 que nao fez em 2023 - vale olhar onde.""")
