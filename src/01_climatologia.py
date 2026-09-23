#!/usr/bin/env python
"""
Etapa 01 - Climatologia e submissao baseline.

Faz tres coisas:
  1. calcula a climatologia de precipitacao por (lat, lon, mes calendario)
     em duas janelas candidatas, e escolhe entre elas com validacao honesta
  2. salva a climatologia escolhida em disco
  3. gera o CSV de submissao baseline (climatologia pura)

ALINHAMENTO TEMPORAL - o ponto critico deste problema:
  treino: a coordenada time eh o mes das FEATURES; o alvo (tp_alvo) eh o
          mes seguinte. Logo mes-alvo = mes(time) + 1.
  teste : a coordenada time (e o id do sample_submission) ja eh o mes-ALVO;
          as covariaveis daquela linha sao do mes anterior.
  Verificado em 00b: teste[time=2023-01].t2 == treino[time=2022-12].t2

  Toda climatologia e sazonalidade sao indexadas pelo MES-ALVO.

Uso:
    python src/01_climatologia.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

DADOS = "dados_brutos"
SAIDA = "dados_processados"
SUBS = "saidas"

os.makedirs(SAIDA, exist_ok=True)
os.makedirs(SUBS, exist_ok=True)

sep = lambda t: print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO PRECIPITACAO OBSERVADA")

tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
print(f"  tp: {dict(tp.sizes)}")
print(f"  periodo: {str(tp.time.values[0])[:7]} a {str(tp.time.values[-1])[:7]}")
print(f"  media global: {float(tp.mean()):.4f} mm/dia")
print(f"  NaN: {int(tp.isnull().sum())} (esperado 0)")


# ---------------------------------------------------------------------------
sep("2. ESCOLHA DA JANELA DE NORMAIS")

print("""  Duas candidatas:
    A) 1940-2022 (periodo inteiro)
    B) 1991-2020 (normais climatologicas padrao OMM)

  ERA5 antes de 1979 eh a extensao retroativa preliminar, com pouquissima
  observacao assimilada sobre a America do Sul. Incluir esse periodo pode
  contaminar a media. Por outro lado, mais anos reduzem o ruido amostral.

  Teste honesto: replicar a estrutura real do problema. A janela real
  termina em 2022 e preve 2023-2024. Entao aqui a janela termina em 2012
  e avaliamos em 2013-2022, que sao anos NAO vistos pela climatologia.""")

def climatologia(da, ano_ini, ano_fim):
    """Media por (mes calendario, lat, lon) no intervalo de anos dado."""
    sub = da.sel(time=slice(f"{ano_ini}-01-01", f"{ano_fim}-12-31"))
    return sub.groupby("time.month").mean("time")

def rmse_clim(da, clim, ano_ini, ano_fim):
    """RMSE da climatologia como previsao, no intervalo de avaliacao."""
    obs = da.sel(time=slice(f"{ano_ini}-01-01", f"{ano_fim}-12-31"))
    pred = clim.sel(month=obs.time.dt.month)
    err = obs - pred
    return float(np.sqrt((err ** 2).mean()))

clim_A_val = climatologia(tp, 1940, 2012)
clim_B_val = climatologia(tp, 1983, 2012)   # 30 anos, analogo a 1991-2020

rmse_A = rmse_clim(tp, clim_A_val, 2013, 2022)
rmse_B = rmse_clim(tp, clim_B_val, 2013, 2022)

print(f"\n  janela 1940-2012 -> RMSE em 2013-2022: {rmse_A:.4f} mm/dia")
print(f"  janela 1983-2012 -> RMSE em 2013-2022: {rmse_B:.4f} mm/dia")

# busca fina do peso otimo da mistura
pesos = np.arange(0.0, 1.01, 0.05)
rmses = []
for w in pesos:
    mix_val = (1 - w) * clim_A_val + w * clim_B_val
    rmses.append(rmse_clim(tp, mix_val, 2013, 2022))
    print(f"  w={w:.2f} -> RMSE {rmses[-1]:.4f}")

w_otimo = float(pesos[int(np.argmin(rmses))])
print(f"\n  >> peso otimo: {w_otimo:.2f} (RMSE {min(rmses):.4f})")

clim_full = climatologia(tp, 1940, 2022)
clim_rec = climatologia(tp, 1993, 2022)
clim = (1 - w_otimo) * clim_full + w_otimo * clim_rec
janela = f"mistura {1-w_otimo:.2f}*1940-2022 + {w_otimo:.2f}*1993-2022"

# terceira candidata: media ponderada das duas, caso fiquem empatadas
clim_full = climatologia(tp, 1940, 2022)
clim_rec = climatologia(tp, 1993, 2022)
for w in (0.25, 0.5, 0.75):
    mix_val = (1 - w) * clim_A_val + w * clim_B_val
    r = rmse_clim(tp, mix_val, 2013, 2022)
    print(f"  mistura {1-w:.2f}*completa + {w:.2f}*recente -> RMSE {r:.4f}")

print("""
  NOTA: se a mistura vencer as duas puras, vale usar o peso otimo.
  Rode de novo com o w escolhido antes de fechar a submissao.""")


# ---------------------------------------------------------------------------
sep("3. DIAGNOSTICO DA CLIMATOLOGIA ESCOLHIDA")

print(f"  janela: {janela}")
print(f"  shape: {dict(clim.sizes)}")
print(f"  media global: {float(clim.mean()):.4f} mm/dia")

# sanidade fisica: janeiro deve ser umido no norte, julho seco no centro-sul
jan = clim.sel(month=1)
jul = clim.sel(month=7)

def ponto(da, lat, lon):
    return float(da.sel(lat=lat, lon=lon, method="nearest"))

locais = {
    "Manaus      (-3.1, -60.0)": (-3.1, -60.0),
    "Sao Paulo   (-23.5, -46.6)": (-23.5, -46.6),
    "Atacama     (-23.0, -69.0)": (-23.0, -69.0),
    "Porto Alegre(-30.0, -51.2)": (-30.0, -51.2),
    "Belem       (-1.5, -48.5)": (-1.5, -48.5),
}
print(f"\n  {'local':28s} {'janeiro':>10s} {'julho':>10s}")
for nome, (la, lo) in locais.items():
    print(f"  {nome:28s} {ponto(jan, la, lo):10.2f} {ponto(jul, la, lo):10.2f}")

print("""
  Esperado: Manaus e Belem umidos em janeiro; Atacama proximo de zero
  nos dois meses; Porto Alegre parecido nos dois (chuva bem distribuida);
  Sao Paulo bem mais umido em janeiro que em julho.
  Se algum valor destoar, suspeite da orientacao da grade.""")

clim.to_netcdf(f"{SAIDA}/climatologia_tp.nc")
print(f"\n  salvo em {SAIDA}/climatologia_tp.nc")


# ---------------------------------------------------------------------------
sep("4. RMSE DA CLIMATOLOGIA POR REGIME ENOS")

print("""  A avaliacao final eh 2024, ano de transicao El Nino -> La Nina.
  Vale saber como a climatologia se comporta em anos analogos.""")

# anos de transicao de El Nino forte para La Nina
analogos = [1983, 1988, 1998, 2010, 2016]
for ano in analogos:
    r = rmse_clim(tp, clim, ano, ano)
    print(f"    {ano}: RMSE {r:.4f}")

todos = [rmse_clim(tp, clim, a, a) for a in range(1993, 2023)]
print(f"\n  media 1993-2022: {np.mean(todos):.4f} +- {np.std(todos):.4f}")
print("  se os analogos tiverem RMSE acima da media, confirma que anos de")
print("  transicao sao mais dificeis - e reforca encolhimento agressivo.")


# ---------------------------------------------------------------------------
sep("5. SUBMISSAO BASELINE")

sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
print(f"  sample_submission: {sub.shape}")

# decompoe o id: ano_mes_lat_lon. O mes do id EH o mes-alvo.
partes = sub["id"].str.split("_", expand=True)
mes = partes[1].astype(int).values
lat = partes[2].astype(float).values
lon = partes[3].astype(float).values

# indices na grade, derivados dos valores (grade regular de 0.25 graus)
lat0 = float(clim.lat.values[0])
lon0 = float(clim.lon.values[0])
passo = 0.25

ilat = np.rint((lat - lat0) / passo).astype(int)
ilon = np.rint((lon - lon0) / passo).astype(int)

# conferencia: os indices reconstroem as coordenadas originais?
assert np.allclose(clim.lat.values[ilat], lat), "indice de latitude errado"
assert np.allclose(clim.lon.values[ilon], lon), "indice de longitude errado"
print("  indices de grade conferem com as coordenadas do id")

# clim vem como (month, lat, lon); indexacao vetorizada
arr = clim.transpose("month", "lat", "lon").values
pred = arr[mes - 1, ilat, ilon]

print(f"  previsoes: media {pred.mean():.4f}, min {pred.min():.4f}, "
      f"max {pred.max():.4f}")
assert np.isfinite(pred).all(), "ha NaN ou inf na previsao"
assert (pred >= 0).all(), "ha previsao negativa"

sub["tp_mm_day"] = pred
caminho = f"{SUBS}/sub01_climatologia.csv"
sub.to_csv(caminho, index=False, float_format="%.4f")

print(f"\n  salvo em {caminho}")
print(f"  linhas: {len(sub)} (esperado 1885464)")
print("\n  ORDEM PRESERVADA: o arquivo mantem a ordem do sample_submission.")
print("  Confira as primeiras linhas antes de enviar:")
print(sub.head(3).to_string(index=False))


# ---------------------------------------------------------------------------
sep("6. EXPECTATIVA")

rmse_esperado = np.mean([rmse_clim(tp, clim, a, a) for a in (2019, 2020, 2021, 2022)])
print(f"""  RMSE medio da climatologia em anos recentes: {rmse_esperado:.4f}

  ANOTE ISSO ANTES DE SUBMETER. Se o placar publico voltar proximo desse
  valor, o pipeline esta correto. Se voltar muito diferente, ha bug de
  unidade, de ordenacao ou de alinhamento de mes - nao eh o modelo.

  Proximo passo: 02_anomalias.py""")