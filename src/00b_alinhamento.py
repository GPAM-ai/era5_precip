#!/usr/bin/env python
"""
Resolve a ambiguidade de alinhamento temporal do teste.

Pergunta 1: o arquivo de teste ja vem deslocado? Ou seja, as covariaveis
            em time=2023-01 sao na verdade de dezembro/2022?
Pergunta 2: tp_ultima_obs eh constante nos 24 passos ou varia?

Uso:
    python 00b_alinhamento.py
"""

import numpy as np
import xarray as xr

D = "dados_brutos"

sep = lambda t: print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ---------------------------------------------------------------------------
sep("1. O TESTE JA VEM DESLOCADO?")

tr = xr.open_dataset(f"{D}/treino_t2.nc").t2
te = xr.open_dataset(f"{D}/teste_features.nc").t2

ultimo_treino = tr.isel(time=-1)          # 2022-12
primeiro_teste = te.isel(time=0)          # rotulado 2023-01

print(f"  ultimo passo do treino : {str(tr.time.values[-1])[:10]}")
print(f"  primeiro passo do teste: {str(te.time.values[0])[:10]}")

a = ultimo_treino.values
b = primeiro_teste.values
iguais = np.allclose(a, b, equal_nan=True)
dif = np.nanmean(np.abs(a - b))

print(f"\n  identicos? {iguais}")
print(f"  diferenca absoluta media: {dif:.4f}")

if iguais:
    print("""
  >> O ARQUIVO DE TESTE JA VEM DESLOCADO.
     As covariaveis rotuladas 2023_01 sao de dezembro/2022.
     O id do sample_submission eh o MES-ALVO.
     Alinhamento: usar as linhas do teste como estao, sem shift.""")
else:
    print(f"""
  >> NAO sao identicos (dif media {dif:.4f}).
     As covariaveis parecem ser do proprio mes do rotulo.
     Nesse caso 'id 2023_01' provavelmente significa: features de
     janeiro/2023 -> prever fevereiro/2023, e a submissao estaria
     desalinhada em um mes em relacao ao id.
     CONFIRMAR NO DISCORD antes de escrever o pipeline.""")

# comparacao de controle: quao diferentes sao dois meses quaisquer?
# serve para saber se 'dif' acima eh grande ou pequena na escala da variavel
c = tr.isel(time=-2).values                # 2022-11
dif_controle = np.nanmean(np.abs(a - c))
print(f"\n  controle (2022-12 vs 2022-11): diferenca media {dif_controle:.4f}")
print("  se a diferenca do teste for MUITO menor que o controle,")
print("  os campos sao praticamente o mesmo mes.")


# ---------------------------------------------------------------------------
sep("2. tp_ultima_obs VARIA NO TEMPO?")

tu = xr.open_dataset(f"{D}/teste_features.nc").tp_ultima_obs

p0 = tu.isel(time=0).values
p12 = tu.isel(time=12).values
p23 = tu.isel(time=23).values

print(f"  passo 0 vs passo 12 identicos? {np.allclose(p0, p12, equal_nan=True)}")
print(f"  passo 0 vs passo 23 identicos? {np.allclose(p0, p23, equal_nan=True)}")

nans = tu.isnull().sum(dim=("lat", "lon")).values
print(f"\n  NaN por passo de tempo: {nans}")

if np.allclose(p0, p12, equal_nan=True) and np.allclose(p0, p23, equal_nan=True):
    print("""
  >> tp_ultima_obs eh CONSTANTE nos 24 passos: eh sempre dez/2022.
     Como feature, so faz sentido para o primeiro mes previsto.
     Para os demais, eh informacao velha de ate 2 anos -> inutil.
     Confirma a decisao de NAO usar persistencia de precipitacao.""")
else:
    print("""
  >> tp_ultima_obs VARIA. Investigar: talvez seja a chuva observada
     do mes anterior a cada alvo, o que reabriria a persistencia.""")


# ---------------------------------------------------------------------------
sep("3. tp_alvo NO TESTE ESTA MESMO VAZIO?")

ta = xr.open_dataset(f"{D}/teste_features.nc").tp_alvo
frac_nan = float(ta.isnull().mean())
print(f"  fracao de NaN em tp_alvo (teste): {frac_nan:.4f}")
if frac_nan < 0.999:
    print("  >> ATENCAO: nem tudo eh NaN. Verificar se vazou algum valor.")
else:
    print("  >> tudo NaN, como esperado.")


# ---------------------------------------------------------------------------
sep("4. ESCALAS DAS VARIAVEIS (para conferir unidades)")

vars_teste = ["t2", "cloud_cover", "shum_850", "surface_pressure",
              "u_850", "v_850", "temperature_850", "rel_hum_850",
              "geopotential_850"]

dste = xr.open_dataset(f"{D}/teste_features.nc")
print(f"  {'variavel':20s} {'media':>12s} {'min':>12s} {'max':>12s}")
for v in vars_teste:
    x = dste[v].isel(time=0).values
    x = x[np.isfinite(x)]
    print(f"  {v:20s} {x.mean():12.3f} {x.min():12.3f} {x.max():12.3f}")

print("""
  Referencia rapida:
    t2 / temperature_850 ~ 280-300  -> Kelvin
    surface_pressure     ~ 1e5      -> Pascal
    geopotential_850     ~ 1.4e4    -> m2/s2  (dividir por 9.81 da altura em m)
    shum_850             ~ 0.01     -> kg/kg
    rel_hum_850          ~ 0-100    -> porcentagem
    cloud_cover          ~ 0-1      -> fracao
    u_850 / v_850        ~ -20 a 20 -> m/s""")


# ---------------------------------------------------------------------------
sep("5. tp DO TREINO: CONFERIR mm/dia E O ALINHAMENTO COM tp_alvo")

tp = xr.open_dataset(f"{D}/treino_tp.nc").tp
alvo = xr.open_dataset(f"{D}/treino_tp_alvo.nc").tp_alvo

x = tp.isel(time=slice(0, 12)).values
x = x[np.isfinite(x)]
print(f"  tp media global (primeiro ano): {x.mean():.4f} mm/dia")
print(f"  tp max: {x.max():.2f} mm/dia")

# o alvo em t deve ser igual a tp em t+1
a1 = alvo.isel(time=100).values
t1 = tp.isel(time=101).values
print(f"\n  tp_alvo[t=100] == tp[t=101]? "
      f"{np.allclose(a1, t1, equal_nan=True)}")
print("  (se True, confirma que o alvo eh a chuva do mes seguinte)")

print(f"\n  NaN em tp_alvo no ultimo passo: "
      f"{float(alvo.isel(time=-1).isnull().mean()):.4f}")