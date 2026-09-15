#!/usr/bin/env python
"""
Etapa 02 - Anomalias.  (versao com disciplina de memoria)

Converte tudo para desvio em relacao a climatologia do mes calendario.
Remove o ciclo sazonal e a assinatura topografica (critico para
surface_pressure e geopotential_850, onde os Andes dominam o bruto).

MEMORIA - por que o codigo esta escrito assim:
  A operacao ingenua `bruto - clim.sel(month=bruto.time.dt.month)` expande
  a climatologia para os 996 passos em resolucao cheia: 313 MB so para o
  operando, mais o original e o resultado. Da ~1 GB de pico por variavel,
  e o no de login mata o processo. Aqui a grade eh reduzida ANTES da
  subtracao:
    treino -> subamostra espacial (stride), ~35 MB por variavel
    eof    -> agrega a 2 graus antes de anomalizar; coarsen eh linear,
              entao coarsen(anomalia) == anomalia(coarsen)
    teste  -> resolucao cheia, mas sao so 24 passos (~7 MB)

Produz em dados_processados/:
  anomalias_treino.nc, anomalias_teste.nc, campos_eof.nc

ALINHAMENTO (verificado em 00b):
  treino: time = mes das FEATURES; mes-alvo = (mes(time) % 12) + 1
  teste : time = mes-ALVO; mes-feature = ((mes(time) - 2) % 12) + 1

Uso:
    python src/02_anomalias.py [stride]
"""

import gc
import os
import sys

import numpy as np
import xarray as xr

DADOS = "/prj/cptec/alex.campos/satrain/previsao_precipitacao_america_sul/dados"
SAIDA = "dados_processados"
os.makedirs(SAIDA, exist_ok=True)

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
W_MIX = 0.75
ANO_REC = 1993
COARSEN = 8

COVARS = ["t2", "cloud_cover", "shum_850", "surface_pressure",
          "u_850", "v_850", "temperature_850", "rel_hum_850",
          "geopotential_850"]

R_TERRA = 6.371e6
SL = slice(None, None, STRIDE)


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


def mem():
    """Memoria residente do processo, em MB."""
    try:
        with open("/proc/self/status") as f:
            for linha in f:
                if linha.startswith("VmRSS:"):
                    return int(linha.split()[1]) / 1024
    except Exception:
        pass
    return float("nan")


def clim_mista(da):
    """Climatologia por mes calendario: mistura das janelas escolhida em 01."""
    completa = da.groupby("time.month").mean("time")
    recente = (da.sel(time=slice(f"{ANO_REC}-01-01", None))
                 .groupby("time.month").mean("time"))
    out = ((1 - W_MIX) * completa + W_MIX * recente).astype("float32")
    del completa, recente
    return out


def anomalia(campo, clim, meses):
    return (campo - clim.sel(month=meses).drop_vars("month")).astype("float32")


def coarse(da):
    return da.coarsen(lat=COARSEN, lon=COARSEN, boundary="trim").mean()


def transporte_umidade(q, u, v):
    """qu, qv e convergencia do fluxo de umidade.

    xarray.differentiate devolve derivada por GRAU; converte-se para metro
    com R*cos(lat)*pi/180 na zonal e R*pi/180 na meridional.
    """
    qu = (q * u).astype("float32")
    qv = (q * v).astype("float32")
    coslat = np.cos(np.deg2rad(q.lat))
    dqu_dx = qu.differentiate("lon") / (R_TERRA * coslat * np.pi / 180.0)
    dqv_dy = qv.differentiate("lat") / (R_TERRA * np.pi / 180.0)
    conv = (-(dqu_dx + dqv_dy) * 1e6).astype("float32")
    del dqu_dx, dqv_dy
    return {"qu": qu, "qv": qv, "conv_umid": conv}


# ---------------------------------------------------------------------------
sep(f"CONFIGURACAO (stride = {STRIDE})")

ref = xr.open_dataset(f"{DADOS}/treino_tp.nc")
nlat, nlon = ref.sizes["lat"], ref.sizes["lon"]
n_sub = len(ref.lat[SL]) * len(ref.lon[SL])
print(f"  grade cheia       : {nlat} x {nlon} = {nlat*nlon:,}")
print(f"  grade subamostrada: {len(ref.lat[SL])} x {len(ref.lon[SL])} = {n_sub:,}")
print(f"  linhas de treino  : {n_sub * 995:,}")
print(f"  memoria inicial   : {mem():.0f} MB")
ref.close()


# ---------------------------------------------------------------------------
sep("1. ANOMALIA DO ALVO")

clim_tp = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")

alvo = xr.open_dataset(f"{DADOS}/treino_tp_alvo.nc").tp_alvo
mes_alvo = (alvo.time.dt.month % 12) + 1
print(f"  primeiros meses-alvo: {mes_alvo.values[:6]}  (esperado 2,3,4,5,6,7)")

anom_alvo = anomalia(alvo.isel(lat=SL, lon=SL).load(),
                     clim_tp.isel(lat=SL, lon=SL), mes_alvo)
anom_alvo.name = "anom_alvo"
alvo.close()

tp_obs = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.isel(lat=SL, lon=SL).load()

print(f"  media  : {float(anom_alvo.mean()):+.5f}  (deve ser proximo de 0)")
print(f"  desvio : {float(anom_alvo.std()):.4f}")
print(f"  faixa  : {float(anom_alvo.min()):.2f} a {float(anom_alvo.max()):.2f}")
print(f"  memoria: {mem():.0f} MB")

if abs(float(anom_alvo.mean())) > 0.05:
    print("\n  !! media longe de zero: suspeitar do alinhamento de mes.")


# ---------------------------------------------------------------------------
sep("2. ANOMALIAS DAS COVARIAVEIS")

teste = xr.open_dataset(f"{DADOS}/teste_features.nc")
mes_feat_te = ((teste.time.dt.month - 2) % 12) + 1
print(f"  meses-feature do teste: {mes_feat_te.values[:6]} ...")
print("  (esperado 12,1,2,3,4,5 - dezembro precede o alvo de janeiro)\n")

ds_tr, ds_te, ds_eof = xr.Dataset(), xr.Dataset(), xr.Dataset()

for v in COVARS:
    bruto = xr.open_dataset(f"{DADOS}/treino_{v}.nc")[v].load()
    meses = bruto.time.dt.month

    clim_cheia = clim_mista(bruto)

    # teste: resolucao cheia, 24 passos
    ds_te[f"a_{v}"] = anomalia(teste[v], clim_cheia, mes_feat_te)

    # treino: subamostra antes de subtrair
    ds_tr[f"a_{v}"] = anomalia(bruto.isel(lat=SL, lon=SL),
                               clim_cheia.isel(lat=SL, lon=SL), meses)

    # eof: agrega antes de anomalizar (coarsen eh linear)
    cg = coarse(bruto)
    ds_eof[f"a_{v}"] = anomalia(cg, clim_mista(cg), meses)

    print(f"  {v:20s} desvio {float(ds_tr[f'a_{v}'].std()):10.4f}   "
          f"media {float(ds_tr[f'a_{v}'].mean()):+.2e}   [{mem():.0f} MB]",
          flush=True)

    del bruto, cg, clim_cheia, meses
    gc.collect()


# ---------------------------------------------------------------------------
sep("3. TRANSPORTE DE UMIDADE (features derivadas)")

print("""  A variavel fisicamente mais direta para chuva nao foi entregue.
  Com shum_850 e o vento em 850 hPa da para construi-la:

    qu = q*u, qv = q*v          transporte de umidade
    conv = -(dqu/dx + dqv/dy)   convergencia

  Convergencia positiva significa umidade se acumulando na coluna - eh de
  onde a chuva vem. Calculada no campo BRUTO e so depois anomalizada,
  porque o produto q*u nao eh linear.
""")

q = xr.open_dataset(f"{DADOS}/treino_shum_850.nc").shum_850.load()
u = xr.open_dataset(f"{DADOS}/treino_u_850.nc").u_850.load()
v = xr.open_dataset(f"{DADOS}/treino_v_850.nc").v_850.load()
meses = q.time.dt.month

deriv_te = transporte_umidade(teste.shum_850, teste.u_850, teste.v_850)

for chave in ("qu", "qv", "conv_umid"):
    if chave == "qu":
        campo = (q * u).astype("float32")
    elif chave == "qv":
        campo = (q * v).astype("float32")
    else:
        qu = (q * u).astype("float32")
        qv = (q * v).astype("float32")
        coslat = np.cos(np.deg2rad(q.lat))
        campo = (-((qu.differentiate("lon") / (R_TERRA * coslat * np.pi / 180.0))
                   + (qv.differentiate("lat") / (R_TERRA * np.pi / 180.0)))
                 * 1e6).astype("float32")
        del qu, qv
        gc.collect()

    clim_cheia = clim_mista(campo)

    ds_te[f"a_{chave}"] = anomalia(deriv_te[chave], clim_cheia, mes_feat_te)
    ds_tr[f"a_{chave}"] = anomalia(campo.isel(lat=SL, lon=SL),
                                   clim_cheia.isel(lat=SL, lon=SL), meses)
    cg = coarse(campo)
    ds_eof[f"a_{chave}"] = anomalia(cg, clim_mista(cg), meses)

    print(f"  {chave:20s} desvio {float(ds_tr[f'a_{chave}'].std()):10.4f}   "
          f"[{mem():.0f} MB]", flush=True)

    del campo, cg, clim_cheia
    gc.collect()

del q, u, v, deriv_te
gc.collect()


# ---------------------------------------------------------------------------
sep("4. GRAVANDO")

ds_tr["anom_alvo"] = anom_alvo
ds_tr["tp_obs"] = tp_obs        # referencia, NAO usar como feature

for ds, arq in ((ds_tr, "anomalias_treino.nc"),
                (ds_te, "anomalias_teste.nc"),
                (ds_eof, "campos_eof.nc")):
    caminho = f"{SAIDA}/{arq}"
    enc = {k: {"zlib": True, "complevel": 1} for k in ds.data_vars}
    ds.to_netcdf(caminho, encoding=enc)
    print(f"  {arq:24s} {os.path.getsize(caminho)/1e6:8.1f} MB   "
          f"{len(ds.data_vars)} variaveis")

print(f"""
  treino: {dict(ds_tr.sizes)}
  teste : {dict(ds_te.sizes)}
  eof   : {dict(ds_eof.sizes)}
  memoria ao final: {mem():.0f} MB""")


# ---------------------------------------------------------------------------
sep("5. SANIDADE FISICA")

print("""  Em El Nino forte a assinatura na America do Sul eh conhecida:
  seca no norte da Amazonia e no Nordeste, chuva acima do normal no sul
  do Brasil. Em La Nina o padrao se inverte. Se os sinais vierem
  trocados, ha erro de alinhamento ou de orientacao da grade.
""")

regioes = {
    "Amazonia norte": ((-5, 2), (-70, -55)),
    "Nordeste      ": ((-12, -4), (-45, -36)),
    "Sul do Brasil ": ((-32, -25), (-57, -50)),
}

for rotulo, periodo in (("El Nino 2015/16", ("2015-11-01", "2016-02-01")),
                        ("La Nina 2010/11", ("2010-11-01", "2011-02-01"))):
    print(f"  {rotulo}:")
    for nome, (la, lo) in regioes.items():
        x = float(anom_alvo.sel(time=slice(*periodo),
                                lat=slice(*la), lon=slice(*lo)).mean())
        print(f"    {nome}  {x:+.3f} mm/dia")
    print()

print("  Proximo passo: 03_eofs.py")