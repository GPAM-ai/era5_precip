#!/usr/bin/env python
"""
Download dos demais sistemas sazonais do C3S via CDS. RODAR NO MAC.

    python3 baixar_c3s.py            # todos os centros
    python3 baixar_c3s.py ukmo meteo_france   # so alguns

Para cada centro, tenta os sistemas do mais novo ao mais antigo ate achar
um que cubra 2023-2024 (operacional). Desse sistema, baixa:
  - hindcast 1993-2016 (monthly_mean, com membros; blocos de 6 anos)
  - operacional 2017-2022, se existir para o mesmo sistema (ou versoes
    anteriores, para que os folds recentes do CV tambem enxerguem o modelo)
  - operacional 2023-2024 (ensemble_mean)

Nomes de saida:
  c3s_<centro>_hind_s<sys>_<anos>.nc
  c3s_<centro>_fcst_s<sys>_<anos>.nc

O 13d le por padrao de nome; o sistema fica gravado no nome para
rastreabilidade.
"""

import os
import sys

import cdsapi

c = cdsapi.Client()
AREA = [15, -90, -60, -25]
MESES = [f"{m:02d}" for m in range(1, 13)]

# sistemas por centro, do mais novo ao mais antigo
CENTROS = {
    "ukmo":         ["603", "602", "601", "600"],
    "meteo_france": ["9", "8", "7"],
    "dwd":          ["22", "21"],
    "cmcc":         ["4", "35"],
    "jma":          ["3"],
}
pedidos = sys.argv[1:] or list(CENTROS)

BASE = dict(variable=["total_precipitation"], leadtime_month=["1", "2"],
            area=AREA, data_format="netcdf")


def pedir(arq, centro, sistema, anos, meses, produto):
    if os.path.exists(arq) and os.path.getsize(arq) > 10_000:
        print(f"    {arq} ja existe"); return True
    try:
        c.retrieve("seasonal-monthly-single-levels",
                   {**BASE, "originating_centre": centro, "system": sistema,
                    "year": [str(y) for y in anos], "month": meses,
                    "product_type": [produto]}, arq)
        print(f"    {arq} ok ({os.path.getsize(arq)/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"    {arq} falhou: {str(e).replace(chr(10), ' ')[:90]}")
        if os.path.exists(arq):
            os.remove(arq)
        return False


def blocos(a, b, n):
    return [list(range(i, min(i + n, b + 1))) for i in range(a, b + 1, n)]


for centro in pedidos:
    print(f"\n== {centro}")
    escolhido = None
    for s in CENTROS[centro]:
        arq = f"c3s_{centro}_fcst_s{s}_2023-2024.nc"
        if pedir(arq, centro, s, [2023, 2024], MESES, "ensemble_mean"):
            escolhido = s
            break
    if escolhido is None:
        print(f"  {centro}: nenhum sistema cobre 2023-24 - pulado")
        continue
    print(f"  sistema escolhido: {escolhido}")

    for anos in blocos(1993, 2016, 6):
        pedir(f"c3s_{centro}_hind_s{escolhido}_{anos[0]}-{anos[-1]}.nc",
              centro, escolhido, anos, MESES, "monthly_mean")

    # operacional 2017-2022: tenta o mesmo sistema, depois os anteriores
    for anos in blocos(2017, 2022, 3):
        ok = False
        for s in CENTROS[centro][CENTROS[centro].index(escolhido):]:
            if pedir(f"c3s_{centro}_fcst_s{s}_{anos[0]}-{anos[-1]}.nc",
                     centro, s, anos, MESES, "ensemble_mean"):
                ok = True
                break
        if not ok:
            print(f"    {centro} {anos[0]}-{anos[-1]}: sem operacional em nenhuma versao")

print("\n  Depois: scp c3s_*.nc sd:~/era5_precip/dados_processados/indices_brutos/")
