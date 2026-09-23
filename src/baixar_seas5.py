#!/usr/bin/env python
"""
Download do ECMWF SEAS5 via Copernicus CDS. RODAR NO MAC.

    python3 -m pip install --user cdsapi
    python3 baixar_seas5.py

Versao para a API nova do CDS (2024+): a chave eh `data_format`, nao
`format`. Os pedidos sao quebrados em blocos de anos - cada um entra
separado na fila e falha isolado se algum ano nao existir.

  system 51 (SEAS5.1)  hindcast 1981-2016  + operacional nov/2022 em diante
  system  5 (SEAS5)    hindcast 1981-2016  + operacional 2017 a out/2022

Saida: varios seas5_<tag>_<anos>.nc, que o 13d concatena.
"""

import os

import cdsapi

c = cdsapi.Client()
AREA = [15, -90, -60, -25]
MESES = [f"{m:02d}" for m in range(1, 13)]

BASE = dict(
    originating_centre="ecmwf",
    variable=["total_precipitation"],
    leadtime_month=["1", "2"],
    area=AREA,
    data_format="netcdf",
)
# ensemble_mean so existe para operacionais; hindcasts vem so como
# monthly_mean com os membros individuais (25). A media eh feita no 13d.
OPER = dict(product_type=["ensemble_mean"])
HIND = dict(product_type=["monthly_mean"])


def blocos(a, b, n):
    return [list(range(i, min(i + n, b + 1))) for i in range(a, b + 1, n)]


PEDIDOS = []
for anos in blocos(1981, 2016, 6):
    PEDIDOS.append((f"seas5_hind_s5_{anos[0]}-{anos[-1]}.nc",
                    dict(system="5", year=[str(y) for y in anos], month=MESES, **HIND)))
for anos in blocos(2017, 2022, 3):
    PEDIDOS.append((f"seas5_fcst_s5_{anos[0]}-{anos[-1]}.nc",
                    dict(system="5", year=[str(y) for y in anos], month=MESES, **OPER)))
PEDIDOS.append(("seas5_fcst_s51_2022.nc",
                dict(system="51", year=["2022"], month=["11", "12"], **OPER)))
PEDIDOS.append(("seas5_fcst_s51_2023-2024.nc",
                dict(system="51", year=["2023", "2024"], month=MESES, **OPER)))

falhas = []
for arq, extra in PEDIDOS:
    if os.path.exists(arq) and os.path.getsize(arq) > 10_000:
        print(f"  {arq} ja existe")
        continue
    print(f"  pedindo {arq} ...", flush=True)
    try:
        c.retrieve("seasonal-monthly-single-levels", {**BASE, **extra}, arq)
        print(f"  {arq} ok ({os.path.getsize(arq)/1e6:.1f} MB)")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        print(f"  {arq} FALHOU: {msg}")
        falhas.append((arq, msg))

print("\n  RESUMO")
ok = [a for a, _ in PEDIDOS if os.path.exists(a) and os.path.getsize(a) > 10_000]
print(f"  {len(ok)} de {len(PEDIDOS)} arquivos")
for a, m in falhas:
    print(f"  falhou: {a}  ->  {m[:80]}")
print("""
  Se os hindcasts s51 falharem todos com 'no data', o s51 pode nao ter
  hindcast no CDS para esse produto - nesse caso o s5 serve para os dois
  (o 13d ja trata). O essencial eh ter: pelo menos UM hindcast completo
  e os operacionais 2017-2024.

  Confira a estrutura de um arquivo e me mande:
    python3 -c "import xarray; print(xarray.open_dataset('seas5_fcst_s51_2023-2024.nc'))"
""")
