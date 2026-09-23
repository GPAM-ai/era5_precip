#!/usr/bin/env python
"""
ERA5 mensal em 200 e 500 hPa via CDS. RODAR NO MAC.

    python3 baixar_era5_niveis.py

Por que: a competicao entregou so 850 hPa. A Alta da Bolivia (200 hPa) e o
cavado do Nordeste sao a circulacao que governa a estacao chuvosa; estao
nos niveis superiores. Mesma reanalise, mesmo periodo - so niveis que nao
foram incluidos.

Variaveis: geopotencial e vento (u, v) em 200 hPa; geopotencial em 500.
Grade de 1 grau (basta: essas features entram so via EOF, agregadas a 2).
Blocos de 10 anos para nao estourar a fila.

Saida: era5_niveis_<anos>.nc
"""

import os

import cdsapi

c = cdsapi.Client()
AREA = [15, -90, -60, -25]
MESES = [f"{m:02d}" for m in range(1, 13)]

BASE = dict(
    product_type=["monthly_averaged_reanalysis"],
    variable=["geopotential", "u_component_of_wind", "v_component_of_wind"],
    pressure_level=["200", "500"],
    month=MESES,
    time=["00:00"],
    area=AREA,
    data_format="netcdf",
    download_format="unarchived",
)

blocos = [list(range(a, min(a + 10, 2025))) for a in range(1940, 2025, 10)]

for anos in blocos:
    arq = f"era5_niveis_{anos[0]}-{anos[-1]}.nc"
    if os.path.exists(arq) and os.path.getsize(arq) > 100_000:
        print(f"  {arq} ja existe"); continue
    print(f"  pedindo {arq} ...", flush=True)
    req = {**BASE, "year": [str(y) for y in anos]}
    try:
        # tenta grade de 1 grau (arquivo 16x menor); se o CDS recusar, cheia
        c.retrieve("reanalysis-era5-pressure-levels-monthly-means",
                   {**req, "grid": [1.0, 1.0]}, arq)
    except Exception as e:
        print(f"    grade 1 grau falhou ({str(e)[:60]}); pedindo 0.25")
        try:
            c.retrieve("reanalysis-era5-pressure-levels-monthly-means", req, arq)
        except Exception as e2:
            print(f"    FALHOU: {str(e2)[:120]}")
            continue
    print(f"  {arq} ok ({os.path.getsize(arq)/1e6:.1f} MB)")

print("\n  scp era5_niveis_*.nc rmm.txt sd:~/era5_precip/dados_processados/indices_brutos/")
