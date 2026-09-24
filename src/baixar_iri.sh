#!/bin/bash
# Download dos hindcasts do NMME e do CanSIPS via IRI Data Library.
# Rodar numa maquina com internet. Os operacionais ja estao versionados em
# dados_processados/indices_brutos/; os hindcasts ficam fora do git pelo tamanho.
#
#   bash src/baixar_iri.sh
#   (depois mover os .nc para dados_processados/indices_brutos/)
#
# Fonte: IRI Data Library, http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
# Recorte: 90W-25W, 60S-15N; media do ensemble; lead 0.5 e 1.5.

set -u
R="http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"
OPS05="L/0.5/VALUE/X/270/335/RANGEEDGES/Y/-60/15/RANGEEDGES/[M]average/data.nc"
OPS15="L/1.5/VALUE/X/270/335/RANGEEDGES/Y/-60/15/RANGEEDGES/[M]average/data.nc"

baixa () { curl -g -m 900 -fo "$1" "$2" || echo "FALHOU: $1"; }

# CFSv2
baixa cfsv2_hind.nc      "$R/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec/$OPS05"
baixa cfsv2_hind_L15.nc  "$R/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec/$OPS15"

# GFDL-SPEAR e NASA-GEOSS2S
for m in GFDL-SPEAR NASA-GEOSS2S; do
  baixa ${m}_hind.nc     "$R/.$m/.HINDCAST/.MONTHLY/.prec/$OPS05"
  baixa ${m}_hind_L15.nc "$R/.$m/.HINDCAST/.MONTHLY/.prec/$OPS15"
done

# CanSIPS IC3
baixa CanSIPS-IC3_hind.nc "$R/.CanSIPS-IC3/.HINDCAST/.MONTHLY/.prec/$OPS05"

# CanSIPS IC4 em blocos de 10 anos (o IRI encerra requisicoes acima de ~9 min)
IC4="$R/.CanSIPS-IC4/.HINDCAST/.MONTHLY/.prec"
baixa CanSIPS-IC4_hind_1991-2000.nc "$IC4/S/(Jan%201991)/(Dec%202000)/RANGEEDGES/$OPS05"
baixa CanSIPS-IC4_hind_2001-2010.nc "$IC4/S/(Jan%202001)/(Dec%202010)/RANGEEDGES/$OPS05"
baixa CanSIPS-IC4_hind_2011-2020.nc "$IC4/S/(Jan%202011)/(Dec%202020)/RANGEEDGES/$OPS05"

ls -la *_hind*.nc
