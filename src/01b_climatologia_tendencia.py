#!/usr/bin/env python
"""
Etapa 01b - A climatologia pode ser melhor?

MOTIVACAO:
  A submissao de climatologia otimizada marcou 1.8395 no placar publico,
  atras de concorrentes em 1.785 e 1.791. Como o teto de habilidade de
  modelo nesse problema eh de ~2%, uma diferenca de 3% dificilmente vem
  de modelagem - vem de um BASELINE melhor.

  E o diagnostico de 04 mostrou deriva forte nas variaveis (a_t2 desloca
  0.8 K entre treino e teste). Se a precipitacao tambem tem tendencia
  local, uma climatologia que EXTRAPOLA essa tendencia deve bater uma que
  apenas faz media.

O QUE SE TESTA AQUI:
  1. janelas de varios comprimentos (5, 10, 15, 20, 30, 40 anos, tudo)
  2. climatologia + tendencia linear local por ponto e mes
  3. o quanto a tendencia deve ser encolhida (extrapolar tendencia
     ajustada em poucos anos eh arriscado)

DESENHO DA VALIDACAO:
  Sempre o mesmo: ajusta ate um ano de corte, avalia nos anos seguintes,
  imitando a extrapolacao real. Dois cortes sao usados para checar se a
  conclusao eh estavel e nao artefato de um periodo especifico:
    corte 2012 -> avalia 2013-2022
    corte 2007 -> avalia 2008-2017

Uso:
    python src/01b_climatologia_tendencia.py
"""

import numpy as np
import xarray as xr

DADOS = "/prj/cptec/alex.campos/satrain/previsao_precipitacao_america_sul/dados"
SAIDA = "dados_processados"


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
anos = tp.time.dt.year
print(f"serie: {int(anos[0])} a {int(anos[-1])}, {tp.sizes['time']} meses")


def clim_janela(ini, fim):
    """Media por mes calendario no intervalo [ini, fim]."""
    return (tp.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
              .groupby("time.month").mean("time"))


def clim_tendencia(ini, fim, ano_ref):
    """Climatologia com tendencia linear local, avaliada em ano_ref.

    Para cada (ponto, mes calendario), ajusta chuva ~ a + b*ano no
    intervalo, e devolve o valor predito em ano_ref. Captura mudancas
    locais de regime que uma media simples ignora.
    """
    sub = tp.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
    a_ = sub.time.dt.year
    nivel, incl = [], []

    for m in range(1, 13):
        sm = sub.sel(time=sub.time.dt.month == m)
        x = sm.time.dt.year.values.astype("float64")
        xc = x - x.mean()
        den = (xc ** 2).sum()
        y = sm.values                                   # (anos, lat, lon)
        b = np.tensordot(xc, y - y.mean(0), axes=(0, 0)) / den
        a0 = y.mean(0) - b * x.mean()
        nivel.append(a0)
        incl.append(b)

    nivel = np.stack(nivel)
    incl = np.stack(incl)
    return nivel, incl


def avaliar(pred_por_mes, ano_ini, ano_fim):
    """RMSE de uma previsao dada como array (12, lat, lon)."""
    obs = tp.sel(time=slice(f"{ano_ini}-01-01", f"{ano_fim}-12-31"))
    meses = obs.time.dt.month.values
    err = obs.values - pred_por_mes[meses - 1]
    return float(np.sqrt(np.nanmean(err ** 2)))


# ---------------------------------------------------------------------------
sep("1. COMPRIMENTO DA JANELA")

print("""  So testamos 83 anos contra 30. Janelas curtas respondem mais
  rapido a mudancas de regime, mas tem mais ruido amostral. Existe um
  ponto otimo, e ele pode nao ser 30.
""")

for corte, (v_ini, v_fim) in ((2012, (2013, 2022)), (2007, (2008, 2017))):
    print(f"  corte {corte}, avaliando {v_ini}-{v_fim}:")
    melhor = (1e9, None)
    for n in (5, 10, 15, 20, 30, 40, 60, 999):
        ini = max(1940, corte - n + 1)
        c = clim_janela(ini, corte).transpose("month", "lat", "lon").values
        r = avaliar(c, v_ini, v_fim)
        rotulo = f"{ini}-{corte}" if n != 999 else f"1940-{corte} (tudo)"
        marca = ""
        if r < melhor[0]:
            melhor = (r, rotulo)
        print(f"    {rotulo:20s} ({n if n!=999 else corte-1939:3d} anos)  "
              f"RMSE {r:.4f}{marca}")
    print(f"    >> melhor: {melhor[1]} com {melhor[0]:.4f}\n")


# ---------------------------------------------------------------------------
sep("2. CLIMATOLOGIA COM TENDENCIA LOCAL")

print("""  Ajusta chuva ~ a + b*ano por ponto e mes, e extrapola. A
  inclinacao eh encolhida por um fator k: extrapolar tendencia crua eh
  arriscado, porque parte da inclinacao eh ruido amostral.

    previsao = nivel + k * inclinacao * ano_alvo
""")

for corte, (v_ini, v_fim) in ((2012, (2013, 2022)), (2007, (2008, 2017))):
    print(f"  corte {corte}, avaliando {v_ini}-{v_fim}:")

    for n_anos in (30, 40, 60):
        ini = max(1940, corte - n_anos + 1)
        nivel, incl = clim_tendencia(ini, corte, corte)

        base = clim_janela(ini, corte).transpose("month", "lat", "lon").values
        r_base = avaliar(base, v_ini, v_fim)

        linha = [f"    janela {ini}-{corte}: base {r_base:.4f}"]
        for k in (0.25, 0.5, 0.75, 1.0):
            # avalia ano a ano, porque a extrapolacao depende do ano
            erros = []
            for ano in range(v_ini, v_fim + 1):
                pred = nivel + k * incl * ano
                obs = tp.sel(time=slice(f"{ano}-01-01", f"{ano}-12-31"))
                meses = obs.time.dt.month.values
                erros.append(((obs.values - pred[meses - 1]) ** 2).ravel())
            r = float(np.sqrt(np.mean(np.concatenate(erros))))
            linha.append(f"k={k:.2f} {r:.4f}")
        print("  ".join(linha))
    print()


# ---------------------------------------------------------------------------
sep("3. LEITURA")

print("""  O que procurar:

  - Se uma janela curta (10-20 anos) vencer com folga, a nossa mistura
    de 0.75 esta subotima e vale refazer a submissao baseline.

  - Se a tendencia com k pequeno (0.25-0.5) melhorar, ha deriva local
    real na precipitacao e extrapola-la de forma conservadora ajuda.
    k=1.0 raramente vence: extrapolar inclinacao crua amplifica ruido.

  - Se nada bater a mistura atual (1.8406 no corte 2012), entao a
    diferenca para os concorrentes vem de modelagem, nao de baseline,
    e o esforco volta inteiro para o LightGBM.

  Qualquer que seja o resultado, ele vai para o README: testamos
  alternativas de baseline com validacao honesta, em vez de assumir.""")