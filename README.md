# Previsão Climática Mensal de Precipitação sobre a América do Sul

**Equipe Rain-NP-Hard** · GPAM / UNIFESP · Hackathon WorCAP 2026 (INPE)

Solução para o desafio de prever a precipitação média mensal (mm/dia) do mês seguinte sobre a América do Sul, em 78.561 pontos de grade a 0,25°, a partir de campos atmosféricos de reanálise ERA5 do mês corrente.

**Resultado no placar público (2023): RMSE 1,48895 — 1º lugar**, com 0,035 de vantagem sobre o 2º colocado. A climatologia de referência marca 1,83952; o modelo reduz o erro em 19,1%.

---

## Sumário

1. [Resumo executivo](#1-resumo-executivo)
2. [O problema e sua dificuldade](#2-o-problema-e-sua-dificuldade)
3. [Trajetória: de onde veio cada ganho](#3-trajetória-de-onde-veio-cada-ganho)
4. [Metodologia](#4-metodologia)
5. [Previsão dinâmica como preditor (MOS)](#5-previsão-dinâmica-como-preditor-mos)
6. [Validação](#6-validação)
7. [Hipóteses testadas e rejeitadas](#7-hipóteses-testadas-e-rejeitadas)
8. [Fontes de dados externas](#8-fontes-de-dados-externas)
9. [Pipeline e reprodução](#9-pipeline-e-reprodução)
10. [Limitações conhecidas](#10-limitações-conhecidas)
11. [Referências](#11-referências)

---

## 1. Resumo executivo

A precipitação mensal é dominada pelo ciclo sazonal e pelo padrão geográfico, que uma climatologia bem construída captura sem modelo algum. A parte difícil — e a única que vale nota — é a **anomalia**: quanto um mês foge do normal daquele lugar. Com um mês de antecedência, a habilidade disponível para prever essa anomalia a partir de reanálise é pequena: medimos correlação de 0,19 usando apenas os campos entregues pela competição.

A solução final tem quatro camadas:

1. **Climatologia refinada**: janela de 24 anos misturada ao período completo, com tendência local encolhida. Ganho de 1% sobre a média simples.
2. **LightGBM global sobre a anomalia**, com atributos de reanálise (anomalias locais e modos de grande escala por EOF), índices de temperatura da superfície do mar (TSM) da NOAA, e — o componente decisivo — **previsões de nove sistemas de previsão sazonal** de sete centros operacionais, usadas como preditores (*Model Output Statistics*). Correlação de anomalia: 0,528.
3. **Regressão ridge local por ponto de grade** sobre os mesmos sistemas dinâmicos — a formulação clássica de MOS — combinada com o LightGBM em peso fixo validado. As duas formulações erram de jeitos diferentes (correlação 0,85 entre suas previsões); a combinação ganha 0,83 pontos percentuais sobre o LightGBM sozinho.
4. **Duas submissões finais** que diferem apenas na presença da camada 3, cobrindo a hipótese de que a combinação não transfira para 2024.

A conclusão central: **todo ganho relevante veio de informação física nova ou de diversidade de formulação, nunca de capacidade de modelo.** Dezoito hipóteses de modelagem foram testadas com validação cruzada; doze foram rejeitadas. As seis aceitas somam menos de 3 pontos percentuais. A entrada de previsão dinâmica somou 13.

---

## 2. O problema e sua dificuldade

**Dados entregues:** reanálise ERA5 mensal de 1940 a 2022 (treino), com 8 covariáveis atmosféricas em 850 hPa e superfície, e precipitação observada como alvo. Teste: 2023 (público) e 2024 (privado), com as covariáveis mas sem a precipitação.

**Detalhe crítico de alinhamento**, verificado empiricamente em `00b_alinhamento.py`: no treino, a coordenada temporal é o mês das *features* e o alvo é o mês seguinte; no teste, a coordenada é o mês do *alvo* e as covariáveis são do mês anterior. Os campos do teste em 2023-01 são idênticos aos do treino em 2022-12 (diferença 0,0000 contra 1,16 entre dois meses quaisquer). Toda a sazonalidade e a climatologia são indexadas pelo mês-alvo.

**O que torna o problema difícil:**

- A anomalia tem desvio-padrão de 1,80 mm/dia. A climatologia, por definição, tem RMSE igual a isso. Todo modelo é medido contra esse piso.
- Sem TSM nos dados entregues, o ENOS — o principal motor da variabilidade — só entra indiretamente, pela pegada na pressão à superfície.
- O domínio termina em 90°W, cobrindo apenas a borda do Pacífico.
- 2023 e 2024 são anos de temperatura recorde. Várias variáveis do teste ficam **fora da distribuição de treino**, o que penaliza modelos de árvore, que não extrapolam.

---

## 3. Trajetória: de onde veio cada ganho

Cada linha foi validada por CV antes de ser submetida. As colunas mostram o ganho medido no CV walk-forward e o resultado real no placar público.

| etapa | o que entrou | CV: ganho do modelo | placar público |
|---|---|---|---|
| Climatologia v1 | janela otimizada | — | 1,8395 |
| Climatologia v3 | + mistura de janelas + tendência k=0,15 | +0,96% (sobre v1) | 1,8337 |
| LightGBM | anomalias locais + EOFs de reanálise | +1,88% | 1,7678 |
| + índices NOAA robustos | Niño 1+2/3.4/4, ONI, SOI, dipolo do Atlântico | +2,16% | 1,7504 |
| + CFSv2 (NCEP) | primeira previsão dinâmica | +7,20% | 1,6322 |
| + GFDL-SPEAR, NASA-GEOSS2S | MME de 3 | +10,14% | 1,5812 |
| + lag-ensemble | previsão emitida 1 mês antes | +10,27% | 1,5785 |
| + CanSIPS (ECCC) | IC3 + IC4 costurados | +12,42% | 1,5495 |
| + SEAS5 (ECMWF) | via Copernicus CDS | +14,75% | 1,5078 |
| + UKMO, Météo-France, DWD, CMCC | via Copernicus CDS | +14,94% | 1,4992 |
| + learning rate 0,02, 1200 árvores | mesmo modelo, ajuste mais lento | +15,15% | 1,4933 |
| **+ ridge pontual combinado** | MOS local + ML global | **+15,98%** | **1,4890** |

**Correlação de anomalia ao longo da trajetória:** 0,19 → 0,20 → 0,37 → 0,44 → 0,48 → 0,52 → 0,53.

A habilidade individual de cada sistema dinâmico, medida como correlação entre sua anomalia prevista e a observada (1982–2022):

| sistema | centro | r |
|---|---|---|
| SEAS5 | ECMWF | **0,514** |
| CanSIPS | ECCC | 0,445 |
| GloSea6 | UKMO | 0,412 |
| SPS3.5 | CMCC | 0,374 |
| CFSv2 | NCEP | 0,362 |
| SPEAR | GFDL | 0,356 |
| System 8 | Météo-France | 0,325 |
| GEOSS2S | NASA | 0,305 |
| GCFS2.1 | DWD | 0,291 |

O modelo estatístico combinado (r=0,528) supera o melhor sistema individual (0,514) e a média simples de todos (0,504). É o que MOS deve fazer: ponderar os modelos por região e época em vez de tratá-los como iguais.

---

## 4. Metodologia

### 4.1 Decomposição

```
precipitação = climatologia(ponto, mês) + anomalia
```

O modelo aprende só a anomalia. A climatologia é um `groupby` exato; deixar árvores reaprendê-la desperdiça capacidade no que já está resolvido. A decomposição também dá a métrica que importa: se o modelo não bate a climatologia, não há habilidade.

### 4.2 Climatologia (`01_climatologia.py` a `01d_climatologia_final.py`)

Validada em três cortes temporais independentes (2002, 2007, 2012), sempre ajustando até o corte e avaliando nos anos seguintes:

- **Comprimento da janela.** Curva em U: 5 anos é ruidoso, 73 anos é defasado. Ótimo em 20–28 anos; adotados 24.
- **Mistura** 0,50 da janela curta com 0,50 do período completo: reduz variância amostral sem perder recência.
- **Tendência local encolhida.** Ajusta `chuva ~ a + b·ano` por ponto e mês e extrapola com fator k=0,15. k=1 (tendência cheia) leva o RMSE de 1,84 para 2,03: as inclinações locais são majoritariamente ruído amostral. A superfície de resposta é um platô, não uma agulha.

### 4.3 Atributos de reanálise (`02` a `04`)

Especificação completa em `dados_processados/features.json`. Quatro grupos:

| grupo | n | conteúdo |
|---|---|---|
| estáticas | 4 | lat, lon, climatologia do ponto no mês-alvo, climatologia anual |
| sazonais | 2 | sin/cos do mês-alvo |
| locais | 12 | anomalias das 9 covariáveis + transporte de umidade (`q·u`, `q·v`, convergência) |
| grande escala | 205 | 8 EOFs por grupo de variáveis com defasagens 1–3 e tendência, mais um índice de pressão do Pacífico |

Todas as covariáveis entram como **anomalias** em relação à climatologia do mês calendário. **Excluída deliberadamente:** a precipitação do mês anterior — existe no treino, mas no teste só há `tp_ultima_obs`, constante em dezembro de 2022 nos 24 meses.

**Validação dos EOFs.** O modo `pc_slp_2` separa meses DJF de El Niño de meses DJF de La Niña em 1,88 desvios-padrão. Um índice manual de pressão no Pacífico dá 1,83 com o mesmo sinal.

### 4.4 Índices de TSM (`09_indices_noaa.py`)

Oito séries mensais do NOAA/PSL. Validação: Niño 3.4 correlaciona −0,52 com o índice de pressão do Pacífico extraído da reanálise, e −0,68 com o SOI. **TNA e TSA absolutos foram descartados** após diagnóstico de deslocamento (seção 6.2). Fica o dipolo TNA−TSA.

### 4.5 Modelo global: LightGBM

Um único modelo sobre a anomalia, para todos os pontos:

```python
objective="l2", num_leaves=63, learning_rate=0.02, min_child_samples=200,
feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
n_estimators=1200, random_state=42
```

Ajuste final: média de 5 sementes, cada uma sobre 5 milhões de linhas amostradas (subamostragem espacial de fator 3: 8.787 dos 78.561 pontos; a previsão é feita na grade cheia). A configuração original (`lr=0,05`, 400 árvores) foi mantida até a penúltima rodada; a versão mais lenta ganhou +0,21 p.p. em todos os cinco folds.

### 4.6 Modelo local: ridge por ponto e combinação (`17_mos_pontual.py`)

O LightGBM é **global**: uma árvore para todos os pontos, com lat/lon como features. A formulação operacional de MOS é o oposto — uma regressão **local** por ponto de grade, com coeficientes próprios daquele lugar para cada sistema dinâmico.

As duas visões erram de jeitos diferentes. O ridge pontual não consegue usar o estado atmosférico observado (teria features demais para ~490 amostras por ponto); o LightGBM não calibra cada ponto com a precisão de uma regressão dedicada.

- Ridge por ponto (8.787 regressões fechadas, λ=30) sobre 31 colunas: anomalias dos nove sistemas em dois leads, médias, indicadores de disponibilidade, sin/cos do mês.
- Sozinho: r=0,506, pior que o LightGBM (0,525). **Correlação entre as duas previsões: 0,853.**
- Combinação `w·lgbm + (1−w)·ridge` com w validado por fold: **0,63 em todos os cinco** (0,64, 0,63, 0,64, 0,63, 0,63).
- Ganho validado sobre o LightGBM sozinho: **+0,83 p.p.** — o maior ganho de modelagem do projeto.

Previsão final: `climatologia_v3 + α · [w·anom_lgbm + (1−w)·anom_ridge]`, com α = 1,044 (média dos folds).

---

## 5. Previsão dinâmica como preditor (MOS)

### 5.1 O que é e por que é legítimo

*Model Output Statistics* é a técnica padrão de pós-processamento em meteorologia operacional: a saída de um modelo numérico é usada como entrada de um modelo estatístico, que aprende a corrigir viés sistemático e a combiná-la com outras fontes. A previsão oficial de qualquer centro — CPTEC incluso — é produto de MOS.

O regulamento da competição permite dados externos desde que públicos, gratuitos e igualmente acessíveis. O esclarecimento dos organizadores fixa o critério temporal: para prever o mês T, vale qualquer informação que em tese estaria disponível até o fim de T−1, incluindo dados externos, índices climáticos e informação de fora do domínio. Todos os sistemas usados atendem aos dois critérios.

Reconhecemos que a formulação do desafio comporta a leitura de que previsões dinâmicas estão fora do espírito. Por isso o repositório mantém **as duas linhas separadas e documentadas**: o melhor resultado sem previsão dinâmica (1,7504) e o resultado com ela (1,4890). A banca vê o valor de cada componente e julga.

### 5.2 Corte temporal: inicialização, não publicação

O alvo é o mês *T* = *M*+1; as features de reanálise são de *M*. As previsões dinâmicas usadas são:

- **lead 0,5**: para o alvo *T*, com condição inicial do fim de *M*. No CFSv2 (NMME), os 24 membros são inicializados ao longo dos 30 dias de *M*. No SEAS5 e nos sistemas do C3S, a inicialização é às 00 UTC do dia 1 de *T* — o instante final de *M*; a análise contém observações até ali e nenhuma de dentro de *T*.
- **lead 1,5**: emitida no início de *M*, para o mesmo alvo *T*.

Uma previsão é função determinística de sua condição inicial. Se a informação de base existe ao fim de *M*, a previsão existe *em tese* ao fim de *M* — o mesmo critério que torna válidos os índices mensais da NOAA, publicados dias depois do mês a que se referem. É a convenção padrão em verificação de previsão: o lead se conta da inicialização, não da publicação.

Verificação: o lead 1,5 tem sempre r menor que o lead 0,5 do mesmo sistema (CFSv2: 0,199 contra 0,362). Se o realinhamento estivesse deslocado em um mês, a correlação cairia para perto de zero.

### 5.3 Os nove sistemas e como cada um foi tratado

| sistema | fonte | hindcast | operacional | tratamento especial |
|---|---|---|---|---|
| CFSv2 | IRI/NMME | 1982–2010 | 2011– | operacional em `PENTAD_SAMPLES` para casar com o hindcast |
| GFDL-SPEAR | IRI/NMME | 1991–2020 | 2021– | — |
| NASA-GEOSS2S | IRI/NMME | 1981–2017 | 2017– | — |
| CanSIPS | IRI/NMME | IC3: 1980–2020; IC4: 1991–2020 | IC3 até jun/2024; IC4 de jul/2024 | duas versões costuradas, cada uma anomalizada contra o próprio hindcast; o hindcast do IC4 foi obtido em três blocos de 10 anos porque o IRI encerra requisições acima de 9 minutos |
| SEAS5 | CDS | s5: 1981–2016 | s5 até out/2022; s51 de nov/2022 | hindcast só existe como `monthly_mean` com 25 membros (média no loader); s51 vem em grade 75×65 deslocada em relação ao hindcast 76×66 e é reamostrado |
| GloSea6 | CDS | s603: 1993–2016 | s601 2020–22; s603 2023–24 | — |
| Météo-France | CDS | s8: 1993–2016 | s7 2017–19; s8 2020–24 | — |
| GCFS2.1 | CDS | s21: 1993–2016 | s21 2020–24 | — |
| SPS3.5 | CDS | s35: 1993–2016 | s35 2020–24 | — |

Cada sistema é anomalizado contra a climatologia do **próprio hindcast**, por ponto e mês calendário. Onde um sistema não existe (treino anterior ao hindcast), a coluna recebe zero e um indicador `_disp` marca a ausência. Features derivadas: média por lead, média geral, contagem de membros, desvio entre membros, diferença entre leads.

### 5.4 O que a previsão dinâmica não substitui

O grupo `nmme` responde por 41% do ganho do LightGBM. A grande escala da reanálise ainda responde por 28%, as estáticas por 13%, as locais por 10%, os índices de TSM por 7%. O modelo continua usando o estado atmosférico observado para modular a previsão dinâmica por região.

---

## 6. Validação

### 6.1 Walk-forward expansivo

```
fold 1: ajusta ≤1997, avalia 1998–2002
fold 2: ajusta ≤2002, avalia 2003–2007
fold 3: ajusta ≤2007, avalia 2008–2012
fold 4: ajusta ≤2012, avalia 2013–2017
fold 5: ajusta ≤2017, avalia 2018–2022
```

Sempre prevendo o futuro a partir do passado. A climatologia v3 é **reajustada dentro de cada fold**; na primeira versão do pipeline ela era global e o baseline ficava otimista. O bootstrap para intervalo de confiança reamostra **por mês**, não por linha.

### 6.2 Onde o CV errou, e o que se aprendeu

Entre as submissões 4 e 7, o CV melhorou em cinco versões consecutivas e o placar piorou em quatro. O diagnóstico (`10_diagnostico_ood.py`): fração do teste fora do intervalo [p1, p99] do treino, por feature. TNA com 58% das linhas de 2023 e **75% de 2024** fora da distribuição. Árvores não extrapolam.

Correções: descartar TNA/TSA absolutos e manter o dipolo; anomalia móvel de 30 anos antes de decompor TSM em grade; alinhamento de grade nos sistemas do Copernicus. Depois disso o CV voltou a prever o placar — e nas últimas seis rodadas acertou a direção em todas.

### 6.3 Sobre overfitting

- O fold ≤2017 avalia 2018–2022 com previsões **operacionais puras**, não hindcast, e dá ganho de 15,9% — no meio dos outros folds.
- A magnitude é a da literatura: r de 0,3 a 0,5 para previsão dinâmica de precipitação tropical a um mês.
- Nenhum parâmetro foi escolhido pelo placar. Os vinte e três envios foram hipóteses formuladas antes, com CV medido antes.

### 6.4 As duas submissões finais

Diferem em uma única coisa: a presença do ridge pontual. É o componente mais recente e o único com menos histórico de validação. O hedge original em α (principal contra conservadora) perdeu sentido quando o α mínimo entre folds subiu para 0,95 — o modelo com nove sistemas é confiante o bastante em regime fraco para dispensá-lo.

### 6.5 Avaliação fora da amostra (`14_avaliacao.py`)

Todas as métricas abaixo vêm de 2,6 milhões de previsões fora da amostra (1998–2022).

**Overfitting.** Correlação dentro do treino 0,601, na validação 0,525. Gap de 0,076 — faixa normal para gradient boosting regularizado (0,05–0,15).

**Robustez temporal.** Ganho positivo sobre a climatologia em **25 de 25 anos**. Mínimo +8,4% (2013), máximo +23,7% (1998). Média 14,7%, mediana 14,1%: habilidade distribuída, não concentrada.

**Por regime de ENOS.**

| regime | anos | RMSE clim. | RMSE modelo | ganho | r |
|---|---|---|---|---|---|
| El Niño | 6 | 1,894 | 1,569 | +17,2% | 0,561 |
| neutro | 10 | 1,821 | 1,549 | +15,0% | 0,526 |
| La Niña | 9 | 1,811 | 1,572 | +13,2% | 0,496 |

Para 2024 — transição para La Niña — a expectativa é de 13–15% sobre a climatologia no conjunto privado, contra 19% em 2023.

**Por região.** Nordeste +23,4% (r 0,644), Amazônia norte +17,5%, Sul +16,4%, Centro-Oeste/Sudeste +15,3%, Patagônia +14,8%, oceano +14,6%, Argentina +14,3%, Andes/Peru +13,8%, Amazônia sul +10,9%.

**Por mês.** Novembro a maio: 14–18%. Junho a outubro: 9–14%. Menos variância e teleconexões mais fracas na estação seca.

**Calibração.** Por decil da anomalia prevista, a média observada acompanha a prevista com inclinação **1,016**. O decil mais seco prevê −1,54 e observa −1,55; o mais úmido prevê +1,89 e observa +1,95.

**Extremos.** Nos 10% mais secos, +19,9% sobre a climatologia; nos 10% mais úmidos, +19,0%.

**Métricas agregadas.**

| métrica | climatologia | modelo |
|---|---|---|
| RMSE (mm/dia) | 1,835 | **1,562** |
| MAE (mm/dia) | 1,087 | 0,925 |
| correlação de anomalia | — | 0,525 |
| MSSS | 0 | **0,276** |
| acerto de tercil | 33% | 58% |

MSSS de 0,15 a 0,30 é considerado bom para precipitação a um mês; a maioria dos centros operacionais reporta abaixo de 0,20 sobre a América do Sul.

### 6.6 Verificação do conjunto privado sem rótulos (`18_sanidade_2024.py`)

O placar público valida 2023. O segundo semestre de 2024 tem um componente que nunca passou por validação: a costura IC3→IC4 do CanSIPS. Sem ler nenhuma observação de 2023 ou 2024, verificou-se:

- A anomalia do CanSIPS em jul–dez/2024 (IC4) tem desvios de 0,71 a 1,05 — indistinguíveis dos 23 meses anteriores (0,72 a 1,16). Não há descontinuidade de escala na costura.
- A previsão final tem amplitude consistente entre os dois anos: desvio 1,067 em 2023, 1,026 em 2024. Nenhum mês fora de 0,7–1,4.
- 0,47% das linhas acima de 3× a climatologia, concentradas em 2023 e em pontos de climatologia próxima de zero.
- As duas finais diferem em 0,109 mm/dia em média, idêntico em 2023 e 2024: o ridge pontual se comporta igual nos dois anos.

---

## 7. Hipóteses testadas e rejeitadas

| hipótese | teste | resultado | veredito |
|---|---|---|---|
| Tendência local cheia (k=1) | 3 cortes | RMSE 2,03 vs 1,84 | rejeitada: ruído amostral |
| Janela climatológica de 5–10 anos | 3 cortes | pior que 20–28 em todos | rejeitada |
| Remoção de tendência das features | holdout 2013–22 | empate em RMSE | não adotada |
| Persistência de precipitação | inspeção | `tp_ultima_obs` constante no teste | impossível |
| 12 ou 16 EOFs por grupo | 3 folds | r cai de 0,160 para 0,153 | rejeitada |
| Grupo de EOF para cobertura de nuvens | ablação | −0,10 p.p. | rejeitada |
| Defasagens de 6/9/12 meses | ablação | −0,05 p.p. | rejeitada |
| Peso amostral por recência (τ=40) | 5 folds + 2 pares no placar | +0,18 p.p. no CV; −0,009 e −0,011 no placar | **rejeitada pelo placar**: descarta os El Niños fortes do passado |
| Ensemble LightGBM + Ridge + MLP (sem dinâmicos) | 5 folds | +2,124% vs +2,134% | rejeitada à época; superada pelo item 4.6 quando os dinâmicos entraram |
| TSM em grade (ERSST, 12 EOFs) | 5 folds + placar | +0,12 p.p. no CV; −0,011 no placar | rejeitada pelo placar |
| Mais folhas e mais árvores (127 folhas, 1000 árvores) | 5 folds | +9,98% vs +10,14% | rejeitada: espalha ganho em ruído |
| Busca de hiperparâmetros (Optuna) | 3 tentativas em 4,5 h | inviável no cluster | abandonada |
| Subamostragem espacial mais densa (stride 2) | 5 folds | r 0,520 vs 0,522 | rejeitada: densidade não é limitante |
| Níveis superiores do ERA5 (200/500 hPa, 48 EOFs) | 5 folds | +14,98% vs +14,94% | rejeitada: redistribui importância sem acrescentar informação |
| Índice MJO (RMM, BoM) | — | série disponível só até fev/2024 | não incorporada |
| α por mês calendário | LOYO sobre 25 anos | −0,04% validado | rejeitada: 12 parâmetros ajustam ruído |
| Arquiteturas de alta capacidade (GNN, U-Net, VLM) | razão parâmetros/amostras; experiência anterior com IMERG | — | descartadas sem teste |
| **Learning rate 0,02, 1200 árvores** | 5 folds | +15,15% vs +14,94%, todos sobem | **aceita** |
| **Ridge pontual combinado** | 5 folds, w validado | +0,83 p.p., w idêntico em todos | **aceita** |

---

## 8. Fontes de dados externas

Todas públicas, gratuitas e citadas. Os operacionais dos sistemas dinâmicos e os índices estão versionados em `dados_processados/indices_brutos/`; os hindcasts (270 MB a 400 MB por sistema) ficam fora do git e são reproduzidos por `baixar_seas5.py` e `baixar_c3s.py`.

- **NOAA/PSL Monthly Climate Indices** — https://psl.noaa.gov/data/timeseries/month/
- **NOAA ERSST v5** — Huang et al. (2017), doi:10.7289/V5T72FNM. Testado, não adotado.
- **NMME** — Kirtman et al. (2014), doi:10.1175/BAMS-D-12-00050.1. Via IRI Data Library.
- **ECMWF SEAS5** — Johnson et al. (2019), doi:10.5194/gmd-12-1087-2019. Via Copernicus CDS.
- **C3S multi-system** (UKMO, Météo-France, DWD, CMCC) — via Copernicus CDS. Gerado usando informação do Copernicus Climate Change Service; nem a Comissão Europeia nem o ECMWF são responsáveis pelo uso feito.
- **ERA5 em níveis de pressão** — Hersbach et al. (2020). Via Copernicus CDS. Testado, não adotado.

**Ambiente:** o ambiente de processamento não tem acesso externo aos três provedores. Todos os downloads foram feitos numa máquina com internet e transferidos por `scp`.

---

## 9. Pipeline e reprodução

```
src/
  00_inspecao.py               inventário, unidades, orientação
  00b_alinhamento.py           verificação do alinhamento temporal
  01_climatologia.py           climatologia v1
  01b/01c/01d_climatologia_*   janela, mistura, tendência → v3
  02_anomalias.py              anomalias, transporte de umidade, campos para EOF
  03_eofs.py                   PCA no treino, projetada no teste
  04_features.py               tabela de atributos (223 colunas)
  04b_detrend.py               remoção de tendência (não adotada)
  05_modelo.py                 walk-forward, bootstrap, importância
  06_submissao.py              LightGBM final, α, sementes, submissão
  07_experimento_eofs.py       8/12/16 componentes
  07b_ablacao.py               nuvens e memória longa
  08_ensemble.py               recência; LightGBM + Ridge + MLP
  09_indices_noaa.py           índices de TSM, modo robusto
  10_diagnostico_ood.py        deslocamento treino/teste
  11_sst_grade.py              EOFs de ERSST (não adotado)
  12_optuna.py                 busca de hiperparâmetros (abandonada)
  13_nmme.py / 13b / 13c       CFSv2, MME de 3, lag-ensemble
  13d_mme_ext.py               CanSIPS + SEAS5 + C3S  ← features finais
  14_avaliacao.py              avaliação fora da amostra
  15_niveis_mjo.py             ERA5 200/500 hPa e MJO (não adotados)
  16_alfa_mensal.py            α por mês (não adotado)
  17_mos_pontual.py            ridge por ponto + combinação  ← modelo final
  18_sanidade_2024.py          verificação do privado sem rótulos
  baixar_seas5.py / baixar_c3s.py / baixar_era5_niveis.py   downloads (fora do cluster)
```

### Reprodução da submissão final

```bash
conda create -n hack python=3.12 -y && conda activate hack
conda install -c conda-forge numpy=1.26.4 pandas=2.2.2 xarray=2023.6.0 \
    scipy=1.13.1 scikit-learn=1.5.1 pyarrow=16.1.0 h5py=3.11.0 -y
pip install lightgbm==4.7.0 netCDF4 cdsapi

# dados externos (fora do cluster; ver seção 8 e os comandos em cada script)
python3 baixar_seas5.py && python3 baixar_c3s.py

python src/01_climatologia.py && python src/01d_climatologia_final.py
python src/02_anomalias.py 3 && python src/03_eofs.py 8 && python src/04_features.py completo
ROBUSTO=1 python src/09_indices_noaa.py
python src/13d_mme_ext.py
LR=0.02 N_ARVORES=1200 FEATURES=mme3 RECENCIA=0 N_SEMENTES=5 python src/06_submissao.py
FEATURES=mme3 python src/14_avaliacao.py
python src/17_mos_pontual.py saidas/sub07_mme3_norec_t1200l63_s5.csv
```

Semente fixa (42) em todos os pontos estocásticos. Executado em nó de CPU AMD EPYC, 16 núcleos alocados, 1,5 TB de RAM (as matrizes de treino ocupam até 23 GB em memória). Sem GPU.

---

## 10. Limitações conhecidas

1. **Versões operacionais anteriores a 2020** dos sistemas do Copernicus vêm de versões diferentes das que cobrem 2023–24. Anomalizadas contra o hindcast da versão mais nova. Afeta apenas o treino.
2. **SEAS5 operacional é sistematicamente mais úmido que o hindcast** (3,75 contra 3,45 mm/dia). A anomalia absorve o viés médio.
3. **JMA CPS3** não entrou: os arquivos operacionais do CDS vieram sem dimensão temporal.
4. **Viés recente positivo.** De 2017 a 2022 o modelo prevê +0,05 a +0,13 mm/dia acima do observado. Pequeno (5% do RMSE), mas consistente.
5. **PCA de reanálise ajustada uma vez** em 1940–2022, não por fold. Não supervisionada; efeito desprezível.
6. **A precipitação do mês anterior** não está disponível no teste.

---

## 11. Referências

- Glahn, H. R. & Lowry, D. A. (1972). The Use of Model Output Statistics (MOS) in Objective Weather Forecasting. *J. Appl. Meteor.*, 11, 1203–1211.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.*, 146, 1999–2049.
- Huang, B. et al. (2017). ERSSTv5. *J. Climate*, 30, 8179–8205.
- Johnson, S. J. et al. (2019). SEAS5: the new ECMWF seasonal forecast system. *Geosci. Model Dev.*, 12, 1087–1117.
- Ke, G. et al. (2017). LightGBM. *NeurIPS*.
- Kirtman, B. P. et al. (2014). The North American Multimodel Ensemble. *Bull. Amer. Meteor. Soc.*, 95, 585–601.
- Murphy, A. H. (1988). Skill scores based on the mean square error. *Mon. Wea. Rev.*, 116, 2417–2424.
