# Model Documentation — Previsão Climática de Precipitação sobre a América do Sul

Documento no formato *Kaggle Winning Model Documentation Guidelines*. Complementa o `README.md` do repositório, que contém a discussão técnica completa.

---

## A1. Background on you/your team

**Competition:** Previsão Climática de Precipitação sobre a América do Sul — Hackathon WorCAP 2026 (INPE)
**Team name:** Rain-NP-Hard
**Public leaderboard:** RMSE 1,48895, 1º lugar (de N equipes)
**Private leaderboard:** *(preencher após a divulgação)*

**Membros** *(preencher para cada um: nome, cidade, e-mail, formação e experiência com ML/clima, motivação, horas dedicadas)*

| membro | localização | e-mail | formação / experiência | horas |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

**Motivação da equipe:** o problema une dois interesses do grupo — aprendizado de máquina aplicado a geociências e a pergunta de quanto sinal um modelo estatístico consegue extrair de um sistema caótico com um mês de antecedência. A experiência anterior do grupo com estimativa de precipitação por satélite (IMERG/GMI) informou várias decisões de projeto, em particular a de não testar arquiteturas de alta capacidade.

---

## A2. Summary

A solução decompõe a precipitação em climatologia mais anomalia e modela apenas a anomalia. Dois modelos são combinados: um **LightGBM global** (um modelo para toda a grade, com lat/lon como atributos) e uma **regressão ridge local por ponto de grade**. Ambos recebem como preditores as anomalias de reanálise ERA5 entregues pela competição, índices de temperatura da superfície do mar da NOAA, e — o componente decisivo — as previsões mensais de **nove sistemas de previsão sazonal** de sete centros operacionais (NCEP, GFDL, NASA, ECCC, ECMWF, UKMO, Météo-France, DWD, CMCC), usadas como *Model Output Statistics*. As duas formulações têm correlação de 0,85 entre suas previsões e são combinadas em peso fixo 0,63/0,37, validado por *walk-forward*. Ferramentas: Python 3.12, LightGBM 4.7, xarray, scikit-learn. Treino completo em ~40 minutos num nó de CPU (16 núcleos); predição em ~2 minutos.

---

## A3. Features Selection / Engineering

### Como as features foram selecionadas

Por validação *walk-forward* em cinco cortes temporais (1997, 2002, 2007, 2012, 2017), cada um avaliando os cinco anos seguintes. Uma feature ou grupo entrava se melhorasse o ganho sobre a climatologia na maioria dos folds; era rejeitada caso contrário. Dezoito hipóteses foram testadas; doze rejeitadas (tabela completa na seção 7 do README).

### As features mais importantes

Importância medida como fração do ganho total do LightGBM (`gain`), agregada por grupo:

| grupo | % do ganho | conteúdo |
|---|---|---|
| **previsões dinâmicas** | **41** | anomalias dos 9 sistemas em 2 leads, médias, dispersão entre membros |
| grande escala (reanálise) | 28 | 8 EOFs por grupo de variáveis (pressão, geopotencial, circulação, umidade, fluxo), defasagens 1–3 |
| estáticas | 13 | lat, lon, climatologia do ponto no mês-alvo, climatologia anual |
| locais (reanálise) | 10 | anomalias das 9 covariáveis + transporte de umidade |
| índices de TSM | 7 | Niño 1+2/3.4/4, ONI, SOI, PDO, dipolo do Atlântico |
| sazonais | 1 | sin/cos do mês |

Gráfico das 20 features individuais mais importantes: `figuras/importancia_top20.png` (gerado por `src/19_importancia_plot.py`).

### Features únicas ou não óbvias

1. **Previsões de modelos dinâmicos como preditores.** Cada sistema é anomalizado contra a climatologia do próprio *hindcast*, por ponto e mês calendário, o que remove seu viés médio antes da combinação. Onde um sistema não existe (treino anterior ao hindcast), a coluna recebe zero e um indicador marca a ausência.
2. **Dipolo do Atlântico (TNA−TSA) em vez das TSMs absolutas.** As absolutas estavam fora da distribuição de treino em 2023–24 (58% e 75% das linhas além do p99 histórico); o gradiente é robusto ao aquecimento uniforme.
3. **Transporte de umidade** (`q·u`, `q·v`, convergência) construído a partir de umidade específica e vento em 850 hPa, porque a variável fisicamente mais direta para chuva não foi entregue.
4. **Lag-ensemble:** a previsão emitida um mês antes para o mesmo alvo entra como coluna separada. A média simples com o lead mais antigo piora; o LightGBM aprende a ponderar.

### Features excluídas deliberadamente

A precipitação do mês anterior — existe no treino, mas no teste só há um valor constante. Treinar com uma feature ausente na inferência seria falha silenciosa.

---

## A4. Training Method(s)

### Alvo

`anomalia = precipitação − climatologia_v3(ponto, mês-alvo)`, onde a climatologia v3 mistura uma janela de 24 anos com o período completo e aplica tendência local encolhida (k=0,15). Reajustada dentro de cada fold do CV para evitar vazamento.

### Modelo global: LightGBM

```
objective=l2, num_leaves=63, learning_rate=0.02, n_estimators=1200,
min_child_samples=200, feature_fraction=0.7, bagging_fraction=0.7,
bagging_freq=1, lambda_l2=5, random_state=42
```

Ajuste final: média de 5 sementes, cada uma sobre 5 milhões de linhas amostradas do treino (subamostragem espacial de fator 3 na grade; a predição é feita na grade cheia de 78.561 pontos).

### Modelo local: ridge por ponto

8.787 regressões ridge (λ=30), uma por ponto de grade, sobre 31 colunas: anomalias dos sistemas dinâmicos em dois leads, médias, indicadores de disponibilidade, sin/cos do mês. Solução fechada; segundos de execução. Na grade cheia do teste, cada ponto usa os coeficientes do ponto de treino mais próximo.

### Combinação

`anomalia_final = 0,63 · anom_lgbm + 0,37 · anom_ridge`. O peso foi ajustado em quatro folds e aplicado no quinto; saiu idêntico (0,63–0,64) em todos.

### Encolhimento

`previsão = climatologia_v3 + α · anomalia_final`, com α = 1,044 (média dos folds; o modelo está calibrado — inclinação 1,016 por decil).

### Sem ensemble de outros aprendizes

LightGBM + Ridge + MLP foi testado antes da entrada dos dinâmicos e rejeitado. Com os dinâmicos, o ridge *pontual* (não global) trouxe diversidade real.

---

## A5. Interesting findings

**1. Informação vale mais que modelo — por uma ordem de grandeza.** A correlação de anomalia foi de 0,19 (só reanálise) para 0,53 (com nove sistemas dinâmicos). Dezoito hipóteses de modelagem testadas somaram menos de 3 pontos percentuais de ganho; a entrada de previsão dinâmica somou 13.

**2. A validação cruzada divergiu do placar por um motivo físico, não estatístico.** Entre as submissões 4 e 7, o CV melhorou cinco vezes e o placar piorou quatro. O diagnóstico mostrou que TNA e TSA absolutas estavam fora da distribuição de treino em 2023–24 (anos de oceano recorde). Árvores não extrapolam. A correção — usar o gradiente em vez das absolutas — resolveu, e o CV voltou a prever o placar nas seis rodadas seguintes.

**3. O peso por recência, que melhorou o CV em cinco de cinco folds, piorou o placar em dois pares comparáveis.** Descartava os El Niños fortes do passado, que são os análogos de 2023. Foi rejeitado pelo placar, não pelo CV.

**4. O SEAS5 do ECMWF sozinho (r=0,514) supera a média dos três modelos americanos do NMME (0,431).** E a média simples de nove sistemas (0,504) fica abaixo do SEAS5 isolado — média simples pesa igual um modelo de 0,51 e um de 0,29. A regressão aprende os pesos.

**5. MOS local e ML global são complementares.** O ridge por ponto sozinho é pior que o LightGBM (r 0,506 vs 0,525), mas a correlação entre as duas previsões é só 0,85. A combinação ganha 0,83 p.p. — o maior ganho de modelagem do projeto.

**O que não funcionou:** mais componentes de EOF, defasagens longas, mais capacidade no LightGBM, busca de hiperparâmetros, subamostragem espacial mais densa, níveis superiores do ERA5, TSM em grade, α por mês. Todos documentados com números no README.

---

## A6. Simple Features and Methods

Um modelo que atinge **~90% do desempenho final** com uma fração da complexidade:

- Climatologia simples (média por ponto e mês, todos os anos).
- LightGBM com parâmetros padrão sobre **apenas as anomalias dos cinco sistemas dinâmicos de maior habilidade** (SEAS5, CanSIPS, UKMO, CMCC, CFSv2) no lead 0,5, mais lat, lon e sin/cos do mês.
- Sem ridge, sem sementes múltiplas, sem índices de TSM, sem EOFs.

Estimativa: ganho de ~13% sobre a climatologia (contra 16% do modelo completo), RMSE público em torno de 1,56. Onze colunas em vez de 310.

Uma versão ainda mais simples — **só a média dos cinco sistemas, multiplicada por um α ajustado, somada à climatologia** — dá cerca de 80% do desempenho sem nenhum aprendizado de máquina. É a linha de base que qualquer solução com previsão dinâmica deve superar.

---

## A7. Model Execution Time

Hardware: nó de CPU AMD EPYC, 16 núcleos alocados, 1,5 TB de RAM (pico de uso ~25 GB). Sem GPU. Sistema Linux, Python 3.12 via conda.

| etapa | tempo |
|---|---|
| Preparação de dados (anomalias, EOFs, features, índices) | ~15 min |
| Montagem das matrizes com os sistemas dinâmicos (`13d`) | ~10 min |
| Validação walk-forward (5 folds) | ~10 min |
| Treino final LightGBM (5 sementes × 5M linhas × 1200 árvores) | ~22 min |
| Ridge pontual (8.787 regressões) + combinação | ~12 min |
| **Predição na grade cheia (1,9M linhas)** | **~2 min** |

Downloads externos (fora do cluster): NMME via IRI ~30 min; SEAS5 e C3S via Copernicus ~4 h de fila. Predição rápida com modelos salvos (`predict.py`): ~3 min.

---

## A8. References

- Glahn, H. R. & Lowry, D. A. (1972). The Use of Model Output Statistics (MOS) in Objective Weather Forecasting. *J. Appl. Meteor.*, 11, 1203–1211.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.*, 146, 1999–2049.
- Johnson, S. J. et al. (2019). SEAS5: the new ECMWF seasonal forecast system. *Geosci. Model Dev.*, 12, 1087–1117.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Kirtman, B. P. et al. (2014). The North American Multimodel Ensemble. *Bull. Amer. Meteor. Soc.*, 95, 585–601.
- Murphy, A. H. (1988). Skill scores based on the mean square error. *Mon. Wea. Rev.*, 116, 2417–2424.

**Dados:** NOAA/PSL climate indices; NMME via IRI Data Library; ECMWF SEAS5 e C3S multi-system via Copernicus Climate Data Store (contém informação modificada do Copernicus Climate Change Service; nem a Comissão Europeia nem o ECMWF são responsáveis pelo uso feito).

**Código:** https://github.com/*(organização)*/era5_precip — licença MIT.

---

## B. Code (checklist)

| item | arquivo | status |
|---|---|---|
| README com hardware, SO, software, como treinar, como prever, premissas | `README.md` + este documento | ✓ |
| `requirements.txt` com versões exatas | `requirements.txt` (gerado por `pip freeze`) | ✓ |
| Estrutura de diretórios | `directory_structure.txt` (`find . -type d`) | ✓ |
| Arquivo de configuração com caminhos | `settings.json` | ✓ |
| Ponto de entrada de preparação | `prepare_data.py` | ✓ |
| Ponto de entrada de treino | `train.py` | ✓ |
| Ponto de entrada de predição (modo rápido com modelo salvo) | `predict.py` | ✓ |
| Modelos treinados | `dados_processados/modelo_final_*.pkl`, `ridge_pontual.pkl` | ✓ |
| Semente fixa | 42 em todos os pontos estocásticos | ✓ |

**Premissas do código:** o diretório `dados_processados/` deve existir e ser gravável; `saidas/` idem. Os dados brutos da competição ficam no caminho indicado em `settings.json`. Os hindcasts dos sistemas dinâmicos (270–400 MB por sistema) não estão no repositório e são baixados por `baixar_seas5.py` e `baixar_c3s.py` numa máquina com acesso à internet.
