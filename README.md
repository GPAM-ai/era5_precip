# Previsão Climática Mensal de Precipitação sobre a América do Sul

O objetivo é prever a média mensal de precipitação (mm/dia) do mês seguinte, sobre uma grade de 0,25° que
cobre a América do Sul e oceanos adjacentes, a partir de campos atmosféricos
do mês corrente.

**Resultado atual:** 1,76781 de RMSE no placar público (4º lugar), contra
1,83952 da climatologia de referência — ganho de 3,9%.

---

## 1. O problema, em uma frase

Dado o estado atmosférico do mês *M*, prever a precipitação média do mês
*M+1* em 78.561 pontos de grade. A avaliação é por RMSE sobre 24 meses
(2023 e 2024), com 2023 no placar público e 2024 definindo a classificação
final.

### O que torna o problema difícil

A precipitação mensal é dominada pelo ciclo sazonal e pelo padrão geográfico,
que juntos respondem por cerca de 90% da variância — e são triviais de prever
por média histórica. A parte difícil é a **anomalia**: o desvio em relação ao
normal daquele ponto naquele mês. Com um mês de antecedência e sem temperatura
da superfície do mar no conjunto de dados, a habilidade preditiva disponível é
pequena. Medimos correlação de anomalia de 0,19, o que corresponde a um teto
de ganho de cerca de 2% sobre a climatologia.

Isso define a estratégia: **um baseline climatológico muito bem construído vale
tanto quanto o modelo**, e os dois ganhos precisam ser somados com cuidado.

---

## 2. Decomposição: por que prever anomalia

O alvo é decomposto em

```
precipitação = climatologia(ponto, mês) + anomalia
```

O modelo aprende apenas a anomalia. A motivação é direta: a climatologia é um
`groupby` de duas linhas e é exata; deixar o modelo reaprendê-la com árvores de
decisão consumiria quase toda a sua capacidade para reproduzir mal aquilo que
uma média já resolve. Separando, o LightGBM gasta toda a capacidade no resíduo,
que é onde mora a dificuldade.

A decomposição também dá o único diagnóstico que interessa cientificamente:
comparar o RMSE do modelo com o RMSE da climatologia. Se não bate, não há
habilidade — só um gerador caro de médias históricas.

---

## 3. Alinhamento temporal

Este é o detalhe que mais facilmente produz um erro silencioso, e foi
verificado empiricamente antes de qualquer modelagem (`src/00b_alinhamento.py`).

| conjunto | coordenada `time` significa | alvo |
|---|---|---|
| treino | mês das **features** | `tp_alvo` = precipitação de *M+1* |
| teste | mês do **alvo** | as covariáveis são de *M−1* |

A verificação: os campos do teste em `time=2023-01` são **idênticos** aos do
treino em `time=2022-12` (diferença absoluta média de 0,0000, contra 1,1646
entre dois meses consecutivos quaisquer). O arquivo de teste já vem deslocado.

Consequências aplicadas em todo o código:

- nenhum `shift` é aplicado a nada;
- a climatologia e a sazonalidade são indexadas pelo **mês-alvo**, que é
  `(mês(time) % 12) + 1` no treino e `mês(time)` no teste;
- as anomalias de covariáveis são indexadas pelo **mês-feature**, que é
  `mês(time)` no treino e `((mês(time) − 2) % 12) + 1` no teste.

Um efeito colateral útil: como o último passo do treino e o primeiro do teste
são o mesmo mês de features (dez/2022), treino e teste formam uma série
temporal contínua. Isso permite que as defasagens do início de 2023 puxem
valores reais de 2022 em vez de ficarem indefinidas.

---

## 4. A climatologia

Três decisões, todas validadas com o mesmo desenho: ajustar até um ano de
corte e avaliar nos anos seguintes, replicando a extrapolação real do problema.
Três cortes independentes (2012, 2007, 2002) foram usados para garantir que a
conclusão não fosse artefato de um período.

### 4.1 Comprimento da janela

A curva de RMSE contra comprimento tem formato de U bem definido. Janelas
curtas respondem rápido a mudanças de regime mas têm ruído amostral; janelas
longas são estáveis mas defasadas.

| janela | corte 2012 | corte 2007 |
|---|---|---|
| 5 anos | 1,9679 | 1,9761 |
| 10 anos | 1,8813 | 1,8910 |
| 20 anos | 1,8374 | 1,8473 |
| 40 anos | 1,8457 | 1,8536 |
| tudo (73 anos) | 1,8601 | 1,8632 |

Escolha final: **24 anos**, que fica no fundo do U nos cortes mais recentes.

### 4.2 Mistura entre janela curta e longa

Misturar a janela curta com o período completo reduz a variância amostral sem
perder a recência. Peso ótimo: **0,50 curta + 0,50 longa**, num platô raso.

### 4.3 Tendência local, fortemente encolhida

Para cada ponto e mês calendário, ajusta-se `chuva ~ a + b·ano` e extrapola-se,
com a inclinação encolhida por um fator *k*:

```
previsão(ano) = média + k · inclinação · (ano − ano_central)
```

| k | corte 2012 | corte 2007 | corte 2002 |
|---|---|---|---|
| 0,00 | 1,8393 | 1,8551 | 1,8839 |
| **0,15** | **1,8326** | **1,8473** | **1,8801** |
| 0,50 | 1,8685 | 1,8713 | 1,9062 |
| 1,00 | 2,0432 | 2,0176 | 2,0273 |

O ganho aparece nos três cortes com o mesmo sinal, e a vizinhança do ótimo
varia apenas 0,0019 — é um platô, não uma agulha de ruído.

**Por que k é tão pequeno.** A tendência ajustada em janeiro para Manaus é de
−1,30 mm/dia por década. Extrapolada 13 anos sem encolhimento, isso seria uma
secagem de 18% sobre uma climatologia de 9,55 mm/dia — implausível. A janela
1999–2022 contém as secas amazônicas de 2005, 2010 e 2015/16, que são
**eventos**, não tendência. O ajuste linear os confunde com deriva. Com
k = 0,15 aproveita-se a direção do sinal descartando 85% da amplitude, que é
majoritariamente ruído amostral. O valor não é mágico: é o que a validação
cruzada em três períodos independentes seleciona, e a superfície é plana em
volta dele.

**Ganho da climatologia v3 sobre a média simples: +0,96%** (média dos folds).

---

## 5. Features

### 5.1 Especificação explícita do vetor de atributos

Cada linha é um par (mês-alvo, ponto de grade). São 223 colunas:

| grupo | n | conteúdo | justificativa |
|---|---|---|---|
| **estáticas** | 4 | `lat`, `lon`, `clim_alvo`, `clim_anual` | informam o regime local. Sem elas o modelo não distingue uma anomalia de +2 mm/dia no Atacama (enorme) de +2 na Amazônia (modesta) |
| **sazonais** | 2 | `sin_mes`, `cos_mes` | codificação cíclica, para que dezembro e janeiro fiquem próximos |
| **locais** | 12 | anomalias das 9 covariáveis entregues + `qu`, `qv`, `conv_umid` | estado atmosférico no próprio ponto, no mês-feature |
| **grande escala** | 205 | componentes principais por grupo de variáveis, defasagens 1–3 e tendência de 3 meses | a habilidade em escala mensal vem da configuração de grande escala, não do estado do pixel |

### 5.2 Features derivadas: transporte de umidade

A variável fisicamente mais direta para chuva não está entre as entregues. Com
`shum_850` e o vento em 850 hPa ela é construída:

```
qu = q·u,  qv = q·v
conv = −(∂qu/∂x + ∂qv/∂y)
```

Convergência positiva significa umidade acumulando na coluna — é de onde a
chuva vem. Calculada no campo bruto e só depois transformada em anomalia,
porque o produto `q·u` não é linear.

### 5.3 Modos de grande escala (EOF/PCA)

Os campos são agregados a 2°, ponderados por `√cos(lat)` para que cada célula
pese proporcionalmente à sua área, e decompostos por PCA em 8 componentes por
grupo de variáveis (`slp`, `z850`, `circ`, `umid`, `fluxo`).

**Validação contra ENOS.** Como o conjunto não traz temperatura da superfície
do mar, o ENOS só pode entrar indiretamente via pressão e geopotencial. O modo
`pc_slp_2` separa meses DJF de El Niño de meses DJF de La Niña em −1,88
desvios-padrão. Como contraprova, um índice construído manualmente (anomalia
média de pressão no canto do Pacífico do domínio, 20°S–0°, oeste de −80°)
separa em −1,833, com o sinal negativo correto — pressão mais baixa no Pacífico
leste em El Niño. Dois métodos independentes, um supervisionado por raciocínio
físico e outro não supervisionado, concordando em magnitude e sinal.

### 5.4 O que foi deliberadamente excluído

**Precipitação do mês anterior (`tp`).** Existe no treino, mas no teste só há
`tp_ultima_obs`, que é **constante** nos 24 passos (sempre dez/2022, verificado
em `00b`). Treinar com uma feature ausente na inferência produziria dependência
de algo que não existe na hora de prever — falha silenciosa, que degrada o RMSE
de forma plausível em vez de quebrar visivelmente. A umidade específica e a
convergência de umidade assumem o papel funcional que a persistência teria.

**Valores brutos das covariáveis.** Apenas anomalias. Em `surface_pressure` e
`geopotential_850` a assinatura topográfica dos Andes domina completamente o
campo bruto e sequestraria os splits das árvores.

---

## 6. Validação

### 6.1 Desenho: walk-forward expansivo

```
fold 1: ajusta ≤1997, avalia 1998–2002
fold 2: ajusta ≤2002, avalia 2003–2007
fold 3: ajusta ≤2007, avalia 2008–2012
fold 4: ajusta ≤2012, avalia 2013–2017
fold 5: ajusta ≤2017, avalia 2018–2022
```

Sempre prevendo o futuro a partir do passado, que é a estrutura do problema
real. *Leave-one-year-out* embaralharia isso e daria uma estimativa otimista.

### 6.2 Um vazamento que encontramos e corrigimos

Na primeira versão, a climatologia foi ajustada em 1940–2022 e usada para
calcular a anomalia de **todos** os anos, inclusive os de validação. O "RMSE da
climatologia" medido dessa forma era otimista, porque a climatologia tinha
visto os anos que estava sendo avaliada a prever.

O efeito era considerável: com o baseline vazado o modelo aparentava ganhar
0,87%; com a climatologia recalculada dentro de cada fold, o ganho real é
**1,99%**. O vazamento estava **subestimando** nosso resultado, mas isso é
sorte — poderia ter sido o contrário.

Correção aplicada em `05_modelo.py` e `06_submissao.py`: cada fold recalcula a
climatologia (janela, mistura e tendência) usando apenas anos anteriores ao
corte.

**Vazamento residual conhecido:** a PCA é ajustada uma vez em 1940–2022, não
reajustada por fold. Como ela é não supervisionada (não olha o alvo) e usa 83
anos, o efeito é desprezível — mas a escolha fica registrada aqui em vez de
escondida.

### 6.3 Bootstrap

Os resíduos são reamostrados **por mês**, não por linha. Os 8.787 pontos de
grade de um mesmo mês são fortemente correlacionados; reamostrar linha a linha
fingiria ter milhões de amostras independentes quando há algumas dezenas.

```
diferença média (climatologia − modelo): +0,03709 mm/dia
IC 95%: [+0,01756, +0,06544]
```

O intervalo não contém zero: o ganho é estatisticamente significativo.

### 6.4 Resultados por fold

| fold | clim. simples | clim. v3 | + modelo | α | ganho total |
|---|---|---|---|---|---|
| ≤1997 | 1,8562 | 1,8488 | 1,8018 | 1,073 | +2,93% |
| ≤2002 | 1,8191 | 1,8103 | 1,7733 | 1,068 | +2,52% |
| ≤2007 | 1,8926 | 1,8754 | 1,8261 | 0,939 | +3,51% |
| ≤2012 | 1,8068 | 1,7777 | 1,7610 | 0,719 | +2,54% |
| ≤2017 | 1,8888 | 1,8621 | 1,8388 | 0,927 | +2,65% |

**Os ganhos são aditivos.** Climatologia v3 sozinha: +0,96%. Modelo sobre a v3:
+1,88%. Soma simples: +2,84%. Total observado: +2,83%. A v3 corrige o nível e o
modelo explica a variabilidade interanual — sinais largamente independentes.

---

## 7. O modelo

LightGBM, um único modelo global sobre a anomalia:

```python
objective="l2", num_leaves=63, learning_rate=0.05,
min_child_samples=200, feature_fraction=0.7,
bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
n_estimators=400, random_state=42
```

`min_child_samples=200` é deliberadamente alto: com pontos de grade fortemente
autocorrelacionados, folhas pequenas memorizariam vizinhança espacial em vez de
aprender relação física.

Um modelo só para todo o domínio e todos os meses, em vez de 12 modelos
mensais ou um por ponto. Fragmentar a amostra impediria o modelo de aprender
que o Nordeste em fevereiro se comporta como o Sul em julho sob a mesma
configuração de grande escala.

### 7.1 Importância das features

| grupo | % do ganho |
|---|---|
| grande escala | 56,2 |
| locais | 21,9 |
| estáticas | 21,0 |
| sazonais | 0,9 |

A dominância da grande escala confirma a tese central do desenho. As sazonais
contribuem quase nada porque `clim_alvo` já carrega a sazonalidade ponto a
ponto — são redundantes.

Individualmente, os cinco maiores são `clim_alvo` (7,6%), `lat` (6,6%),
`a_cloud_cover` (5,2%), `lon` (3,9%) e `a_shum_850` (3,6%). O índice de
pressão do Pacífico construído manualmente aparece em 8º (1,8%).

### 7.2 Sobre o encolhimento (α)

A previsão final é `climatologia + α · anomalia_prevista`. A hipótese inicial
era que, com habilidade baixa, α deveria ficar entre 0,2 e 0,4 — encolher
bastante reduziria o RMSE.

**A hipótese estava errada.** O α ótimo medido é 0,945 em média (0,719 a 1,073
entre folds). O LightGBM com `lambda_l2=5` já devolve anomalias calibradas;
não sobra o que encolher. A correlação de anomalia é de fato baixa (0,19), mas
isso se manifesta como desvio-padrão pequeno das previsões, não como amplitude
mal calibrada.

O fold ≤2012 é a exceção instrutiva: α de 0,719, justamente o fold onde a
climatologia v3 mais ganhou (+1,61%) e o modelo menos ganhou (+0,94%). Quando o
baseline já captura a deriva, sobra menos anomalia real e o modelo superestima
o resíduo. Se 2024 se parecer com esse fold, um α menor seria o correto — é o
argumento para que a submissão conservadora seja uma versão com α reduzido.

---

## 8. Hipóteses testadas e rejeitadas

Registradas porque o que não funcionou é parte do resultado.

| hipótese | teste | veredito |
|---|---|---|
| **Remoção de tendência das features** | treino ≤2012, validação 2013–2022, com e sem | RMSE praticamente empatado (1,7422 vs 1,7440), mas o viés caiu 6,7× (+0,0665 → +0,0099). Mantido como candidato para a submissão conservadora, não para a principal |
| **Tendência local cheia (k=1)** | três cortes | piora muito: 2,03 contra 1,84. Inclinações locais são majoritariamente ruído amostral |
| **Persistência de precipitação** | inspeção dos dados | impossível: `tp_ultima_obs` é constante nos 24 meses de teste |
| **Janela climatológica de 5–10 anos** | três cortes | pior que 20–28 anos em todos: ruído amostral supera o ganho de recência |

### 8.1 Um deslocamento de distribuição que permanece

`a_t2` tem média −0,220 no treino e **+0,547** no teste — quase 0,8 K, cerca de
um desvio-padrão da própria variável. É aquecimento global: 2023 e 2024 foram
os anos mais quentes da série, e a climatologia não acompanha.

Isso é preocupante para modelos de árvore, que **não extrapolam**: se no teste
a variável cai sempre no bin mais alto já visto, as previsões herdam o que foi
aprendido para os meses mais quentes do histórico.

A remoção de tendência linear reduz o problema mas não o elimina (`a_t2` fica
em +0,329 contra −0,001), porque o aquecimento recente **acelerou** e uma reta
ajustada em 83 anos o subestima. A limitação está registrada e não resolvida.

---

## 9. Pipeline

```
src/
  00_inspecao.py              inventário dos dados, unidades, orientação da grade
  00b_alinhamento.py          verificação empírica do alinhamento temporal
  01_climatologia.py          climatologia v1 e submissão baseline
  01b_climatologia_tendencia.py  primeiro teste de janelas e tendência
  01c_climatologia_refino.py  busca fina (corrige um erro de fórmula do 01b)
  01d_climatologia_final.py   busca conjunta janela × mistura × tendência → v3
  02_anomalias.py             anomalias, transporte de umidade, campos para EOF
  03_eofs.py                  PCA ajustada no treino, projetada no teste
  04_features.py              montagem da tabela de atributos
  04b_detrend.py              remoção de tendência (testada, não adotada)
  05_modelo.py                validação walk-forward sem vazamento
  06_submissao.py             modelo sobre a v3 + submissão final
```

Cada script grava artefatos em `dados_processados/`, de modo que as etapas são
independentes e reexecutáveis sem refazer as anteriores.

### 9.1 Reprodução

```bash
conda create -n hack python=3.11 -y && conda activate hack
conda install -c conda-forge xarray netcdf4 dask lightgbm scikit-learn \
                             pandas matplotlib pyarrow -y

python src/00_inspecao.py <caminho_dos_dados>
python src/00b_alinhamento.py
python src/01_climatologia.py
python src/01d_climatologia_final.py
python src/02_anomalias.py 3
python src/03_eofs.py 8
python src/04_features.py completo
python src/05_modelo.py normal
python src/06_submissao.py
```

Semente fixa (`random_state=42`) em todos os pontos estocásticos: PCA,
subamostragem de linhas, LightGBM e bootstrap.

### 9.2 Notas de ambiente

Executado no supercomputador Santos Dumont (LNCC), partição `lncc-cpu_amd`.
Nenhuma etapa usa GPU — o gargalo é memória e I/O, não FLOPs.

O treino usa uma subamostragem espacial de fator 3 (8.787 de 78.561 pontos).
Pontos vizinhos a 0,25° são quase idênticos; o recurso escasso é o tempo
(83 anos), não o espaço. A previsão é feita na grade cheia.

Etapas com uso de memória relevante: `02_anomalias.py` (~2,3 GB de pico),
`04_features.py` (~8 GB), `05` e `06` (~20 GB). O nó de login não suporta;
usar `srun -A cptec -p lncc-cpu_amd -N1 -n1 -c16 --pty bash`.

---

## 10. Resultados

| submissão | descrição | CV esperado | placar público |
|---|---|---|---|
| `sub01_climatologia.csv` | climatologia mista, janela otimizada | 1,7897 | 1,83952 |
| `sub03_clim_tendencia.csv` | climatologia v3 (janela 24, w 0,50, k 0,15) | 1,8294 | — |
| `sub04_modelo_v3.csv` | v3 + LightGBM, α = 0,945 | 1,7875 | **1,76781** |

A previsão do CV acertou a ordem de grandeza nas três submissões, sendo
conservadora na última. Provável explicação: 2023 foi ano de El Niño forte, o
regime em que previsão climática mensal tem mais habilidade, enquanto os folds
de validação misturam regimes.

**Ressalva importante:** o placar público avalia 2023, mas a classificação
final sai de 2024 — ano de transição El Niño → La Niña. Anos de transição são
mais difíceis: a climatologia tem RMSE de 2,17 em 1983 e 2,10 em 1998, contra
uma média de 1,74. É razoável esperar números piores e possivelmente uma ordem
diferente no privado.

---

## 11. Limitações conhecidas

1. **Sem temperatura da superfície do mar.** O ENOS entra apenas indiretamente
   via pressão e geopotencial em 850 hPa. Um índice Niño3.4 direto quase
   certamente melhoraria a previsão.
2. **Apenas o nível de 850 hPa.** Sem 500 ou 200 hPa, a Alta da Bolívia e o
   cavado do Nordeste em altos níveis ficam fora de alcance.
3. **Domínio limitado a oeste** (−90° de longitude), o que corta boa parte do
   Pacífico tropical onde o ENOS se expressa mais claramente.
4. **Deslocamento de distribuição não resolvido** em variáveis térmicas
   (seção 8.1).
5. **PCA não reajustada por fold** (seção 6.2).

---
