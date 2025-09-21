# MVP: *Machine Learning & Analytics*
**Autor:** Frederico Francesco Gribaudi Cardozo

# Sumário

- [1) Descrição do Problema](#1-descrição-do-problema)
  - [1.1 Seleção de Dados](#11-seleção-de-dados)
  - [1.2 Tipo de Problema](#12-tipo-de-problema)
  - [1.3 Atributos do Dataset](#13-atributos-do-dataset)
- [2) Importação das Bibliotecas Necessárias, Carga de Dados e Ambiente](#2-importação-das-bibliotecas-necessárias-carga-de-dados-e-ambiente)
- [3) Análise Exploratória](#3-análise-exploratória)
- [4) Tratamento de dados](#4-tratamento-de-dados)
  - [4.1 Nulos](#41-nulos)
  - [4.2 Outliers](#42-outliers)
  - [4.3 Definição do target, variáveis e divisão dos dados](#43-definição-do-target-variáveis-e-divisão-dos-dados)
- [5) Treinamento dos modelos](#5-treinamento-dos-modelos)
  - [5.1 Criação do baseline e modelos candidatos](#51-criação-do-baseline-e-modelos-candidatos)
  - [5.2 Separando novamente os dados](#52-separando-novamente-os-dados)
  - [5.3 Treino e avaliação rápida](#53-treino-e-avaliação-rápida)
  - [5.4 Otimização de hiperparâmetros](#54-otimização-de-hiperparâmetros)
    - [5.4.1 Tunning do LR](#541-tunning-do-lr)
    - [5.4.2 Tunning do SVC](#542-tunning-do-svc)
    - [5.4.3 Tunning do KNN](#543-tunning-do-knn)
    - [5.4.5 Definição dos modelos e criação dos pipelines](#545-definição-dos-modelos-e-criação-dos-pipelines)
  - [5.5 Salvando os pipelines treinados](#55-salvando-os-pipelines-treinados)
  - [5.6 Validação](#56-validação)
- [6) Finalização do Modelo](#6-finalização-do-modelo)
- [7) Conclusões e próximos passos](#7-conclusões-e-próximos-passos)


# **1) Descrição do Problema**

O conjunto de dados escolhido é multivariado e consiste em resultados de testes psicológicos, que categoriza pessoas em três tipos distintos de personalidade: introvertido, extrovertido e ambivertido. O objetivo principal é classificar o tipo de personalidade com base em resultados de testes psicológicos considerando vinte e nove características.

## 1.1 Seleção de Dados

O dataset escolhido estava listado como "trending dataset" no site kaggle, e tem sido usado por pessoas que estão aprendendo ou praticando *Machine Learning*. Link para o dataset:  [Introvert, Extrovert & Ambivert Classification](https://www.kaggle.com/datasets/miadul/introvert-extrovert-and-ambivert-classification/data)

## 1.2 Tipo de Problema

Este é um problema de **classificação supervisionada**. Dado um conjunto de características, o objetivo é prever a qual das três personalidades uma pessoa se encaixa.

## 1.3 Atributos do Dataset

O dataset contém 20,000 amostras, com  quantidades de amostras similares de cada uma das três personalidades, **34%** de extrovertidos, **33%** de introvertidos e **33%** de ambivertidos.

Possui trinta atributos:

* **personality type**: tipo de personalidade
* **social energy**: energia social
* **alone time preference**: preferência por solidão
* **talkativeness**: gostar de falar
* **deep reflection**: reflexão profunda
* **group comfort**: confortável ao estar em grupo
* **party liking**: gostar de festejar
* **listening skill**: habilidade de escuta
* **empathy**: empatia
* **creativity**: criatividade
* **organization**: organização
* **leadership**: liderança
* **risk taking**: propensão ao risco
* **public speaking comfort**: confortável ao falar em público
* **curiosity**: curiosidade
* **routine preference**: preferência por rotina
* **excitement seeking**: busca por excitação
* **friendliness**: amigável
* **emotional stability**: estabilidade emocional
* **planning**: planejamento
* **spontaneity**: espontaneidade
* **adventurousness**: espírito aventureiro
* **reading habit**: hábito de leitura
* **sports interest**: interesse por esportes
* **online social usage**: uso de redes sociais
* **travel desire**: desejo de viajar
* **gadget usage**: uso de dispositivos eletrônicos
* **work style collaborative**: estilo colaborativo de trabalho
* **decision speed**: velocidade na tomada de decisões
* **stress handling**: habilidade de lidar com o estresse

Exceto por **personality type** que é uma variável categórica, todas as colunas tem valores númericos variando entre 0 e 10 baseado no resultado de testes psicológicos.

# **2) Importação das Bibliotecas Necessárias, Carga de Dados e Ambiente**

As bibliotecas necessárias foram todas importadas no começo do notebook https://github.com/Frederico-Gribaudi/MVP_Frederico_Gribaudi_Sprint_Machine_Learning/blob/main/sprint_ML.ipynb.

Para facilitar o entendimento, todas as colunas e propriedades foram traduzidas para o português. Nesse primeiro momento verificou-se que os dados foram carregados com êxito em um dataframe.

Foram criadas duas funções python para esse MVP:
1- evaluate_classification: realiza todas as contas necessárias para análise de estatíticas de desempenho do modelo de machine learning.
2- selecionar_melhor_modelo: compara os resultados dos modelos baseado na propriedade escolhida.

# **3) Análise Exploratória**

O gráfico de barras mostra que cada personalidade apresenta valores similares, calculando podemos ver que:

*   Extrovertidos = 6857/20000 = **34,285%**
*   Ambivertidos = 6573/20000 = **32,865%**
*   Introvertidos = 6570/20000 = **32,850%**

Confirmando que o dataset é balanceado em termos de classes.
<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/c6b93ffd-507d-49ce-99df-be1533cd8871" />

A análise dos histogramas e dos boxplot mostram que existem propriedas praticamente sem separação entre as classes, portanto, possivelmente poderiam ser excluídas para melhora do desempenho dos modelos. O mapa de calor da 'Matriz de Correlação -  Fortes a Muito Fortes' revela as maiores correlações entre as características:

* positiva entre *gostar de festejar* e *confortável ao falar em público*, ou seja ambas tendem a ter valores parecidos, indicando que pertencem a mesma personalidade.

* negativa entre *preferência por solidão* ou *confortável ao falar em público* e *gostar de festejar*, logo quando uma tem valor alto, podemos excluir a personalidade da outra, por exemplo, valores altos de *gostar de festejar* implicam em valores baixos de *preferência por solidão*, nesse caso levando a uma personalidade extrovertida.

  # **4) Tratamento de dados**

## 4.1 Nulos

Não haviam valores nulos no dataset, portanto não foi necessária nenhuma  necessidade de tratamento.

## 4.2 Outliers

A análise dos boxplot revelou que existem outliers em todos os atributos, portanto não realizou-se nenhum tratamento para outliers. Conclui-se que a classificação de personalidade depende de um conjunto de características-chave, então a remoção de qualquer entrada seria prejudicial aos dados.

## 4.3 Definição do target, variáveis e divisão dos dados

Inicialmente o dataset inteiro foi dividido entre treino e teste, a coluna "tipo de personalidade" é o target, pois é nela que realmente está a classificação do indivíduo.

Target: tipo_de_personalidade
N features: 29
Treino: (16000, 29) | Teste: (4000, 29)

# **5) Treinamento dos modelos**

## 5.1 Criação do baseline e modelos candidatos

O modelo a ser usado como **baseline simples** é o DummyClassifier e na sequência evoluindo para modelos mais fortes. Os resultados desse primeiro processamento foram: 
Baseline: 0.342875 (0.000306)
KNN: 0.996312 (0.001324)
LR: 0.996562 (0.001459)
CART: 0.941687 (0.006626)
NB: 0.997250 (0.001192)
SVC: 0.997500 (0.001083)

<img width="1214" height="913" alt="image" src="https://github.com/user-attachments/assets/b46dd31f-a849-4ce1-92d7-98177dddc26e" />

Podemos ver acima que o **baseline** teve uma acurácia muito inferior em relação aos outros métodos.
Todos os modelos tiveram uma acurácia muito alta, então para evitar que houvesse **overfitting** decidiu-se por fazer uma análise de PCA (Principal Component Analysis).

Durante a análise de PCA foram usados três métodos: SelectKBest, RFE, ExtraTrees, houve uma convergência deles para seis atributos como podemos ver no gráfico abaixo:

<img width="1056" height="548" alt="image" src="https://github.com/user-attachments/assets/0ed3f0ae-8a36-4ed9-8d89-9e89f83909c5" />

## 5.2 Separando novamente os dados

Nessa etapa definimos novamente o target e as colunas necessárias para separação entre treino e teste:

Target: tipo_de_personalidade
N features: 6
Treino: (16000, 6) | Teste: (4000, 6)

## 5.3 Treino e avaliação rápida

Os novos resultados dos modelos foram:

KNN: 0.972563 (0.003997)
LR: 0.975313 (0.002975)
CART: 0.942438 (0.005806)
NB: 0.975625 (0.002546)
SVC: 0.974938 (0.003671)

<img width="1223" height="913" alt="image" src="https://github.com/user-attachments/assets/fd1cc82c-d978-4dda-841f-4e0f829753bb" />

A diferença entre as acurácias antes e depois de remover as colunas comprovam que havia de fato um **overfitting**, e agora os resultados aparentam ser mais realísticos.

## 5.4 Otimização de hiperparâmetros

### 5.4.1 Tunning do LR

Não foi necessário fazer nenhuma alteração nos valores inicialmente escolhidos, pois o modelo precisou de 87 iterações para convergir para uma solução.

### 5.4.2 Tunning do SVC

Podemos ver nessa etapa que é possível otimizar os resultados usando os seguintes hiperparâmetros: 'C': 100, 'gamma': 'scale', 'kernel': 'linear'

### 5.4.3 Tunning do KNN

Podemos ver nessa etapa que é possível otimizar os resultados usando os seguintes hiperparâmetros: 'KNN__metric': 'euclidean', 'KNN__n_neighbors': 96

### 5.4.5 Definição dos modelos e criação dos pipelines

Foram realizados testes com outros métodos e também realizando padronização e normalização para tentar alcançar uma acurácia maior. Os resultados podem ser encontrados abaixo:

| Modelo       | Accuracy | F1 Weighted | ROC AUC | Tempo Treino (s) |
|--------------|----------|-------------|---------|------------------|
| LR-orig      | 0.976    | 0.976       | NaN     | 0.336            |
| KNN-orig     | 0.976    | 0.976       | NaN     | 0.054            |
| SVM-orig     | 0.976    | 0.976       | NaN     | 33.328           |
| LR-padr      | 0.976    | 0.976       | NaN     | 0.082            |
| KNN-padr     | 0.976    | 0.976       | NaN     | 0.060            |
| SVM-padr     | 0.976    | 0.976       | NaN     | 9.877            |
| Voting-padr  | 0.976    | 0.976       | NaN     | 9.914            |
| KNN-norm     | 0.976    | 0.976       | NaN     | 0.059            |
| LR-norm      | 0.976    | 0.976       | NaN     | 0.137            |
| SVM-norm     | 0.976    | 0.976       | NaN     | 2.635            |
| Voting-norm  | 0.976    | 0.976       | NaN     | 2.842            |
| RF-norm      | 0.975    | 0.975       | NaN     | 7.989            |
| NB-norm      | 0.975    | 0.975       | NaN     | 0.023            |
| NB-orig      | 0.975    | 0.975       | NaN     | 0.020            |
| Voting-orig  | 0.975    | 0.975       | NaN     | 35.241           |
| NB-padr      | 0.975    | 0.975       | NaN     | 0.023            |
| RF-orig      | 0.974    | 0.974       | NaN     | 7.232            |
| GB-norm      | 0.974    | 0.974       | NaN     | 20.527           |
| GB-padr      | 0.974    | 0.974       | NaN     | 20.431           |
| GB-orig      | 0.974    | 0.974       | NaN     | 20.003           |
| RF-padr      | 0.974    | 0.974       | NaN     | 8.014            |
| ET-orig      | 0.973    | 0.973       | NaN     | 1.022            |
| ET-padr      | 0.972    | 0.972       | NaN     | 1.020            |
| ET-norm      | 0.972    | 0.972       | NaN     | 1.050            |
| Bag-padr     | 0.971    | 0.971       | NaN     | 15.243           |
| Bag-norm     | 0.970    | 0.970       | NaN     | 15.233           |
| Bag-orig     | 0.969    | 0.969       | NaN     | 15.957           |
| Ada-orig     | 0.965    | 0.965       | NaN     | 4.994            |
| Ada-padr     | 0.965    | 0.965       | NaN     | 3.988            |
| Ada-norm     | 0.965    | 0.965       | NaN     | 4.024            |
| CART-norm    | 0.945    | 0.945       | NaN     | 0.249            |
| CART-orig    | 0.944    | 0.944       | NaN     | 0.238            |
| CART-padr    | 0.943    | 0.943       | NaN     | 0.258            |


<img width="1998" height="599" alt="image" src="https://github.com/user-attachments/assets/309ecb15-c6f9-4979-bddc-59d8c99c1be7" />


Diversos modelos chegaram aos mesmos resultados, **LR-orig**,**KNN-orig**, **SVM-orig**, **LR-padr**,**KNN-padr**, **SVM-padr**,**Voting-padr**,**KNN-norm**,**LR-norm**,**SVM-norm**,**Voting-norm** tiveram a maior acurácia **97,6%**.
Podemos observar que os modelos de forma geral tiveram desempenhos muito parecidos, exceto pela **árvore de decisão (CART)**, que independente de como tratados os dados ficaram abaixo dos demais alcançando acurácia máxima de **94,5%**.

## 5.5 Salvando os pipelines treinados 

O tempo de processamento dos modelos acima foi em média de 30 minutos, portanto optou-se por salvar os arquivos em outra pasta: 

## 5.6 Validação 

Independente da propriedade usada para a comparação das matrizes de confusão, seja acurácia ou f1-score, o modelo **SVM-norm** apresentou o melhor resultado para cada atributo, ou seja classifica introvertido, extrovertido e ambivertido sem viéses. 

Melhor modelo por Accuracy: SVM-norm (0.976)
| Classe       | Precision | Recall | F1-Score | Suporte |
|--------------|-----------|--------|----------|---------|
| Ambivertido  | 0.96      | 0.97   | 0.96     | 1315    |
| Extrovertido | 0.99      | 0.98   | 0.98     | 1371    |
| Introvertido | 0.98      | 0.98   | 0.98     | 1314    |

| Métrica      | Precision | Recall | F1-Score | Suporte |
|--------------|-----------|--------|----------|---------|
| Accuracy     | —         | —      | 0.98     | 4000    |
| Macro Avg    | 0.98      | 0.98   | 0.98     | 4000    |
| Weighted Avg | 0.98      | 0.98   | 0.98     | 4000    |


<img width="592" height="455" alt="image" src="https://github.com/user-attachments/assets/2e74d3bd-23b7-4c8b-9360-088ec7f28efd" />

Melhor modelo por F1-weighted: SVM-norm (0.976)
| Classe       | Precision | Recall | F1-Score | Suporte |
|--------------|-----------|--------|----------|---------|
| Ambivertido  | 0.96      | 0.97   | 0.96     | 1315    |
| Extrovertido | 0.99      | 0.98   | 0.98     | 1371    |
| Introvertido | 0.98      | 0.98   | 0.98     | 1314    |

| Métrica       | Precision | Recall | F1-Score | Suporte |
|---------------|-----------|--------|----------|---------|
| Accuracy      | —         | —      | 0.98     | 4000    |
| Macro Avg     | 0.98      | 0.98   | 0.98     | 4000    |
| Weighted Avg  | 0.98      | 0.98   | 0.98     | 4000    |


<img width="592" height="455" alt="image" src="https://github.com/user-attachments/assets/4fb61d71-9c18-4ac7-9d73-3d03b6ec2a78" />

# **6) Finalização do Modelo** 

A acurácia estimada do modelo, considerando o conjunto de teste, se mantêm em **97,6%**. 

# **7) Conclusões e próximos passos** 

O modelo que apresentou melhor resultado foi o da **Support Vector Machine (SVM)**, que teve uma acurácia aproximadamente três maior que o **baseline**. O dataset usado neste trabalho é sintético, apresentando uma qualidade de dados de entrada bons e talvez seja artificial até demais, facilitando o aprendizado de quase todos os modelos. Uma melhoria futura seria usar esse modelo com a entrada de dados novos, que sejam mais realísticos.
