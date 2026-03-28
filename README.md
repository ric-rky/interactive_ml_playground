# ML Playground

Uma aplicação web interativa para explorar e comparar algoritmos de classificação. O objetivo é tornar o comportamento dos modelos visível: você ajusta os dados e os hiperparâmetros, treina, e vê o resultado imediatamente na forma de fronteiras de decisão, métricas, curvas de aprendizado e muito mais.

Construído com Streamlit, scikit-learn, PyTorch e Plotly.

## Funcionalidades

### Fronteira de Decisão

Exibe a região de decisão do classificador treinado sobre o espaço de features, com os pontos de treino e teste sobrepostos. Pontos de teste classificados incorretamente são destacados separadamente.

Há um toggle que substitui a fronteira binária por um mapa de confiança: o fundo passa a mostrar a probabilidade máxima predita em cada ponto do espaço, tornando visível onde o modelo tem certeza e onde ele hesita.

### Metricas

- Matriz de confusao normalizada por linha
- Tabela com precisao, recall, F1 e suporte por classe
- Curva ROC com AUC para classificacao binaria; one-vs-rest para multiclasse
- Curvas de aprendizado (acuracia de treino e validacao cruzada em funcao do tamanho do conjunto de treino), calculadas sob demanda com 5-fold CV

### Rede Neural

Aba dedicada aos modelos de rede neural (MLP e PyTorch):

- Diagrama interativo da arquitetura da rede, atualizado em tempo real conforme voce ajusta os sliders de camadas e neuronios, antes mesmo de treinar
- Curva de loss por iteracao/epoch apos o treino, com indicador de convergencia
- Curva de loss de validacao (apenas PyTorch) — 15% dos dados separados automaticamente para monitorar overfitting epoch a epoch
- Distribuicao dos pesos por camada apos o treino (apenas PyTorch) — histograma interativo para inspecionar a inicializacao e o comportamento do otimizador
- Grafico comparativo das funcoes de ativacao mais comuns (ReLU, Tanh, Sigmoid, Leaky ReLU)

### Comparacao de Modelos

Treina todos os 7 algoritmos com hiperparametros padrao sobre o mesmo dataset e exibe um grafico de barras comparando acuracia de treino e teste.

## Algoritmos disponíveis

| Algoritmo | Hiperparametros configuráveis |
|---|---|
| K-Nearest Neighbors | k (numero de vizinhos) |
| SVM com kernel RBF | C, gamma |
| SVM com kernel Linear | C |
| Decision Tree | profundidade maxima |
| Random Forest | numero de arvores, profundidade maxima |
| Gradient Boosting | numero de estimadores, taxa de aprendizado |
| Neural Network (MLP) | camadas ocultas, neuronios por camada, taxa de aprendizado, funcao de ativacao, solver |
| Neural Network (PyTorch) | camadas ocultas, neuronios por camada, taxa de aprendizado, funcao de ativacao, epochs, batch size, dropout, batch normalization |

## Datasets disponíveis

| Dataset | Descricao |
|---|---|
| Moons | Duas classes em forma de crescente |
| Circles | Circulos concentricos |
| XOR | Padrao XOR classico |
| Spiral | Duas espirais entrelacadas |
| Spiral (3) | Tres espirais entrelacadas, problema multiclasse dificil |
| Tabuleiro | Padrao xadrez, requer fronteiras periodicas e nao-lineares |
| Blobs (2) | Dois grupos gaussianos |
| Blobs (3) | Tres grupos gaussianos |

Os parametros de numero de amostras, nivel de ruido, proporcao treino/teste e seed sao controlaveis pela barra lateral.

## Estrutura do projeto

```
ml_playground/
├── app.py                  # Aplicacao principal (Streamlit)
├── requirements.txt
├── utils/
│   ├── datasets.py         # Geracao dos datasets
│   ├── models.py           # Instanciacao dos classificadores
│   ├── plots.py            # Todos os graficos (Plotly)
│   └── torch_model.py      # PyTorchMLP — wrapper scikit-learn sobre rede PyTorch
└── .streamlit/
    └── config.toml         # Tema e configuracao do servidor
```

## Instalação e execução

**Requisitos:** Python 3.10 ou superior.

```bash
# Clone o repositorio
git clone <url-do-repositorio>
cd ml_playground

# Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# Instale as dependencias
pip install -r requirements.txt

# Execute a aplicacao
streamlit run app.py
```

A aplicacao abre automaticamente no navegador em `http://localhost:8501`.

## Dependências

```
streamlit>=1.35.0
scikit-learn>=1.4.0
numpy>=1.26.0
plotly>=6.0.0
pandas>=2.2.0
torch>=2.0.0
```
