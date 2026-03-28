from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


ALGORITHMS = [
    "K-Nearest Neighbors",
    "SVM — RBF",
    "SVM — Linear",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Neural Network (MLP)",
    "Neural Network (PyTorch)",
]

ALGO_DESC = {
    "K-Nearest Neighbors": (
        "Classifica pelo voto dos **k vizinhos mais próximos**. "
        "Simples e intuitivo, mas sensível ao ruído e à escala dos dados.\n\n"
        "**Regra de decisão:**\n\n"
        r"$$\hat{y} = \arg\max_{c} \sum_{i \in N_k(x)} \mathbf{1}[y_i = c]$$"
        "\n\nonde $N_k(x)$ são os $k$ pontos mais próximos de $x$ pela distância Euclidiana. "
        "Não há fase de treino — o dataset inteiro é o modelo."
    ),
    "SVM — RBF": (
        "Encontra o hiperplano de **margem máxima** num espaço de alta dimensão induzido pelo kernel RBF:\n\n"
        r"$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$"
        "\n\n`C` controla o trade-off entre margem e erros de treino. "
        "`gamma` controla o alcance de cada vetor de suporte: "
        "valor alto → fronteira mais fechada; valor baixo → fronteira mais suave."
    ),
    "SVM — Linear": (
        "SVM com kernel linear — ótimo quando as classes são **linearmente separáveis**. "
        "Encontra o hiperplano que maximiza a margem entre as classes:\n\n"
        r"$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{sujeito a} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i$$"
        "\n\n`C` penaliza as violações de margem ($\\xi_i$). "
        "Rápido e eficiente em espaços de alta dimensão."
    ),
    "Decision Tree": (
        "Divide o espaço recursivamente escolhendo o atributo que maximiza o **ganho de informação**:\n\n"
        r"$$\text{Gain}(S, f) = H(S) - \sum_{v} \frac{|S_v|}{|S|}\, H(S_v)$$"
        "\n\nonde a entropia é:\n\n"
        r"$$H(S) = -\sum_{c} p_c \log_2 p_c$$"
        "\n\nMuito interpretável, mas propenso a overfitting sem limitar a profundidade."
    ),
    "Random Forest": (
        "**Ensemble** de $T$ árvores de decisão, cada uma treinada num bootstrap do dataset "
        "com subconjunto aleatório de features. A predição é por voto majoritário:\n\n"
        r"$$\hat{y} = \text{majority}\{h_1(x),\, h_2(x),\, \ldots,\, h_T(x)\}$$"
        "\n\nA média reduz a variância sem aumentar o viés — robusto e resistente a overfitting."
    ),
    "Gradient Boosting": (
        "Constrói um modelo aditivo de forma **sequencial**, ajustando cada nova árvore "
        "ao gradiente negativo da loss da ensemble atual:\n\n"
        r"$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$"
        "\n\nonde $\\eta$ é a taxa de aprendizado e $h_m$ minimiza a loss no passo $m$. "
        "Alta acurácia, mas sensível a overfitting com $\\eta$ elevado."
    ),
    "Neural Network (MLP)": (
        "Rede neural densa **totalmente conectada**. Cada camada aplica uma transformação linear "
        "seguida de não-linearidade:\n\n"
        r"$$a^{(l)} = \sigma\!\left(W^{(l)}\, a^{(l-1)} + b^{(l)}\right)$$"
        "\n\nTreinada por backpropagation. A camada de saída usa softmax para classificação multiclasse.\n\n"
        "**Funções de ativação:**\n"
        "- `relu` — $\\sigma(z) = \\max(0, z)$. Evita vanishing gradient, padrão em redes profundas.\n"
        "- `tanh` — $\\sigma(z) = \\tanh(z)$. Centrada em zero, boa para dados normalizados.\n"
        "- `logistic` — $\\sigma(z) = 1/(1+e^{-z})$. Satura nas extremidades; evitar em redes profundas.\n\n"
        "**Solvers:**\n"
        "- `adam` — Adaptive Moment Estimation. Robusto e geralmente a melhor escolha.\n"
        "- `sgd` — Stochastic Gradient Descent. Mais controlável com learning rate manual."
    ),
    "Neural Network (PyTorch)": (
        "Mesma arquitetura MLP, implementada em **PyTorch** com loop de treino customizado "
        "e regularizações adicionais:\n\n"
        "**Dropout** — zera ativações com probabilidade $p$ durante o treino, "
        "impedindo co-adaptação dos neurônios.\n\n"
        "**Batch Normalization** — normaliza as entradas de cada camada para média zero e variância unitária:\n\n"
        r"$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad y = \gamma\hat{x} + \beta$$"
        "\n\nEstabiliza e acelera o treino, especialmente em redes mais profundas.\n\n"
        "- **Curva de loss de validação** — 15% dos dados separados para monitorar overfitting epoch a epoch.\n"
        "- **Distribuição dos pesos** — histograma por camada após o treino.\n\n"
        "Otimizador: **Adam**. Todos os outros parâmetros são configuráveis."
    ),
}

COMPARE_CLFS = {
    "KNN (k=5)":    KNeighborsClassifier(n_neighbors=5),
    "SVM RBF":      SVC(probability=True, random_state=42),
    "SVM Linear":   SVC(kernel="linear", probability=True, random_state=42),
    "Dec. Tree":    DecisionTreeClassifier(max_depth=5, random_state=42),
    "Rand. Forest": RandomForestClassifier(random_state=42),
    "Grad. Boost":  GradientBoostingClassifier(random_state=42),
    "MLP":          MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}


def build_clf(name, p):
    """Instantiate a classifier from its name and hyperparameter dict."""
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=p["k"])
    if name == "SVM — RBF":
        return SVC(C=p["C"], gamma=p["gamma"], probability=True, random_state=42)
    if name == "SVM — Linear":
        return SVC(kernel="linear", C=p["C"], probability=True, random_state=42)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["depth"], random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=p["trees"], max_depth=p["depth"], random_state=42
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=p["trees"], learning_rate=p["lr"], random_state=42
        )
    if name == "Neural Network (MLP)":
        layers = tuple([p["neurons"]] * p["layers"])
        return MLPClassifier(
            hidden_layer_sizes=layers,
            activation=p.get("activation", "relu"),
            solver=p.get("solver", "adam"),
            learning_rate_init=p["lr"],
            max_iter=800,
            random_state=42,
        )
    if name == "Neural Network (PyTorch)":
        from utils.torch_model import PyTorchMLP
        layers = tuple([p["neurons"]] * p["layers"])
        return PyTorchMLP(
            hidden_layer_sizes=layers,
            activation=p.get("activation", "relu"),
            lr=p["lr"],
            epochs=p.get("epochs", 100),
            batch_size=p.get("batch_size", 32),
            dropout=p.get("dropout", 0.0),
            batch_norm=p.get("batch_norm", False),
        )
    raise ValueError(f"Unknown algorithm: {name}")
