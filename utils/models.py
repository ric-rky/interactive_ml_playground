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
        "Simples e intuitivo, mas sensível ao ruído e à escala dos dados."
    ),
    "SVM — RBF": (
        "Encontra o hiperplano de **margem máxima** usando kernel radial (RBF). "
        "Poderoso para fronteiras não-lineares. `C` controla a regularização e "
        "`gamma` a largura do kernel."
    ),
    "SVM — Linear": (
        "SVM com kernel linear — ótimo quando as classes são **linearmente separáveis**. "
        "Rápido e eficiente em alta dimensão."
    ),
    "Decision Tree": (
        "Divide o espaço recursivamente em **regiões retangulares**. "
        "Muito interpretável, mas propenso a overfitting sem limitar a profundidade."
    ),
    "Random Forest": (
        "**Ensemble** de múltiplas árvores treinadas em subamostras aleatórias. "
        "Robusto, preciso e resistente a overfitting."
    ),
    "Gradient Boosting": (
        "Constrói árvores **sequencialmente**, cada uma corrigindo os erros da anterior. "
        "Alta acurácia, mas mais lento de treinar."
    ),
    "Neural Network (MLP)": (
        "Rede neural densa **totalmente conectada**. Aprende fronteiras de decisão "
        "arbitrariamente complexas com camadas ocultas configuráveis.\n\n"
        "**Funções de ativação:**\n"
        "- `relu` — Rectified Linear Unit. Rápida e padrão para redes profundas.\n"
        "- `tanh` — Tangente hiperbólica. Centrada em zero, boa para dados normalizados.\n"
        "- `logistic` — Sigmoide. Saturação nas extremidades; pode sofrer vanishing gradient.\n\n"
        "**Solvers:**\n"
        "- `adam` — Adaptive Moment Estimation. Robusto e geralmente a melhor escolha.\n"
        "- `sgd` — Stochastic Gradient Descent. Mais controlável com learning rate manual."
    ),
    "Neural Network (PyTorch)": (
        "Rede neural implementada em **PyTorch** com loop de treino customizado. "
        "Oferece recursos não disponíveis no MLP do scikit-learn:\n\n"
        "- **Dropout** — zera aleatoriamente neurônios durante o treino, reduzindo overfitting.\n"
        "- **Batch Normalization** — normaliza as ativações entre camadas, estabilizando e "
        "acelerando o treino.\n"
        "- **Curva de loss de validação** — 15% dos dados de treino são separados para "
        "monitorar overfitting epoch a epoch.\n"
        "- **Distribuição dos pesos** — visualização do histograma de pesos por camada "
        "após o treino.\n\n"
        "O otimizador utilizado é sempre Adam. Todos os outros parâmetros são configuráveis."
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
