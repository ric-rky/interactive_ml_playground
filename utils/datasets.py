import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs


def make_xor(n_samples=300, noise=0.15, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    X += rng.randn(*X.shape) * noise
    return X, y


def make_spiral(n_samples=300, noise=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    n = n_samples // 2

    def _arm(n, offset):
        t = np.linspace(0, 4 * np.pi, n)
        r = t / (4 * np.pi)
        x = r * np.cos(t + offset) + rng.randn(n) * noise
        y = r * np.sin(t + offset) + rng.randn(n) * noise
        return np.c_[x, y]

    X = np.vstack([_arm(n, 0), _arm(n, np.pi)])
    y = np.array([0] * n + [1] * n)
    return X, y


def make_spiral3(n_samples=300, noise=0.15, random_state=42):
    """Three interleaved spirals — challenging multiclass problem."""
    rng = np.random.RandomState(random_state)
    n = n_samples // 3
    arms, labels = [], []
    for k in range(3):
        t = np.linspace(0, 4 * np.pi, n)
        r = t / (4 * np.pi)
        offset = k * 2 * np.pi / 3
        xc = r * np.cos(t + offset) + rng.randn(n) * noise
        yc = r * np.sin(t + offset) + rng.randn(n) * noise
        arms.append(np.c_[xc, yc])
        labels.extend([k] * n)
    return np.vstack(arms), np.array(labels)


def make_checkerboard(n_samples=300, noise=0.05, random_state=42):
    """Checkerboard pattern — requires highly non-linear boundaries."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-2, 2, (n_samples, 2))
    y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)
    X += rng.randn(*X.shape) * noise * 2
    return X, y


DATASETS = {
    "Moons":        lambda n, ns, r: make_moons(n_samples=n, noise=ns, random_state=r),
    "Circles":      lambda n, ns, r: make_circles(n_samples=n, noise=ns, factor=0.4, random_state=r),
    "XOR":          lambda n, ns, r: make_xor(n_samples=n, noise=ns, random_state=r),
    "Spiral":       lambda n, ns, r: make_spiral(n_samples=n, noise=ns, random_state=r),
    "Spiral (3)":   lambda n, ns, r: make_spiral3(n_samples=n, noise=ns, random_state=r),
    "Tabuleiro":    lambda n, ns, r: make_checkerboard(n_samples=n, noise=ns, random_state=r),
    "Blobs (2)":    lambda n, ns, r: make_blobs(n_samples=n, centers=2, cluster_std=ns * 4 + 0.6, random_state=r),
    "Blobs (3)":    lambda n, ns, r: make_blobs(n_samples=n, centers=3, cluster_std=ns * 4 + 0.6, random_state=r),
}

DATASET_DESC = {
    "Moons":      "Duas classes em forma de crescente — não linearmente separável.",
    "Circles":    "Círculos concêntricos — desafia modelos lineares.",
    "XOR":        "Padrão XOR clássico — requer fronteira não-linear.",
    "Spiral":     "Espirais entrelaçadas — fronteira muito complexa.",
    "Spiral (3)": "Três espirais entrelaçadas — problema multiclasse extremamente difícil.",
    "Tabuleiro":  "Padrão de xadrez — requer fronteiras altamente não-lineares e periódicas.",
    "Blobs (2)":  "Dois grupos gaussianos — dataset mais fácil.",
    "Blobs (3)":  "Três grupos gaussianos — problema multiclasse.",
}
