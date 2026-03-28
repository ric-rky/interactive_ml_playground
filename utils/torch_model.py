import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class _MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation, dropout, batch_norm):
        super().__init__()
        act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "logistic": nn.Sigmoid}
        Act = act_map[activation]

        layers = []
        in_sz = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_sz, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_sz = h
        layers.append(nn.Linear(in_sz, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PyTorchMLP(BaseEstimator, ClassifierMixin):
    """
    PyTorch MLP compativel com a API do scikit-learn.
    Suporta dropout, batch normalization e exibe loss de treino e validacao por epoch.
    """

    def __init__(self, hidden_layer_sizes=(64, 64), activation="relu",
                 lr=0.001, epochs=100, batch_size=32,
                 dropout=0.0, batch_norm=False, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.random_state = random_state

    def fit(self, X, y, progress_callback=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y_enc)

        # 15% validation split (fixed, not shuffled to be reproducible)
        n_val = max(1, int(0.15 * len(X)))
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(self.random_state))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        tr_loader = DataLoader(
            TensorDataset(X_t[tr_idx], y_t[tr_idx]),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_t[val_idx], y_t[val_idx]),
            batch_size=len(val_idx),
        )

        self.model_ = _MLP(
            X.shape[1], self.hidden_layer_sizes, n_classes,
            self.activation, self.dropout, self.batch_norm,
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.train_losses_: list[float] = []
        self.val_losses_:   list[float] = []

        n_tr = len(tr_idx)
        for epoch in range(self.epochs):
            self.model_.train()
            running = 0.0
            for Xb, yb in tr_loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(Xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(Xb)
            tr_loss = running / n_tr
            self.train_losses_.append(tr_loss)

            self.model_.eval()
            with torch.no_grad():
                for Xb, yb in val_loader:
                    val_loss = criterion(self.model_(Xb), yb).item()
            self.val_losses_.append(val_loss)

            if progress_callback:
                progress_callback(epoch + 1, self.epochs, tr_loss, val_loss)

        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.FloatTensor(X)).argmax(dim=1).numpy()
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.FloatTensor(X))
            return torch.softmax(logits, dim=1).numpy()

    # ── sklearn-compatible aliases ──────────────────────────────────────────────

    @property
    def loss_curve_(self):
        """Alias para compatibilidade com o codigo do MLP do sklearn."""
        return self.train_losses_

    @property
    def n_iter_(self):
        return len(self.train_losses_)

    @property
    def max_iter(self):
        return self.epochs
