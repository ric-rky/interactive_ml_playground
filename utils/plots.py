import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

PALETTE = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899"]


def _base_layout(fig, title="", h=500):
    fig.update_layout(
        template="plotly_dark",
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,30,0.55)",
        margin=dict(l=10, r=10, t=44, b=10),
        title=dict(text=title, font=dict(size=13, color="#94a3b8"), x=0.5),
        legend=dict(
            bgcolor="rgba(0,0,0,0.45)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )


# ── Existing plots ─────────────────────────────────────────────────────────────

def plot_data_only(X, y, title="Dataset"):
    fig = go.Figure()
    for i, c in enumerate(np.unique(y)):
        m = y == c
        fig.add_trace(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers",
            name=f"Classe {c}",
            marker=dict(size=7, color=PALETTE[i % len(PALETTE)],
                        opacity=0.85, line=dict(width=1, color="white")),
        ))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    _base_layout(fig, title)
    return fig


def plot_boundary(clf, X, y, X_train, X_test, y_train, y_test):
    margin = 0.6
    x0, x1 = X[:, 0].min() - margin, X[:, 0].max() + margin
    y0, y1 = X[:, 1].min() - margin, X[:, 1].max() + margin
    h = max(x1 - x0, y1 - y0) / 160

    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    classes = np.unique(y)
    n_cls = len(classes)

    if n_cls == 2:
        cscale = [[0, "rgba(239,68,68,0.22)"], [1, "rgba(59,130,246,0.22)"]]
    else:
        cscale = [
            [0.0, "rgba(239,68,68,0.22)"],
            [0.5, "rgba(59,130,246,0.22)"],
            [1.0, "rgba(34,197,94,0.22)"],
        ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=np.arange(x0, x1, h), y=np.arange(y0, y1, h), z=Z,
        colorscale=cscale, showscale=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Contour(
        x=np.arange(x0, x1, h), y=np.arange(y0, y1, h), z=Z,
        showscale=False, contours=dict(showlines=True, coloring="none"),
        line=dict(width=2, color="rgba(255,255,255,0.45)"),
        hoverinfo="skip",
    ))

    for i, c in enumerate(classes):
        m = y_train == c
        fig.add_trace(go.Scatter(
            x=X_train[m, 0], y=X_train[m, 1], mode="markers",
            name=f"Treino {c}",
            marker=dict(size=7, color=PALETTE[i % len(PALETTE)], symbol="circle",
                        line=dict(width=1.2, color="white"), opacity=0.78),
        ))

    y_pred = clf.predict(X_test)
    for i, c in enumerate(classes):
        m = y_test == c
        correct = y_pred[m] == c
        if correct.sum():
            fig.add_trace(go.Scatter(
                x=X_test[m][correct, 0], y=X_test[m][correct, 1], mode="markers",
                name=f"Teste {c} ✓",
                marker=dict(size=9, color=PALETTE[i % len(PALETTE)], symbol="diamond",
                            line=dict(width=1.5, color="white")),
            ))
        wrong = ~correct
        if wrong.sum():
            fig.add_trace(go.Scatter(
                x=X_test[m][wrong, 0], y=X_test[m][wrong, 1], mode="markers",
                name=f"Teste {c} ✗",
                marker=dict(size=12, color="rgba(0,0,0,0)", symbol="x",
                            line=dict(width=2.5, color="#fbbf24")),
            ))

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    _base_layout(fig, "Fronteira de Decisão")
    return fig


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [
        [f"{cm[i,j]}<br>({cm_norm[i,j]:.0%})" for j in range(len(classes))]
        for i in range(len(classes))
    ]
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=[f"Pred {c}" for c in classes],
        y=[f"Real {c}" for c in classes],
        text=text, texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale="Blues", showscale=False,
    ))
    _base_layout(fig, "Matriz de Confusão", h=320)
    return fig


def plot_feature_importance(importances):
    fig = go.Figure(go.Bar(
        x=["Feature 1", "Feature 2"], y=importances,
        marker_color=["#a78bfa", "#60a5fa"],
        text=[f"{v:.2%}" for v in importances],
        textposition="outside",
    ))
    fig.update_yaxes(tickformat=".0%")
    _base_layout(fig, "Importância das Features", h=260)
    return fig


def plot_compare(results_df):
    df = results_df.sort_values("Teste", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Algoritmo"], x=df["Treino"], name="Treino",
        orientation="h", marker_color="rgba(139,92,246,0.65)",
    ))
    fig.add_trace(go.Bar(
        y=df["Algoritmo"], x=df["Teste"], name="Teste",
        orientation="h", marker_color="rgba(59,130,246,0.9)",
    ))
    fig.update_xaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_layout(barmode="group")
    _base_layout(fig, "Comparação de Algoritmos (hiperparâmetros padrão)", h=370)
    return fig


# ── Neural Network plots ───────────────────────────────────────────────────────

def plot_network_arch(layer_sizes):
    """Interactive diagram of a neural network architecture."""
    MAX_SHOW = 8
    n_layers = len(layer_sizes)
    layer_x = np.linspace(0.05, 0.95, n_layers)

    all_ys = []
    for n in layer_sizes:
        disp = min(n, MAX_SHOW)
        ys = list(np.linspace(-(disp - 1) / 2.0, (disp - 1) / 2.0, disp))
        all_ys.append(ys)

    max_spread = max(len(ys) for ys in all_ys) / 2.0 + 1.8

    fig = go.Figure()

    # Edges
    ex, ey = [], []
    for li in range(n_layers - 1):
        for y1 in all_ys[li]:
            for y2 in all_ys[li + 1]:
                ex += [layer_x[li], layer_x[li + 1], None]
                ey += [y1, y2, None]

    if ex:
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="rgba(148,163,184,0.10)", width=0.7),
            hoverinfo="skip", showlegend=False,
        ))

    layer_names = (
        ["Entrada"]
        + [f"Oculta {i + 1}" for i in range(n_layers - 2)]
        + ["Saída"]
    )
    layer_colors = ["#60a5fa"] + ["#a78bfa"] * max(0, n_layers - 2) + ["#22c55e"]

    for li, (ys, n, color, name) in enumerate(
        zip(all_ys, layer_sizes, layer_colors, layer_names)
    ):
        xs = [layer_x[li]] * len(ys)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=name,
            marker=dict(
                size=22, color=color, opacity=0.92,
                line=dict(width=2.5, color="rgba(255,255,255,0.65)"),
            ),
            hovertemplate=f"<b>{name}</b><br>{n} neurônios<extra></extra>",
        ))

        top_y = max(ys) if ys else 0
        count_label = str(n) if n <= MAX_SHOW else f"{n}  (↑ mostrando {MAX_SHOW})"
        fig.add_annotation(
            x=layer_x[li], y=top_y + 1.1,
            text=f"<b>{name}</b><br><span style='font-size:9px;color:#64748b'>{count_label}</span>",
            showarrow=False,
            font=dict(size=11, color="#94a3b8"),
            align="center",
        )

        if n > MAX_SHOW:
            fig.add_annotation(
                x=layer_x[li], y=0,
                text="⋮",
                showarrow=False,
                font=dict(size=30, color=color, family="serif"),
            )

    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(visible=False, range=[-max_spread, max_spread])
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,30,0.55)",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def plot_loss_curve(loss_curve):
    """MLP training loss history."""
    iters = list(range(1, len(loss_curve) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=loss_curve,
        mode="lines",
        line=dict(color="#a78bfa", width=2.5),
        name="Loss",
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.08)",
        hovertemplate="Iteração %{x}<br>Loss: %{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Iteração", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title="Loss", gridcolor="rgba(255,255,255,0.05)")
    _base_layout(fig, "Curva de Loss — Treinamento", h=280)
    return fig


def plot_activations():
    """Comparison of common neural network activation functions."""
    x = np.linspace(-4, 4, 300)

    funcs = {
        "ReLU":       (np.maximum(0, x),              "#a78bfa"),
        "Tanh":       (np.tanh(x),                    "#60a5fa"),
        "Sigmoid":    (1 / (1 + np.exp(-x)),          "#34d399"),
        "Leaky ReLU": (np.where(x >= 0, x, 0.1 * x), "#f59e0b"),
    }

    fig = go.Figure()
    for name, (y, color) in funcs.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
        ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.update_xaxes(title="x", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title="f(x)", gridcolor="rgba(255,255,255,0.05)", range=[-1.6, 2.8])
    _base_layout(fig, "Funções de Ativação", h=300)
    return fig


# ── Enhanced metrics plots ─────────────────────────────────────────────────────

def plot_roc(y_true, y_prob, classes):
    """ROC curve — single curve for binary, one-vs-rest for multiclass."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    fig = go.Figure()

    # Reference diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(255,255,255,0.18)", dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {roc_auc:.3f}",
            line=dict(color="#a78bfa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"Classe {c}  (AUC={roc_auc:.3f})",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))

    fig.update_xaxes(
        title="Taxa de Falso Positivo (FPR)",
        gridcolor="rgba(255,255,255,0.05)", range=[-0.02, 1.02],
    )
    fig.update_yaxes(
        title="Taxa de Verdadeiro Positivo (TPR)",
        gridcolor="rgba(255,255,255,0.05)", range=[-0.02, 1.02],
    )
    _base_layout(fig, "Curva ROC", h=340)
    return fig


def plot_probability_boundary(clf, X, y):
    """Confidence heatmap — shows model certainty across the feature space."""
    margin = 0.6
    x0, x1 = X[:, 0].min() - margin, X[:, 0].max() + margin
    y0, y1 = X[:, 1].min() - margin, X[:, 1].max() + margin
    h = max(x1 - x0, y1 - y0) / 160

    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(grid)
        confidence = proba.max(axis=1)
    else:
        dec = clf.decision_function(grid)
        if dec.ndim == 1:
            confidence = np.abs(dec)
        else:
            confidence = dec.max(axis=1) - np.sort(dec, axis=1)[:, -2]
        confidence = (confidence - confidence.min()) / (confidence.ptp() + 1e-9)

    confidence = confidence.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=np.arange(x0, x1, h), y=np.arange(y0, y1, h), z=confidence,
        colorscale=[
            [0.0, "rgba(239,68,68,0.80)"],
            [0.5, "rgba(251,191,36,0.55)"],
            [1.0, "rgba(34,197,94,0.65)"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Confiança", font=dict(size=11, color="#94a3b8")),
            tickformat=".0%",
            tickfont=dict(color="#94a3b8"),
        ),
        hoverinfo="skip",
        zmin=0, zmax=1,
    ))

    classes = np.unique(y)
    for i, c in enumerate(classes):
        m = y == c
        fig.add_trace(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers",
            name=f"Classe {c}",
            marker=dict(
                size=6, color=PALETTE[i % len(PALETTE)],
                opacity=0.80, line=dict(width=1.2, color="white"),
            ),
        ))

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    _base_layout(fig, "Mapa de Confiança do Modelo  (verde = certeza alta, vermelho = incerteza)", h=500)
    return fig


def plot_learning_curves(train_sizes, train_scores, val_scores):
    """Learning curves showing bias/variance trade-off."""
    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std  = val_scores.std(axis=1)

    fig = go.Figure()

    # Train band + line
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([tr_mean + tr_std, (tr_mean - tr_std)[::-1]]),
        fill="toself", fillcolor="rgba(167,139,250,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=tr_mean, mode="lines+markers",
        name="Treino",
        line=dict(color="#a78bfa", width=2.5),
        marker=dict(size=7),
        hovertemplate="n=%{x}<br>Treino: %{y:.2%}<extra></extra>",
    ))

    # Val band + line
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([va_mean + va_std, (va_mean - va_std)[::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=va_mean, mode="lines+markers",
        name="Validação (CV)",
        line=dict(color="#60a5fa", width=2.5),
        marker=dict(size=7),
        hovertemplate="n=%{x}<br>Val: %{y:.2%}<extra></extra>",
    ))

    fig.update_xaxes(title="Amostras de treino", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(
        title="Acurácia", tickformat=".0%",
        gridcolor="rgba(255,255,255,0.05)", range=[0, 1.05],
    )
    _base_layout(fig, "Curvas de Aprendizado  (banda = ±1 desvio padrão, 5-fold CV)", h=320)
    return fig
