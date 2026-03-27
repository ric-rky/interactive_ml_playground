import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.datasets import DATASETS, DATASET_DESC
from utils.models import ALGORITHMS, ALGO_DESC, COMPARE_CLFS, build_clf
from utils.plots import (
    plot_data_only, plot_boundary, plot_confusion,
    plot_feature_importance, plot_compare,
    plot_loss_curve, plot_network_arch, plot_activations,
    plot_roc, plot_probability_boundary, plot_learning_curves,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d1a 0%, #111827 60%, #0f172a 100%);
}
[data-testid="stSidebar"] {
    background: rgba(10, 10, 25, 0.94);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.main-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 0.5rem 0 0.1rem;
}
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Dataset")
    ds_name   = st.selectbox("Tipo", list(DATASETS.keys()))
    n_samples = st.slider("Amostras", 100, 1000, 350, 50)
    noise     = st.slider("Ruído", 0.0, 0.5, 0.15, 0.05)
    test_pct  = st.slider("% Teste", 10, 45, 25, 5)
    seed      = st.number_input("Seed", 0, 9999, 42)

    st.divider()
    st.markdown("## Algoritmo")
    algo = st.selectbox("Modelo", ALGORITHMS)

    st.markdown("#### Hiperparâmetros")
    p = {}
    if algo == "K-Nearest Neighbors":
        p["k"] = st.slider("Vizinhos (k)", 1, 30, 5)

    elif algo in ("SVM — RBF", "SVM — Linear"):
        p["C"] = st.select_slider("C (regularização)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
        p["gamma"] = (
            st.select_slider("Gamma", ["scale", "auto", 0.01, 0.1, 1.0], "scale")
            if algo == "SVM — RBF" else "scale"
        )

    elif algo == "Decision Tree":
        p["depth"] = st.slider("Profundidade máx.", 1, 20, 5)

    elif algo == "Random Forest":
        p["trees"] = st.slider("Árvores", 10, 300, 100, 10)
        p["depth"] = st.slider("Profundidade máx.", 1, 20, 5)

    elif algo == "Gradient Boosting":
        p["trees"] = st.slider("Estimadores", 10, 300, 100, 10)
        p["lr"]    = st.select_slider("Taxa de aprendizado", [0.01, 0.05, 0.1, 0.2, 0.5], 0.1)

    elif algo == "Neural Network (MLP)":
        p["layers"]     = st.slider("Camadas ocultas", 1, 6, 2)
        p["neurons"]    = st.slider("Neurônios / camada", 4, 256, 64, 4)
        p["lr"]         = st.select_slider("Taxa de aprendizado", [0.0001, 0.001, 0.01, 0.1], 0.001)
        p["activation"] = st.selectbox("Ativação", ["relu", "tanh", "logistic"])
        p["solver"]     = st.selectbox("Solver", ["adam", "sgd"])

    st.divider()
    btn_train = st.button("Treinar modelo", type="primary", use_container_width=True)
    btn_cmp   = st.button("Comparar todos os modelos", use_container_width=True)

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ds, n, noise, ts, seed):
    X, y = DATASETS[ds](n, noise, seed)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts / 100, random_state=seed)
    return X, y, Xtr, Xte, ytr, yte

X, y, Xtr, Xte, ytr, yte = load_data(ds_name, n_samples, noise, test_pct, seed)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🤖 ML Playground</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ajuste o dataset e os hiperparâmetros — '
    'veja a fronteira de decisão em tempo real</p>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", len(X))
c2.metric("Treino", len(Xtr))
c3.metric("Teste", len(Xte))
c4.metric("Classes", len(np.unique(y)))

# ── Train ─────────────────────────────────────────────────────────────────────
if btn_train:
    clf = build_clf(algo, p)
    clf.fit(Xtr, ytr)
    st.session_state["clf"]  = clf
    st.session_state["algo"] = algo
    st.session_state["p"]    = p.copy()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_vis, tab_met, tab_nn, tab_cmp, tab_info = st.tabs([
    "Fronteira de Decisão",
    "Métricas",
    "Rede Neural",
    "Comparação de Modelos",
    "Sobre os Algoritmos",
])

# ── Tab 1 — Decision boundary ──────────────────────────────────────────────────
with tab_vis:
    if "clf" in st.session_state:
        clf = st.session_state["clf"]
        train_acc = accuracy_score(ytr, clf.predict(Xtr))
        test_acc  = accuracy_score(yte, clf.predict(Xte))

        m1, m2, m3 = st.columns(3)
        m1.metric("Acurácia — Treino", f"{train_acc:.2%}")
        m2.metric("Acurácia — Teste",  f"{test_acc:.2%}",
                  delta=f"{test_acc - train_acc:+.2%}")
        gap = train_acc - test_acc
        m3.metric("Gap (overfitting)", f"{gap:.2%}",
                  delta="Alto" if gap > 0.1 else "OK", delta_color="inverse")

        show_confidence = st.toggle(
            "Mostrar mapa de confiança",
            value=False,
            help="Exibe a probabilidade/confiança do modelo em vez da fronteira de decisão",
        )

        if show_confidence:
            st.plotly_chart(
                plot_probability_boundary(clf, X, y),
                use_container_width=True,
            )
            st.caption(
                "**Verde** = modelo muito confiante  |  "
                "**Vermelho** = modelo incerto (próximo da fronteira)"
            )
        else:
            st.plotly_chart(
                plot_boundary(clf, X, y, Xtr, Xte, ytr, yte),
                use_container_width=True,
            )
            st.caption(
                "**Legenda:** ● círculos = treino  |  ◆ diamantes = teste correto  |  "
                "✗ amarelo = predição errada  |  fundo colorido = região de decisão"
            )
    else:
        st.plotly_chart(plot_data_only(X, y, f"Dataset: {ds_name}"),
                        use_container_width=True)
        st.info(f"**{ds_name}** — {DATASET_DESC[ds_name]}  \n"
                "Escolha um algoritmo e clique em **Treinar modelo**.")

# ── Tab 2 — Metrics ────────────────────────────────────────────────────────────
with tab_met:
    if "clf" not in st.session_state:
        st.info("Treine um modelo primeiro.")
    else:
        clf    = st.session_state["clf"]
        y_pred = clf.predict(Xte)
        classes = np.unique(y)

        # ── Confusion matrix + per-class table ──
        col_l, col_r = st.columns([1.1, 1])

        with col_l:
            st.plotly_chart(plot_confusion(yte, y_pred, classes),
                            use_container_width=True)

        with col_r:
            rows = []
            for c in classes:
                bt = (yte == c).astype(int)
                bp = (y_pred == c).astype(int)
                rows.append({
                    "Classe":   int(c),
                    "Precisão": f"{precision_score(bt, bp, zero_division=0):.2%}",
                    "Recall":   f"{recall_score(bt, bp, zero_division=0):.2%}",
                    "F1":       f"{f1_score(bt, bp, zero_division=0):.2%}",
                    "Suporte":  int(bt.sum()),
                })
            st.markdown("#### Métricas por Classe (Teste)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            acc   = accuracy_score(yte, y_pred)
            color = "#22c55e" if acc >= 0.9 else "#f59e0b" if acc >= 0.75 else "#ef4444"
            st.markdown(f"""
            <div style="text-align:center;padding:1.2rem;background:rgba(15,15,30,0.6);
                        border-radius:12px;border:1px solid rgba(255,255,255,0.08);margin-top:.8rem;">
                <div style="color:#64748b;font-size:.8rem;letter-spacing:.06em;">ACURÁCIA GERAL</div>
                <div style="font-size:3rem;font-weight:900;color:{color};line-height:1.1;">{acc:.1%}</div>
                <div style="color:#475569;font-size:.8rem;">{st.session_state['algo']}</div>
            </div>""", unsafe_allow_html=True)

        if hasattr(clf, "feature_importances_"):
            st.plotly_chart(plot_feature_importance(clf.feature_importances_),
                            use_container_width=True)

        # ── ROC Curve ──
        st.divider()
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Xte)
            st.plotly_chart(plot_roc(yte, y_prob, classes), use_container_width=True)
            st.caption(
                "Curva ROC — quanto mais próxima do canto superior esquerdo, melhor. "
                "AUC = 1.0 é perfeito; AUC = 0.5 equivale a chute aleatório."
            )
        else:
            st.info("Curva ROC não disponível para este modelo (sem `predict_proba`).")

        # ── Learning Curves ──
        st.divider()
        st.markdown("#### Curvas de Aprendizado")
        st.caption(
            "Mostra como a acurácia evolui com mais dados de treino — "
            "útil para diagnosticar bias (underfitting) vs. variância (overfitting)."
        )
        if st.button("Gerar curvas de aprendizado", key="btn_lc"):
            saved_algo = st.session_state.get("algo", algo)
            saved_p    = st.session_state.get("p", p)
            clf_lc = clone(build_clf(saved_algo, saved_p))

            X_all = np.vstack([Xtr, Xte])
            y_all = np.concatenate([ytr, yte])

            with st.spinner("Calculando (5-fold CV × 10 tamanhos)…"):
                train_sizes_abs, tr_sc, va_sc = learning_curve(
                    clf_lc, X_all, y_all,
                    cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring="accuracy",
                    n_jobs=-1,
                )
            st.plotly_chart(
                plot_learning_curves(train_sizes_abs, tr_sc, va_sc),
                use_container_width=True,
            )

# ── Tab 3 — Neural Network ─────────────────────────────────────────────────────
with tab_nn:
    # Architecture diagram — live preview from sidebar params
    is_mlp_selected = (algo == "Neural Network (MLP)")
    is_mlp_trained  = (
        "clf" in st.session_state
        and st.session_state.get("algo") == "Neural Network (MLP)"
    )

    col_arch, col_act = st.columns([1.4, 1])

    with col_arch:
        st.markdown("#### Arquitetura da Rede")
        if is_mlp_selected:
            n_classes  = len(np.unique(y))
            hidden     = [p["neurons"]] * p["layers"]
            layer_sizes = [2] + hidden + [n_classes]
            st.plotly_chart(plot_network_arch(layer_sizes), use_container_width=True)
            st.caption(
                f"🔵 Entrada (2)  →  🟣 {p['layers']}× oculta ({p['neurons']})  "
                f"→  🟢 Saída ({n_classes})  |  "
                f"Ativação: `{p.get('activation','relu')}`  |  Solver: `{p.get('solver','adam')}`"
            )
        elif is_mlp_trained:
            clf_nn = st.session_state["clf"]
            saved_p = st.session_state.get("p", {})
            n_classes  = len(np.unique(y))
            hidden     = [saved_p.get("neurons", 64)] * saved_p.get("layers", 2)
            layer_sizes = [2] + hidden + [n_classes]
            st.plotly_chart(plot_network_arch(layer_sizes), use_container_width=True)
        else:
            st.info(
                "Selecione **Neural Network (MLP)** na barra lateral para visualizar "
                "a arquitetura em tempo real conforme você ajusta os hiperparâmetros."
            )

    with col_act:
        st.markdown("#### Funções de Ativação")
        st.plotly_chart(plot_activations(), use_container_width=True)
        st.caption(
            "**ReLU** — padrão moderno, evita vanishing gradient  \n"
            "**Tanh** — centrada em zero, boa para dados normalizados  \n"
            "**Sigmoid** — satura nas extremidades, evitar em redes profundas  \n"
            "**Leaky ReLU** — corrige o 'neurônio morto' do ReLU"
        )

    # Loss curve — only when MLP was trained
    if is_mlp_trained:
        st.divider()
        clf_nn = st.session_state["clf"]

        # Convergence info
        ci1, ci2, ci3 = st.columns(3)
        converged = clf_nn.n_iter_ < clf_nn.max_iter
        ci1.metric("Iterações", clf_nn.n_iter_)
        ci2.metric("Loss final", f"{clf_nn.loss_curve_[-1]:.5f}")
        ci3.metric("Convergiu?", "✅ Sim" if converged else "⚠️ Não (aumentar max_iter)")

        st.markdown("#### Curva de Loss")
        st.plotly_chart(plot_loss_curve(clf_nn.loss_curve_), use_container_width=True)
        if not converged:
            st.warning(
                "O modelo atingiu o limite de iterações sem convergir. "
                "Tente aumentar a taxa de aprendizado ou reduzir a complexidade da rede."
            )
    elif not is_mlp_selected and "clf" in st.session_state:
        st.divider()
        st.info(
            "A curva de loss e detalhes de convergência só estão disponíveis "
            "para o **Neural Network (MLP)**."
        )

# ── Tab 4 — Compare ────────────────────────────────────────────────────────────
with tab_cmp:
    if btn_cmp:
        rows = []
        with st.spinner("Treinando 7 modelos…"):
            for name, clf_c in COMPARE_CLFS.items():
                clf_c.fit(Xtr, ytr)
                rows.append({
                    "Algoritmo": name,
                    "Treino":    accuracy_score(ytr, clf_c.predict(Xtr)),
                    "Teste":     accuracy_score(yte, clf_c.predict(Xte)),
                })
        df_cmp = pd.DataFrame(rows)
        st.session_state["cmp_df"] = df_cmp

    if "cmp_df" in st.session_state:
        df_cmp = st.session_state["cmp_df"]
        st.plotly_chart(plot_compare(df_cmp), use_container_width=True)

        df_show = df_cmp.sort_values("Teste", ascending=False).copy()
        df_show["Treino"] = df_show["Treino"].map("{:.2%}".format)
        df_show["Teste"]  = df_show["Teste"].map("{:.2%}".format)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.caption("Hiperparâmetros padrão — use a barra lateral para ajustar individualmente.")
    else:
        st.info("Clique em **Comparar todos os modelos** na barra lateral.")

# ── Tab 5 — Info ───────────────────────────────────────────────────────────────
with tab_info:
    st.markdown("### Algoritmos disponíveis")
    for name, desc in ALGO_DESC.items():
        with st.expander(f"**{name}**"):
            st.markdown(desc)

    st.markdown("### Datasets disponíveis")
    for name, desc in DATASET_DESC.items():
        with st.expander(f"**{name}**"):
            st.write(desc)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#1e293b;font-size:.75rem;">'
    "ML Playground • Streamlit + Scikit-learn + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
