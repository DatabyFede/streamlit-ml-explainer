"""
streamlit-ml-explainer
======================
App interactiva para entrenar modelos de ML y visualizar explicaciones SHAP.
El usuario sube un CSV, elige la variable objetivo y compara 3 modelos.

Uso:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ── Configuración de página ──
st.set_page_config(
    page_title="ML Explainer · DatabyFede",
    page_icon="🤖",
    layout="wide"
)

# ── Estilos ──
st.markdown("""
<style>
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1E40AF; }
    .metric-label { font-size: 0.85rem; color: #64748B; margin-top: 4px; }
    .section-title { font-size: 1.2rem; font-weight: 600; margin: 1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("🤖 ML Explainer")
st.caption("Subí un dataset, elegí tu variable objetivo y compará modelos de Machine Learning con explicaciones SHAP.")
st.divider()

# ──────────────────────────────────────────────
# SIDEBAR — Configuración
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    uploaded_file = st.file_uploader("Subí tu CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**¿No tenés un dataset?**")
    use_demo = st.button("🎲 Usar dataset de demo (Churn)")

    st.markdown("---")
    st.markdown("### Modelos a entrenar")
    use_rf  = st.checkbox("Random Forest",           value=True)
    use_gb  = st.checkbox("Gradient Boosting",       value=True)
    use_lr  = st.checkbox("Logistic Regression",     value=True)

    st.markdown("---")
    test_size = st.slider("Tamaño del test set", 0.1, 0.4, 0.2, 0.05)
    st.caption(f"Train: {int((1-test_size)*100)}% · Test: {int(test_size*100)}%")

# ──────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────

@st.cache_data
def load_demo():
    """Genera un dataset de churn sintético."""
    np.random.seed(42)
    n = 800
    df = pd.DataFrame({
        "edad":             np.random.randint(18, 65, n),
        "meses_cliente":    np.random.randint(1, 72, n),
        "productos":        np.random.randint(1, 5, n),
        "llamadas_soporte": np.random.randint(0, 10, n),
        "uso_mensual":      np.round(np.random.uniform(10, 500, n), 2),
        "plan":             np.random.choice(["basic", "pro", "enterprise"], n),
        "pais":             np.random.choice(["AR", "MX", "CO", "CL"], n),
    })
    # Churn correlacionado con llamadas y uso bajo
    prob = (
        0.05 +
        0.08 * (df["llamadas_soporte"] > 5).astype(int) +
        0.10 * (df["uso_mensual"] < 50).astype(int) +
        0.07 * (df["meses_cliente"] < 6).astype(int)
    )
    df["churn"] = (np.random.rand(n) < prob).astype(int)
    return df

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# Determinar fuente de datos
df = None
if use_demo:
    df = load_demo()
    st.success("Dataset de demo cargado — 800 clientes con variable objetivo `churn`")
elif uploaded_file:
    df = load_csv(uploaded_file)
    st.success(f"Archivo cargado — {df.shape[0]:,} filas · {df.shape[1]} columnas")

# ──────────────────────────────────────────────
# EXPLORACIÓN DE DATOS
# ──────────────────────────────────────────────

if df is not None:
    st.markdown("## 📊 Vista del dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas",     f"{df.shape[0]:,}")
    col2.metric("Columnas",  df.shape[1])
    col3.metric("Numéricas", df.select_dtypes(include=np.number).shape[1])
    col4.metric("Nulos",     df.isnull().sum().sum())

    with st.expander("Ver primeras filas"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Estadísticas descriptivas"):
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.divider()

    # ── Selección de variable objetivo ──
    st.markdown("## 🎯 Seleccioná la variable objetivo")
    target_col = st.selectbox(
        "Variable a predecir (debe ser binaria o categórica con pocas clases)",
        options=df.columns.tolist(),
        index=len(df.columns) - 1
    )

    # ──────────────────────────────────────────────
    # PREPROCESAMIENTO
    # ──────────────────────────────────────────────

    @st.cache_data
    def preprocess(df, target):
        df = df.copy().dropna()
        X = df.drop(columns=[target])
        y = df[target]

        # Encodear categóricas
        le = LabelEncoder()
        for col in X.select_dtypes(include="object").columns:
            X[col] = le.fit_transform(X[col].astype(str))
        if y.dtype == object:
            y = le.fit_transform(y)

        return X, pd.Series(y), X.columns.tolist()

    X, y, feature_names = preprocess(df, target_col)

    st.info(f"Features usadas: **{len(feature_names)}** · Clases en target: **{y.nunique()}**")

    if y.nunique() > 10:
        st.warning("La variable objetivo tiene muchas clases únicas. Asegurate de que sea categórica.")
        st.stop()

    # ──────────────────────────────────────────────
    # ENTRENAMIENTO
    # ──────────────────────────────────────────────

    st.divider()
    st.markdown("## 🏋️ Entrenamiento de modelos")

    if not any([use_rf, use_gb, use_lr]):
        st.warning("Seleccioná al menos un modelo en el sidebar.")
        st.stop()

    if st.button("🚀 Entrenar modelos", type="primary", use_container_width=True):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        models = {}
        if use_rf:  models["Random Forest"]       = RandomForestClassifier(n_estimators=100, random_state=42)
        if use_gb:  models["Gradient Boosting"]   = GradientBoostingClassifier(n_estimators=100, random_state=42)
        if use_lr:  models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)

        results = {}
        trained = {}

        progress = st.progress(0, text="Entrenando...")
        for i, (name, model) in enumerate(models.items()):
            progress.progress((i) / len(models), text=f"Entrenando {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            results[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1":       round(f1_score(y_test, y_pred, average="weighted"), 4),
                "auc":      round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
            }
            trained[name] = (model, y_pred)

        progress.progress(1.0, text="✅ Listo")

        # ── Tabla comparativa de métricas ──
        st.markdown("### 📈 Comparativa de métricas")
        results_df = pd.DataFrame(results).T.reset_index()
        results_df.columns = ["Modelo", "Accuracy", "F1 Score", "AUC-ROC"]
        st.dataframe(
            results_df.style.highlight_max(subset=["Accuracy","F1 Score","AUC-ROC"],
                                           color="#D1FAE5"),
            use_container_width=True, hide_index=True
        )

        best_model_name = max(results, key=lambda k: results[k]["f1"])
        st.success(f"🏆 Mejor modelo por F1: **{best_model_name}** — F1: {results[best_model_name]['f1']}")

        # ── Matrices de confusión ──
        st.markdown("### 🔲 Matrices de confusión")
        cols = st.columns(len(trained))
        for col, (name, (model, y_pred)) in zip(cols, trained.items()):
            with col:
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(name, fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ──────────────────────────────────────────────
        # SHAP — EXPLICACIONES
        # ──────────────────────────────────────────────
        st.divider()
        st.markdown("## 🔍 Explicaciones SHAP")
        st.caption("SHAP muestra cuánto contribuye cada feature a la predicción. Valores positivos empujan hacia la clase 1, negativos hacia la 0.")

        shap_model_name = st.selectbox(
            "Seleccioná el modelo para explicar",
            options=list(trained.keys())
        )
        model_to_explain = trained[shap_model_name][0]

        with st.spinner("Calculando valores SHAP..."):
            if "Logistic" in shap_model_name:
                explainer = shap.LinearExplainer(model_to_explain, X_train)
            else:
                explainer = shap.TreeExplainer(model_to_explain)

            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🐝 Beeswarm", "🔎 Explicación individual"])

        with tab1:
            st.markdown("**Importancia global de features** — promedio del impacto absoluto SHAP")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, X_test, plot_type="bar",
                              feature_names=feature_names, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.markdown("**Beeswarm** — distribución de impacto SHAP por feature y valor")
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_test,
                              feature_names=feature_names, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab3:
            st.markdown("**Waterfall** — explicación de una predicción individual")
            idx = st.slider("Seleccioná un caso del test set", 0, len(X_test) - 1, 0)
            fig, ax = plt.subplots(figsize=(8, 4))
            shap_exp = shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1],
                data=X_test.iloc[idx].values,
                feature_names=feature_names
            )
            shap.plots.waterfall(shap_exp, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ──────────────────────────────────────────────
        # DESCARGA DE RESULTADOS
        # ──────────────────────────────────────────────
        st.divider()
        st.markdown("## 💾 Exportar resultados")
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        csv = shap_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar valores SHAP como CSV",
            data=csv,
            file_name="shap_values.csv",
            mime="text/csv"
        )

else:
    # ── Estado vacío ──
    st.markdown("## 👈 Empezá desde el sidebar")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Subí un CSV")
        st.caption("Cualquier dataset de clasificación binaria o multiclase.")
    with col2:
        st.markdown("### 2️⃣ Elegí tu target")
        st.caption("La variable que querés predecir.")
    with col3:
        st.markdown("### 3️⃣ Entrenás y explorás")
        st.caption("Métricas, matrices de confusión y explicaciones SHAP.")

    st.info("💡 Si no tenés un dataset a mano, usá el botón **'Usar dataset de demo'** para probar con datos de churn.")
