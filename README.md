# ML Explainer

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Una de las cosas que más me costaba explicar cuando trabajaba con datos era por qué un modelo tomaba cierta decisión. Este proyecto nació de esa necesidad — una app donde cualquiera pueda subir un dataset, entrenar modelos y ver exactamente qué features están empujando cada predicción, sin escribir una línea de código.

**[→ Probalo en vivo acá](https://app-ml-explainer-prphqw3mhezpqtfmax88sy.streamlit.app/)**

---

## Qué hace

Subís un CSV, elegís la variable que querés predecir y la app se encarga del resto:

- Entrena hasta 3 modelos en paralelo: Random Forest, Gradient Boosting y Logistic Regression
- Compara métricas: Accuracy, F1 Score y AUC-ROC con el mejor modelo destacado
- Muestra matrices de confusión para cada modelo
- Genera explicaciones SHAP en 3 vistas: importancia global, beeswarm y waterfall por caso individual
- Exporta los valores SHAP como CSV

Si no tenés un dataset a mano, tiene un modo demo con datos sintéticos de churn de clientes.

---

## Instalación local

```bash
git clone https://github.com/DatabyFede/streamlit-ml-explainer.git
cd streamlit-ml-explainer
pip install -r requirements.txt
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`

---

## ¿Qué es SHAP y por qué importa?

SHAP (SHapley Additive exPlanations) es una técnica que explica cuánto contribuye cada variable a una predicción individual. Es la diferencia entre saber que un modelo predice churn y saber *por qué* predice churn para ese cliente específico.

En la práctica, eso es lo que te pide un equipo de producto o de negocio — no el accuracy, sino una explicación que puedan entender y actuar.

---

## Lo que aprendí

Implementar SHAP con distintos tipos de modelos tiene sus particularidades — TreeExplainer para modelos de árboles y LinearExplainer para regresión logística, y los valores que devuelve tienen forma distinta según si es clasificación binaria o multiclase. Esa parte me llevó más tiempo del esperado pero valió la pena porque ahora entiendo qué hay detrás de esos gráficos que uno ve en papers y blogs.

---

## Próximos pasos

- [ ] Agregar soporte para regresión (no solo clasificación)
- [ ] Comparativa de curvas ROC interactiva con Plotly

---

## Otros proyectos

- [SQL Analytics Toolkit](https://github.com/DatabyFede/sql-analytics-toolkit) — cohortes, RFM, funnels y simulador de impacto en revenue
- [LATAM Macro Dashboard](https://github.com/DatabyFede/economic-viz-latam) — indicadores macroeconómicos de LATAM con datos reales del Banco Mundial

---

## Autor

**DatabyFede** · [LinkedIn](https://www.linkedin.com/in/federico-matyjaszczyk/) · [GitHub](https://github.com/DatabyFede)

> Parte de mi portfolio de proyectos de Data & IA — armado para volver al ruedo después de un tiempo fuera del mundo de los datos.
