import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance

# -------------------
# Load Dataset
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Heart_disease_cleveland_new.csv")
    return df

# App Config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
df = load_data()

st.title("❤️ Heart Disease Prediction App")
st.markdown("An advanced ML-powered app to predict and analyze heart disease risk.")

# -------------------
# Preprocessing
# -------------------
X = df.drop("target", axis=1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert scaled arrays back to DataFrame for SHAP interpretability
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# -------------------
# Models
# -------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    }

results_df = pd.DataFrame(results).T

# -------------------
# Tabs for Navigation
# -------------------
tabs = st.tabs(["📊 Dataset", "📈 Model Results", "🔮 Prediction", "📉 Evaluation", "🧠 Explainability"])

# Dataset Tab
with tabs[0]:
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("### Shape:", df.shape)
    st.write("### Missing Values:")
    st.write(df.isnull().sum())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Model Results Tab
with tabs[1]:
    st.subheader("Model Performance Metrics")
    st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

    st.write("### Model Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    results_df[["Accuracy", "F1-Score", "ROC-AUC"]].plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    st.pyplot(fig)

# Prediction Tab
with tabs[2]:
    st.subheader("Try Your Own Prediction")
    with st.form("prediction_form"):
        cols = st.columns(3)
        input_data = []
        for i, col in enumerate(X.columns):
            with cols[i % 3]:
                val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                input_data.append(val)

        input_data = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        chosen_model = st.selectbox("Select Model", list(models.keys()))
        submitted = st.form_submit_button("Predict")

        if submitted:
            model = models[chosen_model]
            prediction = model.predict(input_scaled)[0]
            if prediction == 1:
                st.error("⚠️ Heart Disease Present")
            else:
                st.success("✅ No Heart Disease")

# Evaluation Tab
with tabs[3]:
    st.subheader("Confusion Matrix & ROC Curves")
    chosen_eval_model = st.selectbox("Select Model", list(models.keys()), key="cm")
    model = models[chosen_eval_model]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        st.write("### ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
        ax.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig)

        st.write("### Precision-Recall Curve")
        fig, ax = plt.subplots()
        ax.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        st.pyplot(fig)

# Explainability Tab
with tabs[4]:
    st.subheader("Feature Importance & SHAP Values")

    chosen_explain_model = st.selectbox("Select Model for Explanation", ["Random Forest", "XGBoost"])
    model = models[chosen_explain_model]

    st.write("### Feature Importance")
    if chosen_explain_model == "XGBoost":
        fig, ax = plt.subplots()
        plot_importance(model, ax=ax)
        st.pyplot(fig)
    else:
        importances = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind="barh", ax=ax)
        plt.title("Feature Importance")
        st.pyplot(fig)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_df)

    st.write("### SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values.values, X_test_df, plot_type="bar", show=False)
    st.pyplot(fig)

    st.write("### SHAP Force Plot (1 random sample)")
    idx = np.random.randint(0, X_test_df.shape[0])
    force_html = shap.plots.force(explainer.expected_value[0], shap_values.values[idx], X_test_df.iloc[idx, :], matplotlib=False)
    components.html(force_html.html(), height=300)

st.success("🚀 Advanced Heart Disease Prediction App Ready!")
