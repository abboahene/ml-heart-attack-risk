# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
import shap
import warnings
warnings.filterwarnings("ignore")

# --- Streamlit setup ---
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")
st.title("🫀 Heart Attack Risk Prediction & Analysis App")

# === 1. Load and Preprocess Data ===
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    df = pd.get_dummies(df, drop_first=True)
    return df

# === 2. Train Models ===
@st.cache_resource
def train_models(df):
    target_column = "Heart Attack Risk (Binary)"
    X = df.drop(columns=[target_column, "Heart Attack Risk (Text)"])
    y = df[target_column]

    smote_tomek = SMOTETomek(random_state=42)
    X_res, y_res = smote_tomek.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)

    return xgb, rf, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# === 3. Evaluation ===
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader(f"{name} Evaluation")
    st.text(f"Accuracy: {acc:.4f}")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# === 4. Feature Importance ===
def plot_feature_importance(model, feature_names, name):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_values = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_values, y=sorted_features, palette="viridis", ax=ax)
    ax.set_title(f"{name} Feature Importances")
    st.pyplot(fig)

# === 5. SHAP Explainability ===
@st.cache_resource
def get_shap_values(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    return explainer(X_test)

def plot_shap_summary(shap_values, feature_names):
    st.subheader("SHAP Summary Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(fig)

# === 6. Predict Individual Risk ===
def predict_individual(input_data, model, scaler, feature_names):
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df_input)

    pred = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1]

    st.markdown("### 🎯 Prediction Result")
    st.write(f"**Risk Level:** {'High' if pred == 1 else 'Low'}")
    st.write(f"**Probability of Heart Attack:** {proba:.2%}")

# === 7. Streamlit Main App ===
def main():
    uploaded_file = st.sidebar.file_uploader("Upload Heart Attack Dataset (.csv)", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

        xgb, rf, X_train, X_test, y_train, y_test, scaler, feature_names = train_models(df)

        st.markdown("---")
        evaluate_model(xgb, X_test, y_test, "XGBoost")
        evaluate_model(rf, X_test, y_test, "Random Forest")

        st.markdown("---")
        st.subheader("🎯 Feature Importances")
        plot_feature_importance(rf, feature_names, "Random Forest")
        plot_feature_importance(xgb, feature_names, "XGBoost")

        st.markdown("---")
        shap_values = get_shap_values(xgb, X_train, X_test)
        plot_shap_summary(shap_values, feature_names)

        st.markdown("---")
        st.subheader("🧑‍⚕️ Predict Individual Risk")

        with st.form("risk_form"):
            age = st.number_input("Age", 20, 100, 45)
            cholesterol = st.number_input("Cholesterol", 100, 400, 200)
            resting_bp = st.number_input("Resting BP", 80, 200, 120)
            max_hr = st.number_input("Max HR", 60, 220, 150)
            oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
            sex_m = st.selectbox("Sex", ["Female", "Male"]) == "Male"
            cp_typical = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"]) == "Typical"
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"]) == "Yes"

            submitted = st.form_submit_button("Predict Risk")
            if submitted:
                input_data = {
                    "Age": age,
                    "Cholesterol": cholesterol,
                    "Resting BP": resting_bp,
                    "Max HR": max_hr,
                    "Oldpeak": oldpeak,
                    "Sex_M": int(sex_m),
                    "Chest Pain Type_typical angina": int(cp_typical),
                    "Exercise Angina_Yes": int(exercise_angina)
                }
                predict_individual(input_data, xgb, scaler, feature_names)

# Run app
if __name__ == "__main__":
    main()
