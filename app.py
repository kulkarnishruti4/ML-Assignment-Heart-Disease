
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title("Heart Disease Classification App")
st.write("Upload test data (CSV) and select a trained model to evaluate.")

# -----------------------------
# Load Scaler
# -----------------------------
scaler = joblib.load("model/scaler.pkl")

# Feature columns
feature_columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

# -----------------------------
# Model Selection
# -----------------------------
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]
model = joblib.load(model_path)

# -----------------------------
# Download Test CSV (80-20 Split)
# -----------------------------
st.write("### Download Test CSV (80-20 Split)")

try:
    test_df = pd.read_csv("test.csv")
    csv = test_df.to_csv(index=False)

    st.download_button(
        label="Download Test CSV",
        data=csv,
        file_name="test.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("test.csv not found in project directory.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    required_columns = feature_columns + ["target"]

    # Validate columns
    if not all(col in df.columns for col in required_columns):
        st.error("Uploaded CSV does not match required format.")
        st.write("Required columns:", required_columns)
        st.stop()

    # Ensure correct column order
    df = df[required_columns]

    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = 0.0

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.write("## Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.write("## Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{selected_model_name} Confusion Matrix")

    st.pyplot(fig)