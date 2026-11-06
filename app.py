import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("Machine Learning Classification Model Evaluator")
st.write("BITS Pilani WILP - ML Assignment 2")

# Sidebar for Model Selection
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "SVM"]
)

# File Upload Option
uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if "target" not in data.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        
        try:
            # Load artifacts
            scaler = joblib.load("models/scaler.pkl")
            model = joblib.load(f"models/{model_choice}.pkl")
            
            # Preprocess
            X_test_scaled = scaler.transform(X_test)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = model.decision_function(X_test_scaled)
                
            # Compute Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display Metrics
            st.subheader(f"Evaluation Metrics: {model_choice.replace('_', ' ')}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix & Classification Report
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Class 0", "Class 1"],
                            yticklabels=["Class 0", "Class 1"])
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)
                
            with col_right:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose())
                
        except Exception as e:
            st.error(f"Error loading model or evaluating data: {e}")
else:
    st.info("Please upload `test_data.csv` to evaluate performance.")