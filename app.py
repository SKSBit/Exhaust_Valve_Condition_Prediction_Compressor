
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Exhaust Valve Condition Prediction",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Exhaust Valve Condition Prediction")
st.write(
    "Machine Learning based classification of compressor exhaust valve "
    "condition as Clean or Dirty."
)

# ---------------------------------------------------------
# Feature List
# ---------------------------------------------------------

features = [
    'rpm',
    'motor_power',
    'torque',
    'outlet_pressure_bar',
    'air_flow',
    'noise_db',
    'outlet_temp',
    'wpump_outlet_press',
    'water_inlet_temp',
    'water_outlet_temp',
    'wpump_power',
    'water_flow',
    'oilpump_power',
    'oil_tank_temp',
    'gaccx',
    'gaccy',
    'gaccz',
    'haccx',
    'haccy',
    'haccz'
]

target = 'exvalve'

# ---------------------------------------------------------
# Load Saved Models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression":
            joblib.load("model/logistic_regression_model.pkl"),

        "Decision Tree":
            joblib.load("model/decision_tree_model.pkl"),

        "KNN":
            joblib.load("model/knn_model.pkl"),

        "Gaussian Naive Bayes":
            joblib.load("model/naive_bayes_model.pkl"),

        "Random Forest (Ensemble)":
            joblib.load("model/random_forest_model.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()

# ---------------------------------------------------------
# Upload Test Data
# ---------------------------------------------------------

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully.")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------------------------
    # Validate Dataset
    # -----------------------------------------------------

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        st.error(
            "The uploaded dataset is missing the following features:"
        )

        st.write(missing_features)

    elif target not in df.columns:

        st.error(
            "The uploaded dataset must contain the target column "
            "`exvalve` for model evaluation."
        )

    else:

        # -------------------------------------------------
        # Prepare Data
        # -------------------------------------------------

        X = df[features]
        y = df[target]

        # Convert Clean/Dirty to 0/1 if necessary
        if y.dtype == 'object':

            y = y.map({
                'Clean': 0,
                'Dirty': 1
            })

        y = y.astype(int)

        # -------------------------------------------------
        # Model Selection
        # -------------------------------------------------

        st.header("2. Model Selection")

        selected_model_name = st.selectbox(
            "Select Machine Learning Model",
            list(models.keys())
        )

        selected_model = models[selected_model_name]

        # -------------------------------------------------
        # Prepare Input According to Model
        # -------------------------------------------------

        scaled_models = [
            "Logistic Regression",
            "KNN",
            "Gaussian Naive Bayes"
        ]

        if selected_model_name in scaled_models:

            X_model = scaler.transform(X)

        else:

            X_model = X

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        y_pred = selected_model.predict(X_model)

        # Probability / Decision Score for AUC
        if hasattr(selected_model, "predict_proba"):

            y_score = selected_model.predict_proba(X_model)[:, 1]

        else:

            y_score = selected_model.decision_function(X_model)

        # -------------------------------------------------
        # Evaluation Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(y, y_pred)

        auc = roc_auc_score(y, y_score)

        precision = precision_score(
            y,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y,
            y_pred
        )

        # -------------------------------------------------
        # Display Metrics
        # -------------------------------------------------

        st.header("3. Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Clean", "Actual Dirty"],
            columns=["Predicted Clean", "Predicted Dirty"]
        )

        st.dataframe(cm_df)

        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y,
            y_pred,
            target_names=["Clean", "Dirty"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        # -------------------------------------------------
        # Prediction Summary
        # -------------------------------------------------

        st.header("6. Prediction Summary")

        clean_count = np.sum(y_pred == 0)
        dirty_count = np.sum(y_pred == 1)

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Clean",
            int(clean_count)
        )

        col2.metric(
            "Predicted Dirty",
            int(dirty_count)
        )

        st.info(
            f"Selected Model: {selected_model_name}"
        )

else:

    st.info(
        "Please upload test_data.csv to evaluate the machine learning models."
    )

# ---------------------------------------------------------
# Project Information
# ---------------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.write(
    "Exhaust Valve Condition Prediction"
)

st.sidebar.write(
    "Classification: Clean / Dirty"
)

st.sidebar.write(
    "Five ML models implemented"
)

st.sidebar.write(
    "Evaluation metrics: Accuracy, AUC, Precision, "
    "Recall, F1 and MCC"
)
