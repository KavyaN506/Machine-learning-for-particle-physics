
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Higgs Boson Classification",
    page_icon="⚛️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():
    model_path = BASE_DIR / "model" / "random_forest_bundle.joblib"
    return joblib.load(model_path)

try:
    bundle = load_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# --------------------------------
# 3. SIDEBAR
# --------------------------------

st.sidebar.title("⚛️ Higgs ML Project")

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Model Results",
        "Feature Importance",
        "Live Prediction"
    ]
)

# --------------------------------
# 4. PROJECT OVERVIEW
# --------------------------------

if page == "Project Overview":

    st.title("Machine Learning for Particle Physics")
    st.subheader("Higgs Signal vs Background Classification")

    st.markdown("""
    This project explores how classical machine learning
    algorithms can classify simulated particle physics events.

    **Dataset:** UCI HIGGS Dataset

    **Task:** Binary classification

    - Signal: 1
    - Background: 0

    **Best performing model:** Random Forest
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Events", "80,000")
    col2.metric("Testing Events", "20,000")
    col3.metric("Input Features", "28")

    st.info(
        "Educational project using simulated Monte Carlo data. "
        "This is not a Higgs boson discovery or an analysis of raw "
        "experimental LHC collision data."
    )

# --------------------------------
# 5. MODEL RESULTS
# --------------------------------

elif page == "Model Results":

    st.title("Model Performance Comparison")

    csv_path = BASE_DIR / "results" / "model_comparison.csv"

    if csv_path.exists():
        results = pd.read_csv(csv_path)

        st.subheader("Model Comparison")
        st.dataframe(results, use_container_width=True)

    else:
        st.warning("Model comparison CSV was not found.")

    plot_path = BASE_DIR / "results" / "plots" / "roc_curves.png"

    if plot_path.exists():
        st.subheader("ROC Curves")
        st.image(str(plot_path), use_container_width=True)

    confusion_path = (
        BASE_DIR / "results" / "plots" / "confusion_matrices.png"
    )

    if confusion_path.exists():
        st.subheader("Confusion Matrices")
        st.image(str(confusion_path), use_container_width=True)

# --------------------------------
# 6. FEATURE IMPORTANCE
# --------------------------------

elif page == "Feature Importance":

    st.title("Feature Importance Analysis")

    plot_path = (
        BASE_DIR / "results" / "plots" / "feature_importance.png"
    )

    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)

    csv_path = (
        BASE_DIR / "results" / "random_forest_feature_importance.csv"
    )

    if csv_path.exists():
        importance = pd.read_csv(csv_path)
        st.subheader("Random Forest Feature Importance")
        st.dataframe(importance, use_container_width=True)

    else:
        st.warning("Feature importance CSV was not found.")

# --------------------------------
# 7. LIVE PREDICTION
# --------------------------------

elif page == "Live Prediction":

    st.title("Interactive Higgs Event Classification")

    st.write(
        "Enter values for the 28 input features to obtain "
        "a model prediction."
    )

    with st.form("prediction_form"):

        input_values = {}

        st.subheader("Particle Physics Features")

        columns = st.columns(2)

        for index, feature in enumerate(feature_names):
            with columns[index % 2]:
                input_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"feature_{index}"
                )

        submitted = st.form_submit_button("Classify Event")

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        signal_probability = probabilities[1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Predicted Class: Signal (1)")
        else:
            st.warning("Predicted Class: Background (0)")

        st.metric(
            "Signal Probability",
            f"{signal_probability * 100:.2f}%"
        )

        st.progress(float(signal_probability))

        st.caption(
            "This is a model score for the supplied simulated "
            "feature values, not a discovery-level probability."
        )

# --------------------------------
# 8. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Machine Learning for Particle Physics | "
    "Educational Portfolio Project"
)
