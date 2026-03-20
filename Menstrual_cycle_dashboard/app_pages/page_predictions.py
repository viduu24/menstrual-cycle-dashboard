import streamlit as st
import pandas as pd
import joblib
import os

def patch_tree_model(model):
    # patch the main model
    if not hasattr(model, "monotonic_cst"):
        model.monotonic_cst = None

    # patch all trees inside RandomForest
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None

    # extra safety
    if hasattr(model, "estimator_"):
        if not hasattr(model.estimator_, "monotonic_cst"):
            model.estimator_.monotonic_cst = None

    return model

# =============================
# MAIN PREDICTION FUNCTION
# =============================
def show_predictions():

    st.title("Machine Learning Models")
    st.markdown("Explore predictions from your trained ML models — Phase & Cycle Length.")

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")

    def load_pkl(name):
        return joblib.load(os.path.join(model_dir, name))

    # -------------------------------
    # Load models safely
    # -------------------------------
    try:
        m2_model = load_pkl("model2_phase_prediction.pkl")
        m2_scaler = load_pkl("model2_scaler.pkl")
        m2_encoder = load_pkl("model2_encoder.pkl")
        m2_features = load_pkl("model2_features.pkl")

        m3_model = load_pkl("model3_cycle_length.pkl")
        m3_model = patch_tree_model(m3_model)

        m3_scaler = load_pkl("model3_scaler.pkl")
        m3_features = load_pkl("model3_features.pkl")

        st.success("Models loaded successfully!")

    except Exception as e:
        st.error(f"❌ Failed to load ML models.\nError: {e}")
        st.stop()

    # -------------------------------
    # TABS
    # -------------------------------
    tab1, tab2 = st.tabs(["Phase Prediction (Model 1)", "Cycle Length Prediction (Model 2)"])

    # =======================================================
    # MODEL 2 — PHASE PREDICTION
    # =======================================================
    with tab1:
        st.header("Phase Prediction")
        st.markdown("Enter your values to predict the menstrual cycle phase.")

        cycle_day = st.number_input("Cycle Day (1–28)", min_value=1, max_value=28, value=1)
        estrogen = st.number_input("Estrogen (pg/mL)", value=0.0)
        pdg = st.number_input("PDG (ng/mL)", value=0.0)
        hr_mean = st.number_input("Mean Heart Rate (bpm)", value=60.0)

        if st.button("Predict Phase"):

            normalized_cycle_day = (cycle_day - 1) / 27

            X = {}
            for feat in m2_features:
                if feat == "cycle_day":
                    X[feat] = cycle_day
                elif feat == "normalized_cycle_day":
                    X[feat] = normalized_cycle_day
                elif feat == "estrogen":
                    X[feat] = estrogen
                elif feat == "pdg":
                    X[feat] = pdg
                elif feat == "hr_mean":
                    X[feat] = hr_mean
                else:
                    X[feat] = 0.0

            X = pd.DataFrame([X])
            X_scaled = m2_scaler.transform(X)

            pred_encoded = m2_model.predict(X_scaled)[0]
            pred_phase = m2_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"### Predicted Phase: **{pred_phase}**")

    # =======================================================
    # MODEL 3 — CYCLE LENGTH PREDICTION
    # =======================================================
    with tab2:
        st.header("Cycle Length Prediction")

        user_cycle = {}
        for feat in m3_features:
            user_cycle[feat] = st.number_input(f"{feat}", value=0.0, key=f"m3_{feat}")

        if st.button("Predict Cycle Length"):
            X = pd.DataFrame([user_cycle])
            X_scaled = m3_scaler.transform(X)
            pred = m3_model.predict(X_scaled)[0]

            st.success(f"### 📏 Predicted Cycle Length: **{pred:.1f} days**")

# =============================
# ENTRYPOINT CALLED BY APP
# =============================
def show(models=None):
    show_predictions()
