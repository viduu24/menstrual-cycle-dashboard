import streamlit as st
import pandas as pd
import joblib
import os

def show_predictions(models):

    st.title("🤖 Machine Learning Models")
    st.markdown("Explore predictions from your trained ML models — Phase & Cycle Length.")

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")

    def load_pkl(name):
        return joblib.load(os.path.join(model_dir, name))

    try:
        # --- Load Model 2 (Phase Prediction) ---
        m2_model = load_pkl("model2_phase_prediction.pkl")
        m2_scaler = load_pkl("model2_scaler.pkl")
        m2_encoder = load_pkl("model2_encoder.pkl")
        m2_features = load_pkl("model2_features.pkl")

        # --- Load Model 3 (Cycle Length Regression) ---
        m3_model = load_pkl("model3_cycle_length.pkl")
        m3_scaler = load_pkl("model3_scaler.pkl")
        m3_features = load_pkl("model3_features.pkl")

        st.success("Models loaded successfully!")

    except Exception as e:
        st.error(f"❌ Failed to load ML models. Error: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📌 Phase Prediction (Model 2)", "📌 Cycle Length Prediction (Model 3)"])


    # ======================================================================================
    # 🌙 MODEL 2 — PHASE PREDICTION
    # ======================================================================================
    with tab1:
        st.header("📌 Phase Prediction")
        st.markdown("Enter your cycle + hormone inputs to predict the menstrual phase.")

        # NEW Model 2 user inputs (ONLY what users should enter)
        user_features = ["cycle_day", "normalized_cycle_day", "estrogen", "pdg", "hr_mean"]

        user_input = {}
        for feat in user_features:
            user_input[feat] = st.number_input(f"{feat}", value=0.0)

        if st.button("Predict Phase"):
            X = {}

            # Build full feature vector: user inputs + 0 for engineered ones
            for feat in m2_features:
                if feat in user_input:
                    X[feat] = user_input[feat]
                else:
                    X[feat] = 0.0  # placeholder

            X = pd.DataFrame([X])

            # Scale + predict
            X_scaled = m2_scaler.transform(X)
            pred_encoded = m2_model.predict(X_scaled)[0]
            pred_phase = m2_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"### 🎯 Predicted Phase: **{pred_phase}**")


    # ======================================================================================
    # 📏 MODEL 3 — CYCLE LENGTH PREDICTION
    # ======================================================================================
    with tab2:
        st.header("📌 Cycle Length Prediction")
        st.markdown("Predict a woman's menstrual cycle length in days.")

        user_cycle = {}
        for feat in m3_features:
            user_cycle[feat] = st.number_input(f"{feat}", value=0.0, key=f"m3_{feat}")

        if st.button("Predict Cycle Length"):
            X = pd.DataFrame([user_cycle])
            X_scaled = m3_scaler.transform(X)
            pred = m3_model.predict(X_scaled)[0]

            st.success(f"### 📏 Predicted Cycle Length: **{pred:.1f} days**")


def show(models):
    show_predictions(models)
