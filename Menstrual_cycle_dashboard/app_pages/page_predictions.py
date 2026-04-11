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

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cycle Info")
            cycle_day = st.number_input("Cycle Day (1–28)", min_value=1, max_value=28, value=1)

            st.subheader("Hormone Levels")
            estrogen = st.number_input("Estrogen (pg/mL)", value=94.2, min_value=0.0)
            pdg = st.number_input("PDG / Progesterone (ng/mL)", value=0.0, min_value=0.0)
            lh = st.number_input("LH (mIU/mL)", value=2.9, min_value=0.0)

            st.subheader("Heart Rate (from wearable)")
            hr_mean = st.number_input("Average Heart Rate (bpm)", value=75.0)
            hr_lag1 = st.number_input("Yesterday's Heart Rate (bpm)", value=75.0)
            hr_rolling_7d = st.number_input("7-Day Average Heart Rate (bpm)", value=75.0)
            hr_min = st.number_input("Min Heart Rate today (bpm)", value=60.0)
            hr_max = st.number_input("Max Heart Rate today (bpm)", value=100.0)

        with col2:
            st.subheader("Symptoms")
            options = ['Very Low/Little', 'Low', 'Moderate', 'High', 'Very High']
            encode_map = {'Very Low/Little': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}

            cramps      = st.selectbox("Cramps", options)
            fatigue     = st.selectbox("Fatigue", options)
            bloating    = st.selectbox("Bloating", options)
            moodswing   = st.selectbox("Mood Swings", options)
            sorebreasts = st.selectbox("Sore Breasts", options)

        if st.button("Predict Phase"):
            import numpy as np

            # derived features
            normalized_cycle_day = (cycle_day - 1) / 27
            cycle_week           = (cycle_day - 1) // 7 + 1
            is_ovulation         = 1 if cycle_day == 14 else 0
            hr_range             = hr_max - hr_min
            estrogen_log         = np.log1p(estrogen)
            pdg_log              = np.log1p(pdg)
            lh_log               = np.log1p(lh)
            estrogen_pdg_ratio   = estrogen / (pdg + 1e-5)

            cramps_enc      = encode_map[cramps]
            fatigue_enc     = encode_map[fatigue]
            bloating_enc    = encode_map[bloating]
            moodswing_enc   = encode_map[moodswing]
            sorebreasts_enc = encode_map[sorebreasts]
            total_symptoms  = cramps_enc + fatigue_enc + bloating_enc + moodswing_enc + sorebreasts_enc

            feature_values = {
                'cycle_day':            cycle_day,
                'normalized_cycle_day': normalized_cycle_day,
                'cycle_week':           cycle_week,
                'is_ovulation':         is_ovulation,
                'estrogen':             estrogen,
                'pdg':                  pdg,
                'lh':                   lh,
                'estrogen_log':         estrogen_log,
                'pdg_log':              pdg_log,
                'lh_log':               lh_log,
                'estrogen_pdg_ratio':   estrogen_pdg_ratio,
                'hr_mean':              hr_mean,
                'hr_rolling_7d':        hr_rolling_7d,
                'hr_lag1':              hr_lag1,
                'hr_range':             hr_range,
                'total_symptoms':       total_symptoms,
                'cramps_encoded':       cramps_enc,
                'fatigue_encoded':      fatigue_enc,
                'bloating_encoded':     bloating_enc,
                'moodswing_encoded':    moodswing_enc,
                'sorebreasts_encoded':  sorebreasts_enc,
            }

            X = pd.DataFrame([{f: feature_values.get(f, 0.0) for f in m2_features}])
            X_scaled = m2_scaler.transform(X)

            pred_encoded = m2_model.predict(X_scaled)[0]
            pred_phase   = m2_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"### Predicted Phase: **{pred_phase}**")

            with st.expander("See input features used"):
                st.dataframe(X)

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
