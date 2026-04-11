import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

def patch_tree_model(model):
    if not hasattr(model, "monotonic_cst"):
        model.monotonic_cst = None
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
    if hasattr(model, "estimator_"):
        if not hasattr(model.estimator_, "monotonic_cst"):
            model.estimator_.monotonic_cst = None
    return model

def show_predictions():

    st.title("Machine Learning Models")
    st.markdown("Explore predictions from your trained ML models — Phase & Cycle Length.")

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")

    def load_pkl(name):
        return joblib.load(os.path.join(model_dir, name))

    try:
        m2_model    = load_pkl("model2_phase_prediction.pkl")
        m2_scaler   = load_pkl("model2_scaler.pkl")
        m2_encoder  = load_pkl("model2_encoder.pkl")
        m2_features = load_pkl("model2_features.pkl")

        m3_model    = load_pkl("model3_cycle_length.pkl")
        m3_model    = patch_tree_model(m3_model)
        m3_scaler   = load_pkl("model3_scaler.pkl")
        m3_features = load_pkl("model3_features.pkl")

        st.success("Models loaded successfully!")

    except Exception as e:
        st.error(f"❌ Failed to load ML models.\nError: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["Phase Prediction (Model 1)", "Cycle Length Prediction (Model 2)"])

    # =======================================================
    # MODEL 1 — PHASE PREDICTION
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
            pdg      = st.number_input("PDG / Progesterone (ng/mL)", value=0.0, min_value=0.0)
            lh       = st.number_input("LH (mIU/mL)", value=2.9, min_value=0.0)

            st.subheader("Heart Rate (from wearable)")
            hr_mean       = st.number_input("Average Heart Rate (bpm)", value=75.0)
            hr_lag1       = st.number_input("Yesterday's Heart Rate (bpm)", value=75.0)
            hr_rolling_7d = st.number_input("7-Day Average Heart Rate (bpm)", value=75.0)
            hr_min        = st.number_input("Min Heart Rate today (bpm)", value=60.0)
            hr_max        = st.number_input("Max Heart Rate today (bpm)", value=100.0)

        with col2:
            st.subheader("Symptoms")
            options    = ['Very Low/Little', 'Low', 'Moderate', 'High', 'Very High']
            encode_map = {'Very Low/Little': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}

            cramps      = st.selectbox("Cramps", options)
            fatigue     = st.selectbox("Fatigue", options)
            bloating    = st.selectbox("Bloating", options)
            moodswing   = st.selectbox("Mood Swings", options)
            sorebreasts = st.selectbox("Sore Breasts", options)

        if st.button("Predict Phase"):

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

            X        = pd.DataFrame([{f: feature_values.get(f, 0.0) for f in m2_features}])
            X_scaled = m2_scaler.transform(X)

            pred_encoded = m2_model.predict(X_scaled)[0]
            pred_phase   = m2_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"### Predicted Phase: **{pred_phase}**")

            with st.expander("See input features used"):
                st.dataframe(X)

    # =======================================================
    # MODEL 2 — CYCLE LENGTH PREDICTION
    # =======================================================
    with tab2:
        st.header("Cycle Length Prediction")
        st.markdown("Enter your details from your **previous cycle** to predict your next cycle length.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Info")
            age    = st.number_input("Age", min_value=10, max_value=60, value=25)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=160.0)
            weight = st.number_input("Weight (lbs)", min_value=66.0, max_value=440.0, value=132.0)

            weight_kg = weight * 0.453592
            bmi       = weight_kg / ((height / 100) ** 2)
            st.info(f"BMI (auto-calculated): **{bmi:.1f}**")

            st.subheader("Previous Cycle Info")
            estimated_ovulation = st.number_input("Estimated Day of Ovulation (last cycle)", min_value=1, max_value=28, value=14)
            length_luteal_phase = st.number_input("Length of Luteal Phase last cycle (days)", min_value=1, max_value=20, value=14)
            length_menses       = st.number_input("Length of Last Period (days)", min_value=1, max_value=10, value=5)

        with col2:
            st.subheader("Bleeding Info (last period)")
            bleed_options = ["Very Light", "Light", "Moderate", "Heavy", "Very Heavy"]
            bleed_map     = {"Very Light": 1, "Light": 2, "Moderate": 3, "Heavy": 4, "Very Heavy": 5}

            mean_bleeding      = st.selectbox("Mean Bleeding Intensity", bleed_options, index=2)
            total_high_days    = st.number_input("Total Number of High Flow Days", min_value=0, max_value=10, value=2)
            total_menses_score = st.number_input("Total Menses Score", min_value=0.0, value=10.0)

            st.subheader("Daily Menses Score (last period)")
            day1 = st.number_input("Menses Score Day 1", min_value=0.0, value=3.0)
            day2 = st.number_input("Menses Score Day 2", min_value=0.0, value=3.0)
            day3 = st.number_input("Menses Score Day 3", min_value=0.0, value=2.0)
            day4 = st.number_input("Menses Score Day 4", min_value=0.0, value=1.0)
            day5 = st.number_input("Menses Score Day 5", min_value=0.0, value=1.0)

        if st.button("Predict Cycle Length"):
            feature_values = {
                "Age":                      age,
                "BMI":                      bmi,
                "Height":                   height,
                "Weight":                   weight_kg,
                "MeanBleedingIntensity":    bleed_map[mean_bleeding],
                "TotalNumberofHighDays":    total_high_days,
                "TotalMensesScore":         total_menses_score,
                "MensesScoreDayOne":        day1,
                "MensesScoreDayTwo":        day2,
                "MensesScoreDayThree":      day3,
                "MensesScoreDayFour":       day4,
                "MensesScoreDayFive":       day5,
                "EstimatedDayofOvulation":  estimated_ovulation,
                "LengthofLutealPhase":      length_luteal_phase,
                "LengthofMenses":           length_menses,
            }

            X        = pd.DataFrame([{f: feature_values.get(f, 0.0) for f in m3_features}])
            X_scaled = m3_scaler.transform(X)
            pred     = m3_model.predict(X_scaled)[0]

            st.success(f"### Predicted Cycle Length: **{pred:.1f} days**")
            st.caption("Cycle length can vary due to stress, sleep, and other lifestyle factors.")

            with st.expander("See input features used"):
                st.dataframe(X)

def show(models=None):
    show_predictions()
