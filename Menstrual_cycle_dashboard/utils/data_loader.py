import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_data
def load_data():
    """Load all three datasets"""
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data_imputed.csv")
    final_path = os.path.join(base_path, "final_df.csv")
    merged_hr_dataset = os.path.join(base_path, "final_merged_hr_hormones.csv")

    period_1 = pd.read_csv(data_path)
    period_2 = pd.read_csv(final_path)
    period_3 = pd.read_csv(merged_hr_dataset)

    return period_1, period_2, period_3

@st.cache_resource
def load_models():
    """
    Load all trained ML models & scalers from the models/ folder.
    Cached so they load only once.
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")

    def load_pickle(name):
        path = os.path.join(model_dir, name)
        return joblib.load(path)

    models = {
        # Model 1 – Disabled
        "m1_model": None,
        "m1_scaler": None,
        "m1_features": None,

        # Model 2 – Phase prediction (LightGBM)
        "m2_model": load_pickle("model2_phase_prediction.pkl"),
        "m2_scaler": load_pickle("model2_scaler.pkl"),
        "m2_encoder": load_pickle("model2_encoder.pkl"),
        "m2_features": load_pickle("model2_features.pkl"),

        # Model 3 – Regularity prediction (RandomForestClassifier)
        "m3_model": load_pickle("model3_cycle_length.pkl"),
        "m3_scaler": load_pickle("model3_scaler.pkl"),
        "m3_features": load_pickle("model3_features.pkl"),
    }
    return models
