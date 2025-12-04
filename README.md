# MENSTRUAL CYCLE ANALYSIS AND PREDICTION DASHBOARD
This project uses three datasets:
https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data
https://www.physionet.org/content/mcphases/1.0.0/hormones_and_selfreport.csv
https://www.physionet.org/content/mcphases/1.0.0/heart_rate.csv
## Overview
This project builds a complete menstrual cycle analysis and prediction pipeline, integrating:
- Hormone levels (Estrogen, PDG, LH) </br>
- Heart rate and HRV features </br>
- Symptom logs </br>
- Daily cycle tracking data </br>
- Kaggle per-cycle dataset </br>
Using these inputs, the system trains three machine learning models:
Model 1 — Heart Rate Prediction (Regression)
Model 2 — Menstrual Phase Classification (Multiclass)
Model 3 — Next Cycle Length Prediction (Regression)
This pipeline is designed to support a Streamlit menstrual health application.
