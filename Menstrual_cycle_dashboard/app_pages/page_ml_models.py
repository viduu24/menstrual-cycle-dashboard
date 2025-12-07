import streamlit as st
def show_ml_models():
    st.markdown("""
# 🤖 Machine Learning Models in This Project

This section explains the machine learning models used in the dashboard, why they were chosen,  
and how they contribute to understanding menstrual cycle patterns.

The project includes **two primary models**:

1. **Phase Prediction Model (Model 2)** – predicts which menstrual phase a user is in  
2. **Cycle Length Prediction Model (Model 3)** – predicts a user's cycle length using symptoms and flow patterns  

Both models were designed to be:
- Small enough to run on Streamlit Cloud  
- Accurate and stable  
- Built using interpretable features  
- Compatible with scikit-learn 1.3.2

---

# 🌙 **Model 2: Phase Prediction (XGBoost)**

This model predicts the phase of the cycle using daily biological indicators.

## 📌 **Why XGBoost?**
- Handles non-linear relationships effectively  
- Performs well even when features have different scales  
- Compact model files  
- Captures subtle hormone–cycle interactions  

## 🎯 **Model Performance**
- **Training Accuracy:** 0.952  
- **Macro F1-Score:** 0.952

---

# 🩺 **Model 3: Cycle Length Prediction (Random Forest Regression)**

This model predicts the length of a user's cycle based on their symptoms and menstrual flow patterns.

## 🎯 **Model Performance**
- **RMSE:** 3.24 days  
- **MAE:** 2.39 days  
- **R² Score:** 0.245  

---

# 🧠 **Why These Models Matter**

### ⭐ Understand cycle daily  
The Phase Prediction model reveals where they are in the cycle.

### ⭐ Understand cycle monthly  
The Cycle Length model shows how bleeding patterns influence duration.

### ⭐ Gain accessible, data-driven insights  
Users can see how their body changes across the month.
""")

# REQUIRED by your app router
def show():
    show_ml_models()
