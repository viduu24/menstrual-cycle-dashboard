import streamlit as st

def show():
    st.markdown("""
    <h1 style='text-align: center;'>🌸 Menstrual Cycle Analysis Dashboard</h1>
    """, unsafe_allow_html=True)

    st.markdown("""
<div style='text-align: justify;'>

This project aims to guide users — from cleaning missing values, looking into some EDA done for the datasets, exploring insights through visualizations, and finally integrating ML for predicting cycle phase and cycle length . The dashboard integrates hormone data, symptom logs, heart-rate patterns, and cycle information to reveal how biological and lifestyle factors shape the menstrual cycle.

The goal is to make menstrual health more understandable for everyone, especially beginners, by highlighting meaningful trends, correlations, and phase-based patterns across the cycle.

---

### 🔍 **Next Steps**
Use the sidebar to:

- View cleaned datasets with summary statistics  
- Explore visualizations showing hormone patterns, heart-rate changes, and symptom trends   
- Try out machine-learning powered predictions for cycle phase and cycle length  

</div>
""", unsafe_allow_html=True)

