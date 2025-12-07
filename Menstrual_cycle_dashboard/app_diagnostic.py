import streamlit as st

# Fix Python path
import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(CURRENT_DIR)

# Correct imports for package layout
from Menstrual_cycle_dashboard.utils.data_loader import load_data, load_models
from Menstrual_cycle_dashboard.app_pages import (
    page_readme,
    page_data_description,
    page_missingness,
    page_cleaning,
    page_information,
    page_graphs,
    page_ml_models,
    page_guide,
    page_predictions
)

# Configure page
st.set_page_config(
    page_title="Menstrual Cycle Dashboard",
    page_icon="🩸",
    layout="wide"
)

# Load data and models once
period_1, period_2, period_3 = load_data()
models = load_models()

# Sidebar Navigation
st.sidebar.title("🩸 Menstrual Cycle Dashboard")
page = st.sidebar.radio(
    "Go to", 
    [
        "README",
        "Data Description",
        "Missingness",
        "Cleaning Process",
        "Information",
        "Graphs",
        "ML models",
        "Guide on using the predictions feature",
        "Predictions"
    ]
)

# Route to appropriate page
if page == "README":
    page_readme.show()
    
elif page == "Data Description":
    page_data_description.show(period_1, period_2, period_3)
    
elif page == "Missingness":
    page_missingness.show()
    
elif page == "Cleaning Process":
    page_cleaning.show()
    
elif page == "Information":
    page_information.show()
    
elif page == "Graphs":
    page_graphs.show(period_1, period_2, period_3)
    
elif page == "ML models":
    page_ml_models.show()
    
elif page == "Guide on using the predictions feature":
    page_guide.show()
    
elif page == "Predictions":
    page_predictions.show(models)
