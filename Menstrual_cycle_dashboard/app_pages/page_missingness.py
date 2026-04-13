import streamlit as st
from PIL import Image
import os

def show():
    st.header("Missingness Analysis")
    
    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Kaggle", "Hormones+symptoms"]
    )
    
    base_path = os.path.dirname(os.path.dirname(__file__))
    
    if dataset == "Kaggle":
        img_path = os.path.join(base_path, "missing_values_heatmap.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap - Kaggle Dataset", use_column_width=True)
        
    else:
        img_path = os.path.join(base_path, "missing_values_heatmap1.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap - Hormones+Symptoms Dataset", use_column_width=True)
        
        
    
       
