import streamlit as st

def show():
    st.title("<div style="text-align: center;">
    🌸 Menstrual Cycle Analysis Dashboard")
    st.markdown("""
<div style="text-align: center;">

This interactive dashboard helps users explore menstrual cycle patterns using real-world datasets.  
It is designed especially for people who **are new to menstrual health** and want to understand how  
different biological and lifestyle factors influence the menstrual cycle.

---

## 🎯 **What This App Does**

### **1. Clean & Prepare Menstrual Data**
The raw datasets contain hormone levels, symptoms, heart rate, and cycle information.  
The app automatically:
- Removes missing or incorrect values  
- Merges data from multiple sources  
- Creates meaningful features such as *cycle day*, *phase*, *hormone patterns*, and more  

---

### **2. Visualize Menstrual Cycle Patterns**
To help beginners understand the menstrual cycle, the app provides clear visualizations of:
- Daily hormone fluctuations  
- Heart-rate changes across the cycle  
- Symptom patterns (cramps, bloating, mood changes, etc.)  
- Cycle length and flow intensity trends  

These charts make it easier to see how the cycle works across different phases.

---

### **3. Explore Factors That Influence the Cycle**
The dashboard helps users investigate:
- How hormones affect cycle phases  
- Which symptoms are most common  
- How lifestyle factors such as sleep, stress, and heart rate correlate with cycle patterns  
- What predicts shorter or longer cycles  

---

## 🌼 **Who Is This Dashboard For?**
This app is built for:
- Individuals who want to **learn about their menstrual cycle**
- Students or researchers exploring menstrual health data
- Anyone curious about how hormone, symptom, or lifestyle patterns change across the cycle  

You **do not** need prior medical or data-science knowledge — the dashboard explains everything step by step.

---

## 💡 **Why This Matters**
Understanding menstrual health helps users:
- Recognize what's normal  
- Identify unusual patterns  
- Improve cycle tracking  
- Build awareness of how the body changes throughout the month  

This dashboard makes menstrual science accessible, visual, and easy to explore.

""")
