import streamlit as st

def show():
    st.markdown("""
    <div style='text-align: justify;'>
### 🌺 Understanding the Menstrual Cycle
The menstrual cycle is a repeating process, typically lasting **28 days**, though it varies from person to person. It is driven by changing hormone levels and is divided into four main phases:

#### **Menstrual Phase**
The uterine lining sheds, marking the start of the cycle. Hormone levels are low, and symptoms like cramps or fatigue are common.

#### **Follicular Phase**
Estrogen rises as the body prepares an egg for release. The uterine lining thickens in case of pregnancy.

#### **Ovulation**
A surge in LH triggers the release of a mature egg. This is the most fertile part of the cycle.

#### **Luteal Phase**
Progesterone increases to support a potential pregnancy. If fertilization doesn’t occur, hormone levels drop and the cycle restarts.

Understanding these phases helps explain patterns in symptoms, hormones, and cycle length.
""", unsafe_allow_html=True)
