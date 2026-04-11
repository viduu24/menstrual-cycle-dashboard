import streamlit as st

def show_guide():
    st.header("Input Recommendations for Accurate Predictions")

    tab1, tab2 = st.tabs(["Phase Prediction Inputs", "Cycle Length Prediction Inputs"])

    with tab1:
        st.subheader("Recommended Input Ranges — Phase Prediction")
        st.markdown("""
### Cycle Info
| Input | Range | Notes |
|-------|-------|-------|
| **Cycle Day** | 1–28 | Day 1 = first day of period |

### Hormone Levels
| Hormone | Follicular | Fertility | Luteal | Notes |
|---------|------------|-----------|--------|-------|
| **Estrogen (pg/mL)** | 30–120 | 150–350 | 50–150 | Peaks just before ovulation |
| **PDG / Progesterone (ng/mL)** | < 5 | 2–10 | 10–25 | Rises after ovulation |
| **LH (mIU/mL)** | 2–10 | 20–80 | 1–10 | Spikes at ovulation |

### Heart Rate (from wearable)
| Metric | Typical Range | Notes |
|--------|---------------|-------|
| **Average HR (bpm)** | 55–95 | Today's average |
| **Yesterday's HR (bpm)** | 55–95 | Previous day average |
| **7-Day Rolling HR (bpm)** | 55–95 | Weekly average |
| **Min HR today (bpm)** | 45–70 | Lowest recorded today |
| **Max HR today (bpm)** | 80–180 | Highest recorded today |

### Symptoms
| Level | Meaning |
|-------|---------|
| **Very Low/Little** | Barely noticeable |
| **Low** | Mild |
| **Moderate** | Noticeable but manageable |
| **High** | Significant |
| **Very High** | Severe |

Rate each symptom: **Cramps, Fatigue, Bloating, Mood Swings, Sore Breasts**
        """)

    with tab2:
        st.subheader("Recommended Input Ranges — Cycle Length Prediction")
        st.markdown("""
### Personal Info
| Input | Range | Notes |
|-------|-------|-------|
| **Age** | 10–60 | Years |
| **Height (cm)** | 100–220 | Used to auto-calculate BMI |
| **Weight (lbs)** | 66–440 | Converted to kg automatically for BMI |
| **BMI** | Auto-calculated | No need to enter manually |

### Cycle Info
| Input | Typical Range | Notes |
|-------|---------------|-------|
| **Estimated Day of Ovulation** | 11–17 | Usually around day 14 |
| **Length of Luteal Phase (days)** | 10–16 | Time from ovulation to period |
| **Length of Menses (days)** | 3–7 | How long your period lasts |

### Bleeding Info
| Input | Range | Notes |
|-------|-------|-------|
| **Mean Bleeding Intensity** | Very Light → Very Heavy | Overall average intensity |
| **Total High Flow Days** | 0–10 | Days with heavy bleeding |
| **Total Menses Score** | 0–30 | Sum of all daily scores |

### Daily Menses Score
| Score | Meaning |
|-------|---------|
| **0** | No bleeding |
| **1** | Spotting / Very light |
| **2** | Light flow |
| **3** | Moderate flow |
| **4** | Heavy bleeding |
| **5** | Very heavy bleeding |

Enter a score for each of the **first 5 days** of your period.
        """)

def show():
    show_guide()
