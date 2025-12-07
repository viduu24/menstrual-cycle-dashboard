def show_guide():
    st.header("📝 Input Recommendations for Accurate Predictions")
    
    tab1, tab2 = st.tabs(["📌 Phase Prediction Inputs", "📅 Cycle Length Prediction Inputs"])
    
    with tab1:
        st.subheader("📌 Recommended Input Ranges — Phase Prediction")
        st.markdown("""
### 🧪 **Hormones**
| Hormone | Follicular | Fertility | Luteal | Notes |
|--------|------------|-----------|--------|-------|
| **Estrogen (pg/mL)** | 30–120 | **150–350** | 50–150 | Peaks before ovulation |
| **PDG (ng/mL)** | < 5 | 2–10 | **10–25** | High after ovulation |
| **LH (mIU/mL)** | 2–10 | **20–80** | 1–10 | Spikes during ovulation |

### ⏳ **Cycle Day**
- Range: **1–28**

### ❤️ **Heart Rate Inputs**
| Metric | Typical Range |
|--------|---------------|
| **Mean HR (bpm)** | 55–95 bpm |
| **Lag-1 HR (bpm)** | 55–95 bpm |
| **Rolling 7-day HR (bpm)** | 55–95 bpm |
        """)
    
    with tab2:
        st.subheader("📅 Recommended Input Ranges — Cycle Length Prediction")
        st.markdown("""
### 🔢 **Cycle Length (Prior Cycles)**
- Enter **1–3 past cycle lengths**
- Typical: **24–35 days**

### 🩸 **Mean Menses Length**
- Typical: **3–7 days**

### 📊 **Daily Menses Scores**
| Value | Meaning |
|-------|---------|
| 0 | No bleeding |
| 1 | Spotting |
| 2 | Medium flow |
| 3 | Heavy bleeding |
        """)
