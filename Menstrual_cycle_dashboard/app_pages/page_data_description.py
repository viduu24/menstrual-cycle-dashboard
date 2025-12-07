import streamlit as st
import plotly.express as px
import pandas as pd

def show(period_1, period_2, period_3):

    st.header("Dataset Overview")

    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Period 1 (Kaggle)", "Period 2 (PhysioNet)", "Merged Dataset (Used for ML Models)"]
    )

    # ----------------------------------------------------------
    #  SELECT DATASET
    # ----------------------------------------------------------
    if dataset.startswith("Period 1"):
        df = period_1
        st.markdown("### Demographic and Cycle Information")
        st.markdown("""
        - Source: *Kaggle*  
        - Per-cycle dataset  
        - Includes columns such as **LengthofCycle**, **LengthofMenses**, **Age**, **BMI**, etc.  
        """)

    elif dataset.startswith("Period 2"):
        df = period_2
        st.markdown("### Hormones and Symptoms Dataset")
        st.markdown("""
        - Source: *PhysioNet*  
        - Daily dataset with **hormones**, **symptoms**, and **heart rate**  
        - Includes **phase**, HR, cramps, mood, sleep, etc.  
        """)

    else:
        df = period_3
        st.markdown("### Heart Rate + Hormones + Symptoms (Merged Dataset)")
        st.markdown("""
        - Heart rate merged with hormone & symptom data  
        - Contains engineered features (e.g., **hr_mean**, **hr_rolling_7d**, **estrogen_delta1**)  
        - Used for ML model training  
        """)

    # ----------------------------------------------------------
    #  FIX NUMERIC COLUMNS
    # ----------------------------------------------------------
    df = df.copy()
    df = df.apply(lambda x: pd.to_numeric(x, errors="coerce") if x.dtype == "object" else x)

    # ----------------------------------------------------------
    #  SHOW BASIC INFO
    # ----------------------------------------------------------
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.markdown("---")

    # ----------------------------------------------------------
    #  STATISTICAL SUMMARY
    # ----------------------------------------------------------
    st.subheader("📈 Statistical Summary")
    st.write(df.describe())

    st.markdown("---")
    

    # ----------------------------------------------------------
    # 1️⃣ PERIOD 1: PIE CHART – LENGTH OF MENSES
    # ----------------------------------------------------------
    if dataset.startswith("Period 1"):

        if "LengthofMenses" in df.columns:
            st.markdown("### Length of Menses (Pie Chart)")

            fig = px.pie(
                df,
                names="LengthofMenses",
                title="Length of Menses Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("LengthofMenses column not found in this dataset.")

    # ----------------------------------------------------------
    # 2️⃣ PERIOD 2: BAR CHART – PHASE COUNTS
    # ----------------------------------------------------------
    elif dataset.startswith("Period 2"):

        # Try phase first, fallback to encoded
        phase_col = None
        for col in ["phase", "phase_encoded", "Phase"]:
            if col in df.columns:
                phase_col = col
                break

        if phase_col:
            st.markdown("### Distribution of Cycle Phases")

            phase_counts = df[phase_col].value_counts().reset_index()
            phase_counts.columns = ["phase", "count"]

            fig = px.bar(
                phase_counts,
                x="phase",
                y="count",
                title="Counts of Each Phase",
                color="phase",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Phase column not found in this dataset.")

    # ----------------------------------------------------------
    # 3️⃣ MERGED DATASET: HEART RATE VISUALIZATION
    # ----------------------------------------------------------
    else:

        # Automatically detect heart-rate column
        hr_cols = [c for c in df.columns if "hr" in c.lower()]

        if len(hr_cols) > 0:
            hr_col = hr_cols[0]  # pick the first HR column
            st.markdown(f"### Heart Rate Distribution — **{hr_col}**")

            fig = px.histogram(
                df,
                x=hr_col,
                nbins=30,
                title=f"Distribution of {hr_col}",
                color_discrete_sequence=["#7E57C2"]
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No heart-rate-related columns found in merged dataset.")
