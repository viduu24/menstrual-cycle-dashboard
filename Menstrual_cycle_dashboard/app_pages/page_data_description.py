import streamlit as st
import plotly.express as px

def show(period_1, period_2, period_3):

    st.header("Dataset Overview")

    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Period 1 (Kaggle)", "Period 2 (PhysioNet)", "Merged Dataset (Used for ML Models)"]
    )

    # Select dataset
    if dataset.startswith("Period 1"):
        df = period_1
        st.markdown("### Demographic and Cycle Information")
        st.markdown("""
        - Source: Kaggle  
        - Per-cycle dataset  
        - Includes **LengthofCycle, LengthofMenses, Age, BMI, OvulationDay**  
        """)

    elif dataset.startswith("Period 2"):
        df = period_2
        st.markdown("### Hormones and Symptoms Dataset")
        st.markdown("""
        - Source: PhysioNet  
        - Daily dataset with hormones, symptoms, and heart rate  
        - Includes **phase**, HR, cramps, mood, sleep, etc.  
        """)

    else:
        df = period_3
        st.markdown("### Heart Rate + Hormones + Symptoms (Merged Dataset)")
        st.markdown("""
        - Source: PhysioNet  
        - Heart rate merged with hormone & symptom data  
        - Contains engineered features such as **hr_mean, hr_rolling_7d, estrogen_delta1**, etc.  
        - Used for ML model training  
        """)

    # Show basic info
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.markdown("---")

    # ----------------------------------------------------------
    # 🔢 STATISTICAL SUMMARY
    # ----------------------------------------------------------
    st.subheader("Statistical Summary (Numerical Columns)")
    st.write(df.describe())

    # ----------------------------------------------------------
    # 📊 DATASET-SPECIFIC VISUALIZATIONS
    # ----------------------------------------------------------
    st.markdown("---")
    

    # 1️⃣ PERIOD 1 — Circular plot of Length of Menses
    if dataset.startswith("Period 1"):
        if "LengthofMenses" in df.columns:
            st.markdown("### 🔹 Distribution of Length of Menses (in days)")
            fig = px.pie(
                df,
                names="LengthofMenses",
                title="Length of Menses Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("LengthofMenses column not found.")

    # 2️⃣ PERIOD 2 — Counts of each Phase (already available)
    elif dataset.startswith("Period 2"):
        if "phase" in df.columns:
            st.markdown("### 🔹 Distribution of Cycle Phases")
            phase_counts = df["phase"].value_counts().reset_index()
            phase_counts.columns = ["phase_encoded", "count"]

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
            st.info("Phase column not found.")

    # 3️⃣ MERGED DATASET — Heart Rate Plot
    else:
        # Detect HR column
        hr_cols = [c for c in df.columns if "hr" in c.lower()]
        
        if len(hr_cols) > 0:
            hr_col = hr_cols[0]  # use first HR-related column
            st.markdown(f"### 🔹 Heart Rate Overview — **{hr_col}**")

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
