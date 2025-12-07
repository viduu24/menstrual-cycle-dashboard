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
        st.markdown("###Demographic and Cycle Information")
        st.markdown("""
        - Source: Kaggle
        - Per-cycle dataset with demographic and cycle information  
        - Includes columns like **LengthofCycle, Age, BMI, OvulationDay**  
        """)
        
    elif dataset.startswith("Period 2"):
        df = period_2
        st.markdown("###Hormones and Symptoms Dataset")
        st.markdown("""
        - Source: PhysioNet
        - Daily dataset with hormones, symptoms, and heart rate  
        - Used for time-series analysis  
        """)
        
    else:
        df = period_3
        st.markdown("###Heart Rate and Hormones+Symptoms Dataset")
        st.markdown("""
        - PhysioNet
        - This dataset has the heart rate dataset combined with the symptoms, hormones dataset 
        - Contains engineered features combining heart rate, hormones, and symptoms  
        - Used for training **machine learning models**  
        """)

    # Show shape + preview
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("📊 Quick Visual Overview")

    # Identify numeric + categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # -------- 1️⃣ Basic Histogram for the First Numeric Column --------
    if len(numeric_cols) > 0:
        col = numeric_cols[0]   # choose first numeric column
        st.markdown(f"### 🔹 Distribution of **{col}**")

        fig = px.histogram(
            df, x=col,
            nbins=30,
            color_discrete_sequence=["#7E57C2"]
        )
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available for histogram.")

    # -------- 2️⃣ Basic Bar Chart for the First Categorical Column --------
    if len(categorical_cols) > 0:
        col = categorical_cols[0]  # choose first categorical column
        st.markdown(f"### 🔹 Distribution of **{col}**")

        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]

        fig = px.bar(
            counts, x=col, y="count",
            color=col,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns available for bar chart.")
