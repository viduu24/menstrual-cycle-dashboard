import streamlit as st
import plotly.express as px

def show(period_1, period_2, period_3):
    st.header("📄 Dataset Overview")
    
    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Period 1 (Kaggle)", "Period 2 (PhysioNet)", "Merged Dataset (Used for ML Models)"]
    )
    
    # Pick dataset
    if dataset.startswith("Period 1"):
        df = period_1
    elif dataset.startswith("Period 2"):
        df = period_2
    else:
        df = period_3
    
    # Show dataset shape + head
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    
    # About each dataset
    if dataset.startswith("Period 1"):
        st.markdown("""
            ###  About Period 1 (Kaggle Dataset)
            - Per-cycle dataset with demographic and cycle characteristics  
            - Includes columns such as:  
            **LengthofCycle**, **Age**, **BMI**, **TotalMensesScore**, **OvulationDay**
        """)
    
    elif dataset.startswith("Period 2"):
        st.markdown("""
            ###  About Period 2 (PhysioNet Dataset)
            - Daily hormone + symptoms dataset  
            - Includes **Heart Rate**, **Cramps**, **Mood**, **Sleep**, **Bloating**, etc.  
            - Used for time-series analysis.
        """)
    
    else:
        st.markdown("""
            ###  About Merged Dataset
            - Combination of PhysioNet + Hormones + Symptoms  
            - Used for all **Machine Learning models**  
            - Contains engineered features such as:  
            - **hr_rolling_7d**, **phase_lag1_enc**, **estrogen_delta1**,  
            - **pdg_z**, **normalized_cycle_day**, etc.
        """)
    
    # Summary statistics
    st.subheader("Statistical Summary (Numerical Columns)")
    st.write(df.describe())
    
    # Categorical summary
    st.subheader("Categorical Columns Breakdown")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            st.write(df[col].value_counts())
    else:
        st.write("No categorical columns found in this dataset.")
    
    # Initial visualizations
    st.subheader("📊 Exploratory Visualizations")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Distribution plots
    with st.expander(" Distribution Plots (Histogram + KDE)"):
        for col in numeric_cols:
            fig = px.histogram(
                df, x=col, nbins=40, marginal="box",
                title=f"Distribution of {col}",
                color_discrete_sequence=["#5C6BC0"]
            )
            fig.update_layout(
                template="simple_white",
                title_x=0.5,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    if len(numeric_cols) >= 3:
        with st.expander(" Correlation Heatmap"):
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=False,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
            fig.update_layout(title_x=0.5, height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # Boxplots
    with st.expander(" Boxplots (Outlier Detection)"):
        for col in numeric_cols:
            fig = px.box(
                df, y=col,
                title=f"Boxplot of {col}",
                color_discrete_sequence=["#AB47BC"]
            )
            fig.update_layout(template="simple_white", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # Bar charts for categorical
    if len(categorical_cols) > 0:
        with st.expander(" Categorical Feature Bar Charts"):
            for col in categorical_cols:
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "count"]
                
                fig = px.bar(
                    counts,
                    x=col,
                    y="count",
                    title=f"Counts of {col}",
                    color=col,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_layout(
                    template="simple_white",
                    xaxis_title=col,
                    yaxis_title="Count",
                    title_x=0.5,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
