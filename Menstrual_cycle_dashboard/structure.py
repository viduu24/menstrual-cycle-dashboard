import streamlit as st
import os
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from PIL import Image
from scipy import stats
from scipy.signal import find_peaks
import joblib


class MenstrualCycleVisualizer:
    """
    Comprehensive visualization suite for heart rate and hormone data
    aligned with menstrual cycle phases - ALL ALTAIR VERSION.
    """
    
    def __init__(self, data_path):
        """
        Initialize visualizer with merged HR + hormone dataset.
        
        Args:
            data_path: Path to final_merged_hr_hormones.csv
        """
        self.df = pd.read_csv(data_path)
        # Configure Altair
        alt.data_transformers.disable_max_rows()
        print(f"✓ Loaded dataset: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
    # =====================================================
    # VISUALIZATION 1: Heart Rate Across Menstrual Phases
    # =====================================================
    
    def plot_hr_by_cycle_phase(self):
        """
        Box plots showing HR distribution across cycle phases using Altair.
        Returns Altair chart object for Streamlit.
        """
        print("\n📊 Creating HR by Cycle Phase visualization...")
        
        # Detect HR column name
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        # Define cycle phases
        if 'cycle_phase' in self.df.columns:
            phase_col = 'cycle_phase'
        elif 'phase' in self.df.columns:
            phase_col = 'phase'
        else:
            if 'day_in_study' in self.df.columns:
                self.df['cycle_phase'] = pd.cut(
                    self.df['day_in_study'] % 28,
                    bins=[0, 5, 14, 28],
                    labels=['Menstrual', 'Follicular', 'Luteal']
                )
                phase_col = 'cycle_phase'
            else:
                print("⚠ No cycle phase information found")
                return None
        
        # Prepare data
        plot_data = self.df[[phase_col, hr_col]].dropna()
        
        # Create box plot
        box_plot = alt.Chart(plot_data).mark_boxplot(
            size=60,
            opacity=0.7
        ).encode(
            x=alt.X(f'{phase_col}:N', title='Cycle Phase'),
            y=alt.Y(f'{hr_col}:Q', title='Heart Rate (bpm)', scale=alt.Scale(zero=False)),
            color=alt.Color(f'{phase_col}:N', 
                          scale=alt.Scale(scheme='set2'),
                          legend=None),
            tooltip=[
                alt.Tooltip(f'{phase_col}:N', title='Phase'),
                alt.Tooltip(f'min({hr_col}):Q', title='Min', format='.1f'),
                alt.Tooltip(f'q1({hr_col}):Q', title='Q1', format='.1f'),
                alt.Tooltip(f'median({hr_col}):Q', title='Median', format='.1f'),
                alt.Tooltip(f'q3({hr_col}):Q', title='Q3', format='.1f'),
                alt.Tooltip(f'max({hr_col}):Q', title='Max', format='.1f')
            ]
        ).properties(
            width=700,
            height=400,
            title='Heart Rate Distribution by Cycle Phase'
        ).interactive()
        
        return box_plot
    
    # =====================================================
    # VISUALIZATION 2: Time Series of HR + Hormones
    # =====================================================
    
    def plot_hr_hormone_timeseries(self, participant_id=None):
        """
        Time series showing HR and hormone levels together.
        Returns Altair chart object for Streamlit.
        """
        print("\n📊 Creating HR + Hormone Time Series...")
        
        if participant_id is None:
            participant_id = self.df['id'].iloc[0]
        
        # Filter for one participant
        df_p = self.df[self.df['id'] == participant_id].sort_values('day_in_study').copy()
        
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        # Detect hormone columns
        hormone_cols = [col for col in df_p.columns 
                       if any(x in col.lower() for x in ['estrogen', 'progesterone', 'lh', 'pdg'])]
        
        if not hormone_cols:
            print("⚠ No hormone columns found")
            return None
        
        # Create HR line chart
        hr_chart = alt.Chart(df_p).mark_line(
            color='#ef4444',
            strokeWidth=3,
            point=alt.OverlayMarkDef(filled=True, size=50)
        ).encode(
            x=alt.X('day_in_study:Q', title='Day in Study'),
            y=alt.Y(f'{hr_col}:Q', title='Heart Rate (bpm)', scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip('day_in_study:Q', title='Day'),
                alt.Tooltip(f'{hr_col}:Q', title='Heart Rate', format='.1f')
            ]
        ).properties(
            width=800,
            height=300,
            title=f'Heart Rate - Participant {participant_id}'
        )
        
        # Create hormone charts
        hormone_charts = []
        colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b']
        
        for i, (hormone_col, color) in enumerate(zip(hormone_cols[:4], colors)):
            valid_data = df_p[['day_in_study', hormone_col]].dropna()
            
            if len(valid_data) > 0:
                chart = alt.Chart(valid_data).mark_line(
                    color=color,
                    strokeWidth=3,
                    point=alt.OverlayMarkDef(filled=True, size=50)
                ).encode(
                    x=alt.X('day_in_study:Q', title='Day in Study'),
                    y=alt.Y(f'{hormone_col}:Q', 
                           title=hormone_col.replace('_', ' ').title(),
                           scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip('day_in_study:Q', title='Day'),
                        alt.Tooltip(f'{hormone_col}:Q', title=hormone_col.replace('_', ' ').title(), format='.2f')
                    ]
                ).properties(
                    width=800,
                    height=250,
                    title=hormone_col.replace('_', ' ').title()
                )
                hormone_charts.append(chart)
        
        # Combine charts vertically
        if hormone_charts:
            combined = alt.vconcat(hr_chart, *hormone_charts).resolve_scale(
                x='shared'
            )
            return combined
        else:
            return hr_chart
    
    # =====================================================
    # VISUALIZATION 3: Correlation Heatmap
    # =====================================================
    
    def plot_correlation_matrix(self):
        """
        Correlation heatmap with Altair.
        Returns Altair chart object for Streamlit.
        """
        print("\n📊 Creating Correlation Matrix...")
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Focus on HR and hormone columns
        hr_cols = [col for col in numeric_cols 
                  if any(x in col.lower() for x in ['hr_', 'heart_rate', 'bpm'])]
        hormone_cols = [col for col in numeric_cols 
                       if any(x in col.lower() for x in ['estrogen', 'progesterone', 'lh', 'pdg', 'testosterone'])]
        
        selected_cols = list(set(hr_cols + hormone_cols))[:10]  # Limit to 10 for readability
        
        if len(selected_cols) < 2:
            print("⚠ Not enough numeric columns for correlation")
            return None
        
        # Compute correlation matrix
        corr_matrix = self.df[selected_cols].corr()
        
        # Transform to long format
        corr_data = corr_matrix.stack().reset_index()
        corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']
        
        # Create heatmap
        heatmap = alt.Chart(corr_data).mark_rect().encode(
            x=alt.X('Variable 1:N', title=None),
            y=alt.Y('Variable 2:N', title=None),
            color=alt.Color('Correlation:Q',
                          scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                          title='Correlation'),
            tooltip=[
                alt.Tooltip('Variable 1:N'),
                alt.Tooltip('Variable 2:N'),
                alt.Tooltip('Correlation:Q', format='.3f')
            ]
        ).properties(
            width=600,
            height=600,
            title='Correlation Matrix: HR & Hormones'
        )
        
        # Add text annotations
        text = alt.Chart(corr_data).mark_text(baseline='middle').encode(
            x=alt.X('Variable 1:N', title=None),
            y=alt.Y('Variable 2:N', title=None),
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                alt.datum.Correlation > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        return (heatmap + text).interactive()
    
    # =====================================================
    # VISUALIZATION 4: Heart Rate Variability Analysis
    # =====================================================
    
    def plot_hrv_analysis(self):
        """
        Heart Rate Variability analysis with Altair.
        Returns Altair chart object for Streamlit.
        """
        print("\n📊 Creating HRV Analysis...")
        
        if 'hr_std' not in self.df.columns:
            print("⚠ No hr_std column for variability analysis")
            return None
        
        # Prepare data
        hrv_data = self.df[['hr_std']].dropna()
        
        # 1. HRV distribution histogram
        hist = alt.Chart(hrv_data).mark_bar(
            color='#0d9488',
            opacity=0.7
        ).encode(
            alt.X('hr_std:Q', bin=alt.Bin(maxbins=40), title='Heart Rate Std Dev (bpm)'),
            alt.Y('count()', title='Frequency'),
            tooltip=[
                alt.Tooltip('hr_std:Q', bin=alt.Bin(maxbins=40), title='HRV Range'),
                alt.Tooltip('count()', title='Count')
            ]
        ).properties(
            width=700,
            height=300,
            title='Distribution of Heart Rate Variability'
        )
        
        # Add mean and median lines
        mean_val = hrv_data['hr_std'].mean()
        median_val = hrv_data['hr_std'].median()
        
        mean_line = alt.Chart(pd.DataFrame({'value': [mean_val], 'label': ['Mean']})).mark_rule(
            color='red',
            strokeWidth=2,
            strokeDash=[5, 5]
        ).encode(
            x='value:Q'
        )
        
        median_line = alt.Chart(pd.DataFrame({'value': [median_val], 'label': ['Median']})).mark_rule(
            color='orange',
            strokeWidth=2,
            strokeDash=[5, 5]
        ).encode(
            x='value:Q'
        )
        
        hrv_chart = (hist + mean_line + median_line).interactive()
        
        # 2. HRV over time if available
        if 'day_in_study' in self.df.columns:
            daily_hrv = self.df.groupby('day_in_study')['hr_std'].mean().reset_index()
            
            time_chart = alt.Chart(daily_hrv).mark_line(
                color='#1e40af',
                strokeWidth=2,
                point=True
            ).encode(
                x=alt.X('day_in_study:Q', title='Day in Study'),
                y=alt.Y('hr_std:Q', title='Average HRV', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('day_in_study:Q', title='Day'),
                    alt.Tooltip('hr_std:Q', title='Avg HRV', format='.2f')
                ]
            ).properties(
                width=700,
                height=300,
                title='Heart Rate Variability Over Time'
            ).interactive()
            
            return alt.vconcat(hrv_chart, time_chart)
        
        return hrv_chart
    
    # =====================================================
    # VISUALIZATION 5: Multi-Participant Comparison
    # =====================================================
    
    def plot_participant_comparison(self, n_participants=6):
        """
        Small multiples showing HR patterns across multiple participants.
        Returns Altair chart object for Streamlit.
        """
        print("\n📊 Creating Participant Comparison...")
        
        participants = self.df['id'].unique()[:n_participants]
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        # Filter data for selected participants
        comparison_data = self.df[self.df['id'].isin(participants)].copy()
        comparison_data = comparison_data.sort_values(['id', 'day_in_study'])
        
        # Create small multiples
        chart = alt.Chart(comparison_data).mark_line(
            point=True,
            strokeWidth=2
        ).encode(
            x=alt.X('day_in_study:Q', title='Day in Study'),
            y=alt.Y(f'{hr_col}:Q', 
                   title='Heart Rate (bpm)',
                   scale=alt.Scale(zero=False)),
            color=alt.Color('id:N', legend=None),
            tooltip=[
                alt.Tooltip('id:N', title='Participant'),
                alt.Tooltip('day_in_study:Q', title='Day'),
                alt.Tooltip(f'{hr_col}:Q', title='Heart Rate', format='.1f')
            ]
        ).properties(
            width=250,
            height=200
        ).facet(
            facet=alt.Facet('id:N', title='Participant ID'),
            columns=3
        ).resolve_scale(
            y='independent'
        ).properties(
            title='Heart Rate Patterns Across Participants'
        ).interactive()
        
        return chart
    
    # =====================================================
    # VISUALIZATION 6: Phase Statistics
    # =====================================================
    
    def plot_phase_statistics(self):
        """
        Bar chart showing mean HR and statistics by phase.
        Returns Altair chart object for Streamlit.
        """
        print("\n📊 Creating Phase Statistics...")
        
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        # Define cycle phases
        if 'cycle_phase' in self.df.columns:
            phase_col = 'cycle_phase'
        elif 'phase' in self.df.columns:
            phase_col = 'phase'
        else:
            if 'day_in_study' in self.df.columns:
                self.df['cycle_phase'] = pd.cut(
                    self.df['day_in_study'] % 28,
                    bins=[0, 5, 14, 28],
                    labels=['Menstrual', 'Follicular', 'Luteal']
                )
                phase_col = 'cycle_phase'
            else:
                print("⚠ No cycle phase information found")
                return None
        
        # Calculate statistics
        phase_stats = self.df.groupby(phase_col)[hr_col].agg(['mean', 'std', 'count']).reset_index()
        phase_stats['sem'] = phase_stats['std'] / np.sqrt(phase_stats['count'])
        
        # Create bar chart with error bars
        bars = alt.Chart(phase_stats).mark_bar(
            opacity=0.7,
            color='steelblue'
        ).encode(
            x=alt.X(f'{phase_col}:N', title='Cycle Phase'),
            y=alt.Y('mean:Q', title='Mean Heart Rate (bpm)'),
            tooltip=[
                alt.Tooltip(f'{phase_col}:N', title='Phase'),
                alt.Tooltip('mean:Q', title='Mean HR', format='.2f'),
                alt.Tooltip('std:Q', title='Std Dev', format='.2f'),
                alt.Tooltip('count:Q', title='Sample Size')
            ]
        )
        
        # Error bars
        error_bars = alt.Chart(phase_stats).mark_errorbar(extent='stderr').encode(
            x=alt.X(f'{phase_col}:N'),
            y=alt.Y('mean:Q'),
            yError='sem:Q'
        )
        
        chart = (bars + error_bars).properties(
            width=600,
            height=400,
            title='Mean Heart Rate ± SEM by Cycle Phase'
        ).interactive()
        
        return chart
# --- Load cleaned data ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data_imputed.csv")
    final_path = os.path.join(base_path, "final_df.csv")
    merged_hr_dataset=os.path.join(base_path, "final_merged_hr_hormones.csv")

    period_1 = pd.read_csv(data_path)
    period_2 = pd.read_csv(final_path)
    period_3 = pd.read_csv(merged_hr_dataset)

    return period_1, period_2, period_3
    
period_1, period_2, period_3 = load_data()
#load models
@st.cache_resource
def load_models():
    """
    Load all trained ML models & scalers from the models/ folder.
    Cached so they load only once.
    """
    base_path = os.path.dirname(__file__)
    model_dir = os.path.join(base_path, "models")

    def load_pickle(name):
        path = os.path.join(model_dir, name)
        return joblib.load(path)

    models = {
        # Model 1 – Heart rate prediction (RandomForestRegressor)
                # Disable Model 1 (HR prediction)
        "m1_model": None,
        "m1_scaler": None,
        "m1_features": None,

        # Model 2 – Phase prediction (LightGBM)
        "m2_model": load_pickle("model2_phase_prediction.pkl"),
        "m2_scaler": load_pickle("model2_scaler.pkl"),
        "m2_encoder": load_pickle("model2_encoder.pkl"),
        "m2_features": load_pickle("model2_features.pkl"),

        # Model 3 – Regularity prediction (RandomForestClassifier)
        "m3_model": load_pickle("model3_cycle_length.pkl"),
        "m3_scaler": load_pickle("model3_scaler.pkl"),
        "m3_features": load_pickle("model3_features.pkl"),
    }
    return models

models = load_models()


def decode_phase(phase_encoded):
    phase_map = {1: 'Follicular', 2: 'Fertility', 3: 'Luteal', 4: 'Menstrual'}
    return phase_map.get(phase_encoded, 'Unknown')

# --- Sidebar Navigation ---
st.sidebar.title("🩸 Menstrual Cycle Dashboard")
page = st.sidebar.radio("Go to", ["README", "Data Description", "Missingness","Cleaning Process", "Information","Graphs", "ML models", "Guide on using the predictions feature", "Predictions"])

# --- Page 1: README ---
if page == "README":
    st.title("📘 Menstrual Cycle Analysis Dashboard")
    st.markdown("""
# 🌸 Menstrual Cycle Analysis Dashboard

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
- Recognize what’s normal  
- Identify unusual patterns  
- Improve cycle tracking  
- Build awareness of how the body changes throughout the month  

This dashboard makes menstrual science accessible, visual, and easy to explore.

""")


# --- Page 2: Data Description ---
elif page == "Data Description":
    st.header("📄 Dataset Overview")
    
    # Updated selectbox including the merged dataset
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
    
    # ---------------------- ABOUT EACH DATASET ----------------------
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
    
    # ---------------------- SUMMARY STATISTICS ----------------------
    st.subheader(" Statistical Summary (Numerical Columns)")
    st.write(df.describe())
    
    # ---------------------- CATEGORICAL SUMMARY ----------------------
    st.subheader("🗂 Categorical Columns Breakdown")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            st.write(df[col].value_counts())
    else:
        st.write("No categorical columns found in this dataset.")
    
    # ---------------------- INITIAL GRAPHS ----------------------
    # ---------------------- INITIAL EDA VISUALIZATIONS ----------------------
    st.subheader("📊 Exploratory Visualizations")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # --- 1. Distribution Plots ---
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
    
    # --- 2. Correlation Heatmap ---
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
    
    # --- 3. Boxplots for Distribution Spread ---
    with st.expander(" Boxplots (Outlier Detection)"):
        for col in numeric_cols:
            fig = px.box(
                df, y=col,
                title=f"Boxplot of {col}",
                color_discrete_sequence=["#AB47BC"]
            )
            fig.update_layout(template="simple_white", height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # --- 4. Bar Charts for Categorical Columns ---
    # --- 4. Bar Charts for Categorical Columns ---
    if len(categorical_cols) > 0:
        with st.expander(" Categorical Feature Bar Charts"):
            for col in categorical_cols:
    
                # Prepare dataframe safely
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "count"]   # rename properly
    
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



   
# --- Page 3: Missingness ---
elif page=="Missingness":
    dataset = st.selectbox("Select a dataset to view:", ["Kaggle", "Hormones+symptoms", "heart rate and hormones symptoms merged"])
    if dataset =="Kaggle":
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, "missing_values_heatmap.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap", use_column_width=True)
    elif dataset == "Hormones+symptoms":
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, "missing_values_heatmap1.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap", use_column_width=True)
    else:
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, "hr_missingvalues.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap", use_column_width=True)
# --- Page 4: Cleaning Process ---
elif page == "Cleaning Process":
    st.header("🧹 Data Cleaning Summary")
    dataset = st.selectbox("Select a dataset to view:", ["Kaggle", "Hormones+symptoms", "heart rate and hormones symptoms merged"])
    if dataset == "Kaggle":
        
        st.markdown("""
# Summary of the Imputation Process

This is a **multi-stage hybrid imputation pipeline** for menstrual cycle data with two main phases:

---

## **Phase 1: Demographic & Basic Variables**

### Step 1: Group-Based Imputation
- **Method**: Within-participant median filling
- **Variables**: Weight, Height, Age, Schoolyears, MeanCycleLength, MeanMensesLength
- **Logic**: Fill missing values using each participant's own median across their cycles

### Step 2: Targeted Weight Imputation
- **Method**: MICE (Multiple Imputation by Chained Equations) using Age & Height
- **Special feature**: BMI distribution preservation
  - Calculates median BMI from original data
  - Applies log-normal rescaling to maintain BMI distribution shape
  - Clips extreme outliers (1st-99th percentile)

### Step 3: Advanced Imputation for Remaining Variables
- **Method**: MICE 
- **Scope**: Any columns still missing after group imputation (excluding Weight)
- **Result**: BMI recalculated from imputed Weight/Height

---

## **Phase 2: Cycle-Specific Variables**

### Step 0: MensesScore Handling
- **All MensesScoreDay variables filled with 0** (not imputed)
- Rationale: Missing likely means no bleeding

### Step 1: Iterative Imputation (MICE)
- **Target variables** (9 total):
  - LengthofMenses, LengthofLutealPhase, MeanMensesLength
  - MeanBleedingIntensity, TotalNumberofHighDays, TotalMensesScore
  - TotalDaysofFertility, TotalFertilityFormula, Breastfeeding

- **Predictors used**:
  - Already-imputed demographics (Age, Weight, BMI, Schoolyears, MeanCycleLength)
  - Cycle characteristics (CycleNumber, LengthofCycle, EstimatedDayofOvulation)
  - Categorical variables (ReproductiveCategory, CycleWithPeakorNot)
  - Complete MensesScore columns

- **Configuration**:
  - 10 iterations max
  - BayesianRidge estimator (default)
  - Ascending imputation order (least to most missing)

### Step 2: Post-Processing
- Round categorical variables to integers
- Round count variables and clip to ≥0
- **Enforce logical constraints**:
  - LengthofMenses ≤ LengthofCycle
  - LengthofLutealPhase ≤ LengthofCycle

### Step 3: Fallback Safety Net
- If any NaN values remain: fill with median (numeric) or mode (categorical)
- Final verification ensures 100% completeness

---

## **Key Design Principles**

1. **Hierarchical approach**: Simple → Complex methods
2. **Context-aware**: Uses participant history before population patterns
3. **Biologically informed**: Preserves BMI distribution, enforces physiological constraints
4. **Complete coverage**: Guarantees zero missing values with fallback mechanisms
5. **Validation**: Includes distribution comparison and logical relationship checks

---

## **Correlation Analysis**
The final plot examines the relationship between `MeanCycleLength` and `EstimatedDayofOvulation` using the fully imputed dataset, showing how these menstrual variables correlate after the imputation process.
""")
        
        
        
    elif dataset== "Hormones+symptoms":
        st.markdown("""
        ### 1. Ordinal Encoding

- Each variable was mapped according to its natural order or intensity. 
- For example, the menstrual phase was encoded as: Follicular = 1, Fertility = 2, Luteal = 3, Menstrual = 4. 
- Symptom and lifestyle variables such as appetite, exercise level, headaches, cramps, sore breasts, fatigue, sleep issues, mood swings, stress, food cravings, indigestion, and bloating were encoded using six levels:

0 = “Not at all”
1 = “Very Low” or “Very Low/Little”
2 = “Low”
3 = “Moderate”
4 = “High”
5 = “Very High”

- Flow-related variables like flow volume and flow color had similar ordinal scales with more levels to capture intensity, with values increasing from the lightest or mildest category to the heaviest or most intense category. Each original column was transformed into a corresponding _encoded column, allowing for numerical analysis while preserving the inherent order of the categories.

---

### 2. Data Preparation
- MNAR was observed. It is seen that the days pdg was measured, the symptoms of the women were not stated. Perhaps the missingness is related to the method of testing pdg.
- Each participant is identified by **`id`**.
- Missing values in the hormone marker `pdg` are **filled with 0**.
- Only **numeric columns** (excluding `pdg`) are selected for imputation.

---

### 3. Per-User (Group-Wise) Imputation
- Data is imputed **within each user (id)** to maintain individual consistency.
- For each user:
  - If fewer than 3 records → simple **mean imputation**.
  - Otherwise → apply **Iterative Imputer** with a **Random Forest Regressor**.
- This captures non-linear relationships between features while being efficient.

---

### 4. Imputation Algorithm
- **Iterative Imputer (MICE approach)** repeatedly predicts missing values using other features.
- The **Random Forest model** provides flexible, robust predictions.
- Parameters optimized for speed:
  - `n_estimators=20`
  - `max_depth=6`
  - `max_iter=10`

---

### 5. Output & Validation
- Missing numeric values are filled for each participant.
- Groups with insufficient data fall back to median/mean imputation.
- The final dataset (`final_df`) is **fully imputed**, consistent, and ready for downstream analysis.

---

### ✅ Final Outcome
- **All missing numeric values imputed** per user group.  
- **Ordinal variables encoded** for machine learning.  
- **Logical consistency** and **efficiency maintained** across all steps.
        """)
    else:
        st.markdown("""
    ## 📥 1. Heart Rate Loading Process

    The raw heart rate dataset from MCPhases/Fitbit is extremely large  
    (hundreds of thousands of rows per participant).  
    Loading it fully would crash memory, so the process is **optimized**.

    ### 🔍 **Filtering HR Before Loading**
    Instead of loading the whole file, we load only rows that match:

    - ✔ Participants present in the hormone dataset  
    - ✔ Days that fall inside hormone dataset's date range  

    This reduces HR data to **only what is needed** for merging.

    ### 🧭 Steps Performed During Loading:
    - Detect the correct HR file (`bpm`, `heart_rate`, etc.)
    - Read it in **chunks of 100,000 rows**  
    - Filter each chunk by:
        - `id` (participants present)
        - `day_in_study` (matching hormone date range)
    - Rename `bpm` → `heart_rate`
    - Optimize memory by converting:
        - `float64 → float32`
        - `int64 → uint16 / int32`

    This lowers HR file size by **80–90%** before merging.

    ---
    """)

    st.markdown("""
    ## 📊 2. Aggregating Heart Rate (Daily)

    The original HR dataset may have many readings per day.

    We aggregate readings **per participant, per day** into:

    - `hr_mean`  
    - `hr_min`  
    - `hr_max`  
    - `hr_std`  
    - `hr_count` (number of samples collected that day)

    This results in a clean daily HR dataset aligned with daily hormone measurements.

    Example aggregation:

    ```
    groupby(['id', 'day_in_study']).agg({
        'heart_rate': ['mean', 'std', 'min', 'max', 'count']
    })
    ```

    Aggregation reduces:
    - Noise  
    - Random fluctuations  
    - Excessive data volume  

    And prepares HR for merging with symptoms + hormone values.

    ---
    """)

    st.markdown("""
    ## 💢 3. Introducing Realistic Missingness (Simulation)

    To demonstrate missing data handling, we introduce **15% simulated missingness**
    with realistic patterns:

    - Device glitches  
    - Participant forgets to wear watch  
    - Sensor issues at night  
    - Random dropout  
    - Multi-day missing blocks (device off)

    Patterns include:
    - Random missing  
    - Participant-bias missing  
    - Time-dependent missing (night hours)  
    - Block missing (multiple consecutive days)

    This simulates real Fitbit/MCPhases behavior.

    ---
    """)

    st.markdown("""
    ## 🔧 4. Time-Series Imputation (Advanced)

    Missing HR is imputed using **a multi-step time-series algorithm**:

    ### ✔ Trend estimation
    Smooths long-term HR changes.

    ### ✔ Weekly seasonality (day_in_study % 7)
    Captures biological rhythms (sleep/wake cycles).

    ### ✔ Linear interpolation
    Fills gaps between known values.

    ### ✔ Final fallback mean
    Only used if a participant has too few HR records.

    This produces **smooth, realistic HR curves** unlike simple mean imputation.

    ---
    """)

    st.markdown("""
    ## 🔗 5. Merging Heart Rate with Hormones

    After cleaning HR, it is merged with the hormone/self-report dataset using:

    ```
    merge on: id + day_in_study
    ```

    ⚡ The merged dataset contains:
    - Daily HR metrics  
    - Daily hormone values  
    - Symptoms  
    - Lifestyle features  
    - Engineered cycle-day features  

    This merged file is crucial for:
    - Heart Rate Prediction Model  
    - Phase Prediction Model  
    - Feature Engineering  
    - Final ML pipeline  

    Saved as:

    **`final_merged_hr_hormones.csv`**

    ---
    """)
        
elif page=="Information":
    st.markdown(""" ### 🌺 Understanding the Menstrual Cycle

The **menstrual cycle** is a natural, recurring process that prepares the body for possible pregnancy. It typically lasts about **28 days**, though cycles can range from **21 to 35 days** in adults. The cycle is divided into several key phases, each regulated by hormonal changes:

####  Menstrual Phase (Days 1–5)
- Marks the **start of the cycle**.  
- The **uterine lining (endometrium)** is shed through menstruation if no pregnancy has occurred.  
- Hormone levels (estrogen and progesterone) are at their lowest.  
- Common symptoms: cramps, fatigue, mood changes.

####  Follicular Phase (Days 1–13)
- Begins on the **first day of menstruation** and continues until ovulation.  
- The **pituitary gland releases FSH (follicle-stimulating hormone)**, prompting several ovarian follicles to mature.  
- **Estrogen levels rise**, thickening the uterine lining in preparation for potential implantation.

####  Ovulation Phase (Around Day 14)
- Triggered by a **surge in luteinizing hormone (LH)**.  
- The **mature egg is released** from the ovary and travels through the fallopian tube.  
- This is the **most fertile window** of the cycle.  
- Some individuals experience mild cramping or changes in body temperature.

####  Luteal Phase (Days 15–28)
- After ovulation, the ruptured follicle transforms into the **corpus luteum**, releasing **progesterone**.  
- Progesterone stabilizes the uterine lining for a potential pregnancy.  
- If fertilization does not occur, the corpus luteum breaks down, progesterone drops, and menstruation begins again.

---


Understanding these phases is essential for analyzing patterns in **symptoms**, **hormonal fluctuations**, and **cycle length variations** within the dataset.
""")
# --- Page 4: Graphs and EDA ---
elif page == "Graphs":
    dataset = st.selectbox("Select a dataset to view:", ["Kaggle", "Hormones+symptoms", "Heart rate Hormones symptoms merged"])
    if dataset == "Kaggle":
        st.header("📊 Period 1 Data Visualizations")

    # Sidebar: choose a plot
        with st.sidebar:
            st.subheader("🔍 Choose Visualization")
            plot_type = st.radio(
            "Select a visualization:",
            [
                "Age Distribution",
                "BMI Distribution",
                "Cycle Length Distribution",
                "Age vs Cycle Length (Box Plot)",
                "Luteal Phase Length Distribution",
                "Average Bleeding Intensity"
            ]
        )

        df = period_1.copy()

    # 1️⃣ Age Distribution
        if plot_type == "Age Distribution":
            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df.groupby("ClientID")["Age"].first(), bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of Age")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.markdown (""" The above graph shows the age distribution of the dataset. """)

    # 2️⃣ BMI Distribution
        elif plot_type == "BMI Distribution":
            st.subheader("BMI Distribution")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df.groupby('ClientID')["BMI"].first(), bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of BMI")
            ax.set_xlabel("BMI")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.markdown(""" The above graph shows the bmi distribution of the dataset.""")

    # 1. Cycle Length Distribution
        elif plot_type=="Cycle Length Distribution":
            st.subheader("📊 Cycle Length Distribution")
            df_cycle = df.dropna(subset=['LengthofCycle'])

            hist_cycle = alt.Chart(df_cycle).mark_bar(
            color='#ec4899',
            opacity=0.7
            ).encode(
                alt.X('LengthofCycle:Q', bin=alt.Bin(maxbins=30), title='Days'),
                alt.Y('count()', title='Count'),
                tooltip=[
                    alt.Tooltip('LengthofCycle:Q', bin=alt.Bin(maxbins=30), title='Cycle Length (Days)'),
                    alt.Tooltip('count()', title='Count')
                ]
            ).properties(
                width=600,
                height=400
            ).interactive()

            st.altair_chart(hist_cycle, use_container_width=True)
            st.markdown(""" From the bar chart it can be implied that in most women, the cycle length is about 26-32 days.""")
   

       # 2. Age vs Cycle Length (Boxplot)
        elif plot_type=='Age vs Cycle Length (Box Plot)':
            st.subheader("📦 Cycle Length by Age Group")
            df_age = df.dropna(subset=['Age', 'LengthofCycle']).copy()
            df_age['AgeGroup'] = pd.cut(df_age['Age'], bins=[17, 25, 30, 35, 40, 45], 
                                    labels=['18-25', '26-30', '31-35', '36-40', '41-45'])

            box_age = alt.Chart(df_age).mark_boxplot(
                size=50,
                color='#66c2a5',
                opacity=0.7
            ).encode(
                x=alt.X('AgeGroup:N', title='Age Group'),
                y=alt.Y('LengthofCycle:Q', title='Cycle Length (Days)'),
                color=alt.Color('AgeGroup:N', legend=None, scale=alt.Scale(scheme='set2')),
                tooltip=[
                    alt.Tooltip('AgeGroup:N', title='Age Group'),
                    alt.Tooltip('min(LengthofCycle):Q', title='Min', format='.1f'),
                    alt.Tooltip('q1(LengthofCycle):Q', title='Q1', format='.1f'),
                    alt.Tooltip('median(LengthofCycle):Q', title='Median', format='.1f'),
                    alt.Tooltip('q3(LengthofCycle):Q', title='Q3', format='.1f'),
                    alt.Tooltip('max(LengthofCycle):Q', title='Max', format='.1f')
                ]
            ).properties(
                width=600,
                height=400
            ).interactive()

            st.altair_chart(box_age, use_container_width=True)
            st.markdown(""" In this box plot, we see quite a few outliers, but it can be observed that after the age of 35 there is a slight change in the median of the cycle of days.
                        The presence of outliers show that the gap between subsequent periods can vary quite a bit for different women. So if your cycle is ocassionally 
                        not at the expected time, don't worry!""")

    # 5️⃣ Luteal Phase Length Distribution
        elif plot_type == "Luteal Phase Length Distribution":
            st.subheader("📊 Luteal Phase Length Distribution")
            df_luteal = df.dropna(subset=['LengthofLutealPhase'])

            hist_luteal = alt.Chart(df_luteal).mark_bar(
                color='#8b5cf6',
                opacity=0.7
            ).encode(
                alt.X('LengthofLutealPhase:Q', bin=alt.Bin(maxbins=20), title='Days'),
                alt.Y('count()', title='Count'),
                tooltip=[
                    alt.Tooltip('LengthofLutealPhase:Q', bin=alt.Bin(maxbins=20), title='Luteal Phase Length (Days)'),
                    alt.Tooltip('count()', title='Count')
                ]
            ).properties(
                width=600,
                height=400
            ).interactive()

            st.altair_chart(hist_luteal, use_container_width=True)
            st.markdown(""" Most women's luteal phase is about 12-15 days.""")


    # 8️⃣ Average Bleeding Intensity
        elif plot_type == "Average Bleeding Intensity":
            st.subheader("📊 Average Bleeding Intensity Over Menses Days")
            menses_cols = [f'MensesScoreDay{day}' for day in 
               ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']]
            menses_cols = [col for col in menses_cols if col in df.columns]
    # Compute averages (same logic as before)
            if menses_cols:
                menses_avg = df[menses_cols].mean()
                menses_data = pd.DataFrame({
                    'Day': [f'Day {i+1}' for i in range(len(menses_avg))],
                    'Average Score': menses_avg.values
                })

        # Create interactive Plotly line chart
                fig = px.line(
                    menses_data,
                    x='Day',
                    y='Average Score',
                    title='Average Bleeding Intensity Over Menses Days',
                    markers=True
                )

        # Style the chart
                fig.update_traces(
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=10)
                )
                fig.update_layout(height=500, template="plotly_white")

        # Display chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(""" From this line chart we can tell that the maximum bleeding is on day 2 of period for most women (It is important that you take rest and understand what your body needs). Most women menstruate for up upto 6 days. """)

            else:
                st.warning("No menses-related columns found for plotting.")
    elif dataset=="Hormones+symptoms":
        # Sidebar: choose a plot
        with st.sidebar:
            st.subheader("🔍 Choose Visualization")
            plot_type = st.radio(
            "Select a visualization:",
            [
                "Estrogen Levels by Cycle Phase",
                "LH Levels by Cycle Phase",
                "All Symptoms Across Phases",
                
            ]
        )
        st.title("Cycle Phase Hormone & Symptom Visualizations")
    

# --- Estrogen Levels by Phase ---
        # 1. Estrogen Levels by Cycle Phase
        if plot_type == "Estrogen Levels by Cycle Phase":
            st.subheader("📊 Estrogen Levels by Cycle Phase")
            final_df = period_2.copy()
    
    # Prepare data
            phase_hormones = final_df.groupby('phase_encoded')['estrogen'].mean().reset_index()
            phase_hormones['phase_name'] = phase_hormones['phase_encoded'].apply(decode_phase)
    
    # Create Altair chart
            estrogen_chart = alt.Chart(phase_hormones).mark_bar(
                color='#ef4444',
                opacity=0.8
            ).encode(
                x=alt.X('phase_name:N', title='Cycle Phase', sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('estrogen:Q', title='Estrogen Level'),
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('estrogen:Q', title='Estrogen Level', format='.2f')
                ]
            ).properties(
                width=600,
                height=400
            )
    
    # Add text labels on bars
            text = estrogen_chart.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontWeight='bold'
            ).encode(
                text=alt.Text('estrogen:Q', format='.2f')
            )
    
            final_chart = (estrogen_chart + text).interactive()
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown(""" Estrogen levels are highest during the fertility phase and lowest during the menstrual phase.""")
        # 2. LH Levels by Cycle Phase
        elif plot_type == "LH Levels by Cycle Phase":
            st.subheader("📊 LH Levels by Cycle Phase")
            final_df = period_2.copy()
    
    # Prepare data
            phase_hormones = final_df.groupby('phase_encoded')['lh'].mean().reset_index()
            phase_hormones['phase_name'] = phase_hormones['phase_encoded'].apply(decode_phase)
    
    # Create Altair chart
            lh_chart = alt.Chart(phase_hormones).mark_bar(
                color='#8b5cf6',
                opacity=0.8
            ).encode(
                x=alt.X('phase_name:N', title='Cycle Phase', sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('lh:Q', title='LH Level'),
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('lh:Q', title='LH Level', format='.2f')
                ]
            ).properties(
                width=600,
                height=400
            )
    
    # Add text labels on bars
            text = lh_chart.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontWeight='bold'
            ).encode(
                text=alt.Text('lh:Q', format='.2f')
            )
    
            final_chart = (lh_chart + text).interactive()
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown(""" LH levels are highest during the fertility phase and lowest during the luteal phase.""")

# 3. All Symptoms Across Phases
        elif plot_type == "All Symptoms Across Phases":
            st.subheader("📊 All Symptoms Across Cycle Phases - Grouped Comparison")
            final_df = period_2.copy()
    
            symptom_cols = ['headaches_encoded', 'cramps_encoded', 'sorebreasts_encoded', 
                            'fatigue_encoded', 'sleepissue_encoded', 'moodswing_encoded', 
                            'stress_encoded', 'foodcravings_encoded', 'indigestion_encoded', 
                            'bloating_encoded']
    
            symptom_names = ['Headaches', 'Cramps', 'Sore Breasts', 'Fatigue', 
                             'Sleep Issues', 'Mood Swings', 'Stress', 'Food Cravings', 
                             'Indigestion', 'Bloating']
    
    # Prepare data - melt to long format
            phase_symptoms = final_df.groupby('phase_encoded')[symptom_cols].mean().reset_index()
            phase_symptoms['phase_name'] = phase_symptoms['phase_encoded'].apply(decode_phase)
    
    # Melt the dataframe
            symptoms_melted = phase_symptoms.melt(
                id_vars=['phase_encoded', 'phase_name'],
                value_vars=symptom_cols,
                var_name='symptom',
                value_name='score'
            )
    
    # Map encoded symptom names to readable names
            symptom_mapping = dict(zip(symptom_cols, symptom_names))
            symptoms_melted['symptom_name'] = symptoms_melted['symptom'].map(symptom_mapping)
    
    # Create grouped bar chart
            symptoms_chart = alt.Chart(symptoms_melted).mark_bar(
                opacity=0.8
            ).encode(
                x=alt.X('phase_name:N', 
                        title='Cycle Phase',
                        sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('score:Q', title='Average Symptom Score'),
                color=alt.Color('symptom_name:N', 
                               title='Symptom',
                            scale=alt.Scale(scheme='set3')),
                xOffset='symptom_name:N',
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('symptom_name:N', title='Symptom'),
                    alt.Tooltip('score:Q', title='Score', format='.2f')
                ]
            ).properties(
                width=800,
                height=500
            ).interactive()
    
            st.altair_chart(symptoms_chart, use_container_width=True)
            st.markdown(""" During the menstrual phase, most symptoms start to or increase (especially cramps)""")
    else:
        st.header("📊 Merged Dataset Visualizations")
        df = period_3.copy()    # ← your merged dataset
        
                # Create visualizer safely
        try:
            viz = MenstrualCycleVisualizer(
                os.path.join(os.path.dirname(__file__), "final_merged_hr_hormones.csv")
            )
        except Exception as e:
            st.error(f"❌ Error loading merged dataset: {e}")
            st.stop()
        
                # Sidebar options for merged dataset
        with st.sidebar:
            st.subheader("🔍 Choose Visualization (Merged)")
            plot_type = st.radio(
                "Select a visualization:",
                [
                    "Heart Rate by Cycle Phase",
                    "HR + Hormone Timeseries",
                    "Correlation Heatmap",
                    "Heart Rate Variability (HRV)",
                    "Participant Comparison"
                ]
            )
        
                # Display plots
        if plot_type == "Heart Rate by Cycle Phase":
            viz.plot_hr_by_cycle_phase(save_path="merged_hr_by_phase.png")
            st.image("merged_hr_by_phase.png")
        
        elif plot_type == "HR + Hormone Timeseries":
            viz.plot_hr_hormone_timeseries(save_path="merged_hr_hormone_timeseries.png")
            st.image("merged_hr_hormone_timeseries.png")
        
        elif plot_type == "Correlation Heatmap":
            viz.plot_correlation_matrix(save_path="merged_corr_heatmap.png")
            st.image("merged_corr_heatmap.png")
        
        elif plot_type == "Heart Rate Variability (HRV)":
            viz.plot_hrv_analysis(save_path="merged_hrv.png")
            st.image("merged_hrv.png")
        
        elif plot_type == "Participant Comparison":
            viz.plot_participant_comparison(save_path="merged_participants.png")
            st.image("merged_participants.png")
elif page == "Guide on using the predictions feature":
    st.header("📝 Input Recommendations for Accurate Predictions")

    tab1, tab2 = st.tabs(["📌 Phase Prediction Inputs", "📅 Cycle Length Prediction Inputs"])
    
    # ============================================================
    # TAB 1 — PHASE PREDICTION INPUT RECOMMENDATIONS
    # ============================================================
    with tab1:
        st.subheader("📌 Recommended Input Ranges — Phase Prediction")
    
        st.markdown("""
    The phase prediction model uses **hormones**, **heart rate**, and **cycle day information**.
    
    Below are recommended ranges so users know what values to enter.
    
    ---
    
    ### 🧪 **Hormones**
    Typical values for healthy menstrual cycles:
    
    | Hormone | Follicular | Fertility / Ovulation | Luteal | Notes |
    |--------|------------|------------------------|---------|-------|
    | **Estrogen (pg/mL)** | 30–120 | **150–350** | 50–150 | Peaks before ovulation |
    | **PDG (ng/mL)** | < 5 | 2–10 | **10–25** | High after ovulation |
    | **LH (mIU/mL)** | 2–10 | **20–80 (peak)** | 1–10 | Spikes during ovulation |
    
    ✔ Users can enter values from **lab tests**, **home hormone trackers**, or **app-estimated values**.
    
    ---
    
    ### ⏳ **Cycle Day**
    - Range: **1–28**
    - Enter the cycle day you are currently on.
    - If unsure:  
      - Day 1 = first day of period  
      - Day ~14 = ovulation  
      - Days 15–28 = luteal phase  
    
    ---
    
    ### ❤️ **Heart Rate Inputs**
    These help improve accuracy but are optional.
    
    | Metric | Typical Range | How to Enter |
    |--------|----------------|--------------|
    | **Mean HR (bpm)** | 55–95 bpm | Enter daily average HR |
    | **Lag-1 HR (bpm)** | 55–95 bpm | Yesterday's HR |
    | **Rolling 7-day HR (bpm)** | 55–95 bpm | Average of last 7 days |
    
    If you don't track HR, leave blank (model handles missing values).
    
    ---
    
    ### 😣 **Symptoms (Aggregated Scores)**
    These are calculated from your dataset:
    
    | Feature | Meaning | Typical Range |
    |---------|---------|----------------|
    | **symptom_sum** | Sum of tracked symptoms | 0–30 |
    | **symptom_mean** | Average symptom level | 0–3 |
    | **total_symptoms** | Encoded phase symptoms | 0–10 |
    
    Users do NOT need to manually enter these — the model calculates them automatically.
    
    ---
    
    ### 🎯 Recommended Input Strategy
    For best accuracy:
    - Use **true hormone values** if available  
    - Provide **cycle day** accurately  
    - Enter **heart rate** if you use a smart watch  
    - You **do not** enter symptoms manually  
    
    """)
    
    # ============================================================
    # TAB 2 — CYCLE LENGTH PREDICTION INPUT RECOMMENDATIONS
    # ============================================================
    with tab2:
        st.subheader("📅 Recommended Input Ranges — Cycle Length Prediction")
    
        st.markdown("""
    This model predicts **next cycle length** using bleeding patterns across prior cycles.
    
    ---
    
    ### 🔢 **Cycle Length (Prior Cycles)**
    - Enter **1–3 past cycle lengths**
    - Typical cycle length range: **24–35 days**
    - Must be a *whole number*
    
    Example:
    - Cycle 1: 28  
    - Cycle 2: 30  
    - Cycle 3: 27  
    
    ---
    
    ### 🩸 **Mean Menses Length**
    Average number of days bleeding lasts.
    
    - Typical: **3–7 days**
    
    ---
    
    ### 🌡️ Mean Bleeding Intensity (0–3 scale)
    Use this scale:
    
    | Value | Meaning |
    |-------|---------|
    | **0** | No bleeding |
    | **1** | Light |
    | **2** | Medium |
    | **3** | Heavy |
    
    ---
    
    ### 📊 **Daily Menses Scores (Day 1–Day 10)**
    Users enter values **0–3**:
    
    | Value | Meaning |
    |-------|---------|
    | 0 | No bleeding |
    | 1 | Spotting / very light |
    | 2 | Medium / normal flow |
    | 3 | Heavy bleeding |
    
    You *do not* need to fill all 10 days.  
    Most users bleed 4–7 days.
    
    Example entry:
    - Day 1: 2  
    - Day 2: 3  
    - Day 3: 2  
    - Day 4: 1  
    - Days 5–10: 0  
    
    ---
    
    ### 📌 Additional Features the Model Uses (No Input Required)
    These come from your dataset:
    - Total menses score  
    - Total fertility formula  
    - CycleWithPeakOrNot  
    
    Users do *not* enter these manually.
    
    ---
    
    ### 🎯 Recommended Input Strategy
    To get the most accurate cycle length prediction:
    - Enter **at least 2 previous cycle lengths**
    - Accurately record daily bleeding for 3–7 days  
    - Do NOT enter unrealistic scores (e.g., 10 or -1)
    
    ---
    
    ### 🔍 Example Prediction Flow
    User enters:
    
    - Cycle 1: 28  
    - Cycle 2: 30  
    - Mean menses length: 5  
    - Menses scores: [2,3,2,1,0,0,0]
    
    The model predicts:
    **Next cycle length ≈ 29 days**
    """)

elif page == "ML models":
    import streamlit as st
    import pandas as pd
    st.markdown("""
# 🤖 Machine Learning Models in This Project

This section explains the machine learning models used in the dashboard, why they were chosen,  
and how they contribute to understanding menstrual cycle patterns.

The project includes **two primary models**:

1. **Phase Prediction Model (Model 2)** – predicts which menstrual phase a user is in  
2. **Cycle Length Prediction Model (Model 3)** – predicts a user’s cycle length using symptoms and flow patterns  

Both models were designed to be:
- Small enough to run on Streamlit Cloud  
- Accurate and stable  
- Built using interpretable features  
- Compatible with scikit-learn 1.3.2 (required to avoid pickle errors on Streamlit)

---

# 🌙 **Model 2: Phase Prediction (XGBoost)**
The menstrual cycle is divided into four physiological phases:

- **Follicular Phase**  
- **Fertile Window**  
- **Luteal Phase**  
- **Menstrual Phase**

This model predicts the phase of the cycle using daily biological indicators.

## 📌 **Why XGBoost?**
XGBoost was chosen because:
- It handles non-linear relationships effectively  
- It performs well even when features have different scales  
- It is compact and produces small model files (critical for GitHub + Streamlit deployment)  
- It captures subtle hormone–cycle interactions much better than logistic regression or basic random forests  

## 📌 **Features Used**
To make the model biologically relevant, we engineered high-value features:

### **Hormones**
- Estrogen  
- PDG  
- LH  
- First-order changes (Δ1): how hormones change from yesterday  
- Z-scores: normalized hormone fluctuations  
- Log-scaled hormone values  

### **Cycle Timing**
- Cycle day (1–28)  
- Normalized cycle position  
- Cycle week  
- Ovulation indicator (day 12–16)  

### **Heart-Rate Biomarkers**
- Mean heart rate  
- 1-day lagged heart rate  
- 7-day rolling average of heart rate  

### **Symptoms**
- Total symptom score per day  
- Number of symptoms present  
- Average symptom intensity  

This creates a rich representation of the physiological state of each day.

---

## 🎯 **Model Performance**
After hyperparameter tuning and evaluation:

### **Training Accuracy:** **0.952**  
### **Macro F1-Score:** **0.952**

These results indicate that the model can **very reliably distinguish between the four phases**.

### ✔ Strengths of the Model
- Excellent prediction of **Fertile Window**, which is typically the hardest phase to classify  
- Strong separation between **Follicular vs. Luteal** phases  
- Hormone patterns are captured extremely well  
- Very low confusion across classes  

---

# 🩺 **Model 3: Cycle Length Prediction (Random Forest Regression)**

This model predicts the length of a user's cycle based on their symptoms and menstrual flow patterns.

## 📌 Why Predict Cycle Length?
Cycle length is one of the most important indicators of menstrual health.  
Predicting it helps users understand:
- Whether their cycle is regular  
- How symptoms correlate with longer or shorter cycles  
- How flow intensity and menses duration relate to cycle timing  

This model transforms the qualitative symptom data into meaningful numerical predictors.

---

## 📌 **Features Used**
Model 3 uses the Kaggle dataset, which contains cycle-level summary features:

### **Cycle Flow and Symptom Features**
- Mean bleeding intensity  
- Total number of high-flow days  
- Total menstrual score  
- Menses duration  
- Daily bleeding scores (Day 1–Day 10)  

These features capture **how heavy**, **how long**, and **how consistent** a cycle is — all strong predictors of cycle length.

---

## 🎯 **Model Performance**
Evaluation on the test dataset gives:

- **RMSE:** 3.24 days  
- **MAE:** 2.39 days  
- **R² Score:** 0.245  

### ✔ Interpretation
Cycle length is **difficult to predict** due to high biological variability.  
However:
- The model captures general trends effectively  
- Heavy or longer bleeding patterns correspond to longer cycles  
- Short cycles tend to have shorter or lower-flow bleeding  

This model is **explanatory**, not diagnostic — it helps users understand relationships between flow patterns and timing.

---

# 🧠 **Why These Models Matter**
Together, the models allow users to:

### ⭐ Understand their cycle daily  
The Phase Prediction model reveals where they are in the cycle using hormones, symptoms, and heart rate.

### ⭐ Understand their cycle monthly  
The Cycle Length model shows how bleeding patterns influence overall cycle duration.

### ⭐ Gain accessible, data-driven insights  
Without needing medical knowledge, users can see how their body changes across the month.

---

# 🎓 **Academic Significance**
This project demonstrates:
- Time-series feature engineering  
- Hormone modeling  
- Physiological signal integration (heart rate + symptoms)  
- Multi-class classification  
- Regression modeling  
- Model deployment and persistence  
- Cloud compatibility constraints (model size, pickle versions)  

It provides a complete machine-learning pipeline applied to a real human physiology use-case.

""")

    
    # -------------------------------------
    # LOAD METRICS THAT YOU REPORTED EARLIER
    # -------------------------------------
    
    # MODEL 2 — Phase Prediction (Random Forest)
   # MODEL 2 — Phase Prediction (XGBoost — NEW HIGH ACCURACY MODEL)
    model2_metrics = {
        
       "Training Accuracy": 0.952,
       "Macro F1-score": 0.952,
       
    
        "Model Description": """
        ### 🧠 XGBoost Phase Prediction Model
        This model classifies each day into:
        
        **• Follicular  
        • Fertility  
        • Luteal  
        • Menstrual**
        
        It uses a rich feature set combining:
        - Estrogen, PDG, LH  
        - Hormone dynamics (Δ1 changes)
        - Z-scores  
        - Cycle day + normalized phase  
        - Heart Rate (mean, lag-1, rolling 7-day)
        - Symptom totals
        
        This improves the biological interpretability and robustness of the model.
        """,
        
            "Model Interpretation": """
        ### 📈 Model Interpretation — Why Accuracy Is High
        - The XGBoost model learns nonlinear hormone + HR patterns  
        - Fertility phase is detected more reliably than Random Forest  
        - Minimal confusion between Follicular ↔ Luteal  
        - Normalized cycle position helps align cycles of different lengths  
        
        Overall:
        **A highly stable and biologically meaningful classifier.**
        """
    }
    

    # MODEL 3 — Cycle Length Prediction (Regression)
    model3_metrics = {
        "RMSE": 3.244,
        "MAE": 2.395,
        "R² Score": 0.245,
        "Model Description": """
    The Cycle Length Prediction Model predicts menstrual cycle length 
    using bleeding intensity, menses scores, and hormonal patterns.
    
    Features used:
    - Mean bleeding intensity  
    - Total high-flow days  
    - Daily menses scores (Day 1–5)  
    - Total menses score  
    """,
        "Interpretation": """
    An RMSE of ~3.24 means the model is typically within **±3 days** 
    of the true cycle length. This is reasonable because:
    • Cycle length naturally varies 2–5 days between cycles  
    • Hormonal recordings are noisy  
    • Users differ widely in cycle patterns  
    """
    }
    
    
    # -------------------------------------
    # STREAMLIT PAGE LAYOUT
    # -------------------------------------
    

   # =========================================================
#                   📘   ML MODELS PAGE
# =========================================================
elif page == "Predictions":

    st.title("🤖 Machine Learning Models")
    st.markdown("Explore predictions from your trained ML models — Phase & Cycle Length.")

    # -----------------------------
    # LOAD SAVED MODELS
    # -----------------------------
    base_path = os.path.dirname(__file__)
    model_dir = os.path.join(base_path, "models")

    def load_pkl(name):
        return joblib.load(os.path.join(model_dir, name))

    # LOAD MODEL 2 + MODEL 3 ONLY
    try:
        # model 2
        m2_model = load_pkl("model2_phase_prediction.pkl")
        m2_scaler = load_pkl("model2_scaler.pkl")
        m2_encoder = load_pkl("model2_encoder.pkl")
        m2_features = load_pkl("model2_features.pkl")

        # model 3
        m3_model = load_pkl("model3_cycle_length.pkl")
        m3_scaler = load_pkl("model3_scaler.pkl")
        m3_features = load_pkl("model3_features.pkl")

        st.success("Models loaded successfully!")

    except Exception as e:
        st.error("❌ Failed to load ML models. Check your /models folder.")
        st.stop()

    # ------------------------------------
    # TABS FOR MODELS
    # ------------------------------------
    tab1, tab2 = st.tabs(["📌 Phase Prediction (Model 2)", "📌 Cycle Length Prediction (Model 3)"])


    # ============================================================
    #                       MODEL 2 — PHASE
    # ============================================================
    with tab1:
        st.header("📌 Phase Prediction")
        st.markdown("Use hormone + cycle features to predict: **Menstrual, Follicular, Fertile, Luteal**")

        # User Inputs
        user_input = {}
        for feat in m2_features:
            user_input[feat] = st.number_input(f"{feat}", value=0.0)

        if st.button("Predict Phase"):
            X = pd.DataFrame([user_input])
            X_scaled = m2_scaler.transform(X)
            pred_encoded = m2_model.predict(X_scaled)[0]
            pred_phase = m2_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"### 🎯 Predicted Phase: **{pred_phase}**")


    # ============================================================
    #                   MODEL 3 — CYCLE LENGTH
    # ============================================================
    with tab2:
        st.header("📌 Cycle Length Prediction")
        st.markdown("Predict a woman's cycle length in days (regression).")

        user_cycle = {}
        for feat in m3_features:
            user_cycle[feat] = st.number_input(f"{feat}", value=0.0)

        if st.button("Predict Cycle Length"):
            X = pd.DataFrame([user_cycle])
            X_scaled = m3_scaler.transform(X)
            pred = m3_model.predict(X_scaled)[0]

            st.success(f"### 📏 Predicted Cycle Length: **{pred:.1f} days**")
     
        
