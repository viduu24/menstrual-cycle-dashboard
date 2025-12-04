import streamlit as st
import os
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from PIL import Image
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

def decode_phase(phase_encoded):
    phase_map = {1: 'Follicular', 2: 'Fertility', 3: 'Luteal', 4: 'Menstruall'}
    return phase_map.get(phase_encoded, 'Unknown')

# --- Sidebar Navigation ---
st.sidebar.title("🩸 Menstrual Cycle Dashboard")
page = st.sidebar.radio("Go to", ["README", "Data Description", "Missingness","Cleaning Process", "Information","Graphs"])

# --- Page 1: README ---
if page == "README":
    st.title("📘 Menstrual Cycle Analysis Dashboard")
    st.markdown("""
    This app explores menstrual cycle data using two datasets. It aims to clean, analyse menstrual cycle datasets to understand how 
                different biological, lifestyle factors influence menstrual patterns. This app's primary audience are people who know 
                nothing about the menstrual cycle and hope to learn about it. 
    
    **Goals:**
    - Summarize data cleaning and analysis
    - Visualize cycle patterns
    - Explore factors affecting flow and length
    
    """)

# --- Page 2: Data Description ---
elif page == "Data Description":
    st.header("📄 Dataset Overview")
    dataset = st.selectbox("Select a dataset to view:", ["Period 1", "Period 2"])
    df = period_1 if dataset == "Period 1" else period_2

    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # --- Add explanatory text ---
    if dataset == "Period 1":
        st.markdown("""
        **About Period 1 Dataset:**  
        - Obtained from Kaggle  
        - Contains detailed information about individual menstrual cycles recorded by users.  
        - Key columns include:  
            - `ClientID`: Unique user identifier  
            - `LengthofCycle`: Duration of cycle (days)  
            - `LengthofLutealPhase`: Luteal phase duration (days)  
            - `BleedingIntensity`: Categorical flow strength  
        """)
    else:
        st.markdown("""
        **About Period 2 Dataset:**  
        - Obtained from PhysioNet  
        - Includes contextual lifestyle factors.  
        - Key columns include:  
            - `Exercise`  
            - `Cramps`, `Bloating`, `Indigestion`, `Appetite`, `MoodSwings`, `Fatigue`  
            - `StressLevel`, `SleepQuality`  
        """)

    # --- Statistical Summaries ---
    st.subheader(" Statistical Summaries of Numerical Columns")
    st.write(df.describe())

    # --- Categorical summaries ---
    st.subheader(" Categorical Feature Summary")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            st.write(df[col].value_counts())
    else:
        st.write("No categorical columns found in this dataset.")

   
# --- Page 3: Missingness ---
elif page=="Missingness":
    dataset = st.selectbox("Select a dataset to view:", ["Period 1", "Period 2"])
    if dataset =="Period 1":
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, "missing_values_heatmap.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap", use_column_width=True)
    else:
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, "missing_values_heatmap1.png")
        img = Image.open(img_path)
        st.image(img, caption="Missing Values Heatmap", use_column_width=True)
# --- Page 4: Cleaning Process ---
elif page == "Cleaning Process":
    st.header("🧹 Data Cleaning Summary")
    dataset = st.selectbox("Select a dataset to view:", ["Period 1", "Period 2"])
    if dataset == "Period 1":
        
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
        
        
        
    else:
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
    dataset = st.selectbox("Select a dataset to view:", ["Period 1", "Period 2"])
    if dataset == "Period 1":
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
    else:
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
