import streamlit as st

def show():
    st.header("🧹 Data Cleaning Summary")
    
    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Kaggle", "Hormones+symptoms", "heart rate and hormones symptoms merged"]
    )
    
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
        
    elif dataset == "Hormones+symptoms":
        st.markdown("""
### 1. Ordinal Encoding

- Each variable was mapped according to its natural order or intensity. 
- For example, the menstrual phase was encoded as: Follicular = 1, Fertility = 2, Luteal = 3, Menstrual = 4. 
- Symptom and lifestyle variables such as appetite, exercise level, headaches, cramps, sore breasts, fatigue, sleep issues, mood swings, stress, food cravings, indigestion, and bloating were encoded using six levels:

0 = "Not at all"
1 = "Very Low" or "Very Low/Little"
2 = "Low"
3 = "Moderate"
4 = "High"
5 = "Very High"

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
