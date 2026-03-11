import streamlit as st

def show():
    st.header("Data Cleaning & Imputation Process")
    
    # Dataset selection
    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Kaggle Menstrual Cycle", "Hormones + Symptoms", "Heart Rate Merged Dataset"]
    )
    
    # ==================== KAGGLE DATASET ====================
    if dataset == "Kaggle Menstrual Cycle":
        tabs = st.tabs([
            "Overview", 
            "Phase 1: Demographics", 
            "Phase 2: Cycle Variables",
            "Validation & Quality Checks"
        ])
        
        with tabs[0]:
            st.subheader("Multi-Stage Hybrid Imputation Pipeline")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Approach:** Hierarchical (Simple → Complex)")
                st.metric("Total Phases", "2")
            with col2:
                st.success("**Outcome:** Zero missing values")
                st.metric("Imputation Methods", "3+")
            
            st.markdown("---")
            st.markdown("""
            ### Key Design Principles
            
            1. **Context-Aware**: Uses participant history before population patterns
            2. **Biologically Informed**: Preserves BMI distribution, enforces physiological constraints
            3. **Complete Coverage**: Guarantees zero missing values with fallback mechanisms
            4. **Validation**: Includes distribution comparison and logical relationship checks
            """)
            
            st.markdown("---")
            st.markdown("### Imputation Flow")
            st.code("""
Phase 1: Demographics
├── Step 1: Within-participant median (Weight, Height, Age, etc.)
├── Step 2: MICE for Weight (preserves BMI distribution)
└── Step 3: MICE for remaining variables

Phase 2: Cycle Variables
├── Step 0: MensesScore filled with 0
├── Step 1: MICE with 10 iterations
├── Step 2: Post-processing (rounding, constraints)
└── Step 3: Fallback (median/mode)
            """, language="text")
        
        with tabs[1]:
            st.subheader("Phase 1: Demographic & Basic Variables")
            
            st.markdown("### 🔹 Step 1: Group-Based Imputation")
            st.info("**Method:** Within-participant median filling")
            
            with st.expander("📝 Details", expanded=True):
                st.markdown("""
                **Variables Targeted:**
                - Weight
                - Height
                - Age
                - Schoolyears
                - MeanCycleLength
                - MeanMensesLength
                
                **Logic:**  
                Fill missing values using each participant's own median across their cycles.
                This preserves individual patterns before applying population-level methods.
                """)
                
                st.code("""
# Example pseudocode
for participant in participants:
    participant_data = data[data['id'] == participant]
    for variable in demographic_variables:
        median_value = participant_data[variable].median()
        participant_data[variable].fillna(median_value, inplace=True)
                """, language="python")
            
            st.markdown("---")
            st.markdown("### 🔹 Step 2: Targeted Weight Imputation")
            st.warning("**Method:** MICE (Multiple Imputation by Chained Equations)")
            
            with st.expander("Details", expanded=True):
                st.markdown("""
                **Special Feature:** BMI Distribution Preservation
                
                1. Calculate median BMI from original data
                2. Impute Weight using Age & Height as predictors
                3. Apply log-normal rescaling to maintain BMI distribution shape
                4. Clip extreme outliers (1st-99th percentile)
                
                **Why This Matters:**  
                Preserving BMI distribution ensures that imputed weights are biologically 
                plausible and maintain the natural variation in the population.
                """)
                
                st.code("""
# Preserve BMI distribution
original_bmi_median = calculate_median_bmi(original_data)

# MICE imputation for Weight
mice_imputer = IterativeImputer()
imputed_weight = mice_imputer.fit_transform(data[['Weight', 'Age', 'Height']])

# Rescale to preserve BMI distribution
scaling_factor = original_bmi_median / new_bmi_median
imputed_weight = imputed_weight * scaling_factor

# Clip outliers
imputed_weight = clip(imputed_weight, percentile_1, percentile_99)
                """, language="python")
            
            st.markdown("---")
            st.markdown("### 🔹 Step 3: Advanced Imputation")
            st.success("**Method:** MICE for remaining variables")
            
            with st.expander("Details", expanded=True):
                st.markdown("""
                **Scope:**  
                Any columns still missing after group imputation (excluding Weight, 
                which was handled in Step 2)
                
                **Result:**  
                BMI is recalculated from imputed Weight and Height
                
                **Algorithm:**  
                MICE iteratively models each variable with missing values as a function 
                of other variables in the dataset, creating plausible values.
                """)
        
        with tabs[2]:
            st.subheader("Phase 2: Cycle-Specific Variables")
            
            st.markdown("### Step 0: MensesScore Handling")
            st.info("**All MensesScoreDay variables filled with 0** (not imputed)")
            
            with st.expander("Rationale"):
                st.markdown("""
                Missing MensesScore likely means no bleeding occurred on that day.
                Therefore, filling with 0 is more appropriate than imputation.
                """)
            
            st.markdown("---")
            st.markdown("### Step 1: Iterative Imputation (MICE)")
            st.warning("**Configuration:** 10 iterations max, BayesianRidge estimator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Variables (9 total):**")
                st.markdown("""
                - LengthofMenses
                - LengthofLutealPhase
                - MeanMensesLength
                - MeanBleedingIntensity
                - TotalNumberofHighDays
                - TotalMensesScore
                - TotalDaysofFertility
                - TotalFertilityFormula
                - Breastfeeding
                """)
            
            with col2:
                st.markdown("**Predictors Used:**")
                st.markdown("""
                - Demographics (Age, Weight, BMI, Schoolyears)
                - Cycle characteristics (CycleNumber, LengthofCycle)
                - EstimatedDayofOvulation
                - Categorical (ReproductiveCategory)
                - Complete MensesScore columns
                """)
            
            with st.expander("Technical Details"):
                st.code("""
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42,
    imputation_order='ascending'  # least to most missing
)

imputed_data = imputer.fit_transform(cycle_variables)
                """, language="python")
            
            st.markdown("---")
            st.markdown("### 🔹 Step 2: Post-Processing")
            st.success("**Ensure biological plausibility**")
            
            with st.expander("Operations Performed", expanded=True):
                st.markdown("""
                1. **Round categorical variables** to integers
                2. **Round count variables** and clip to ≥0
                3. **Enforce logical constraints:**
                   - `LengthofMenses ≤ LengthofCycle`
                   - `LengthofLutealPhase ≤ LengthofCycle`
                
                These constraints prevent biologically impossible values.
                """)
                
                st.code("""
# Round and clip
df['TotalNumberofHighDays'] = df['TotalNumberofHighDays'].round().clip(lower=0)

# Enforce constraints
df['LengthofMenses'] = df[['LengthofMenses', 'LengthofCycle']].min(axis=1)
df['LengthofLutealPhase'] = df[['LengthofLutealPhase', 'LengthofCycle']].min(axis=1)
                """, language="python")
            
            st.markdown("---")
            st.markdown("### 🔹 Step 3: Fallback Safety Net")
            st.error("**Last Resort:** Median (numeric) or Mode (categorical)")
            
            with st.expander("Why This Matters"):
                st.markdown("""
                If any NaN values remain after all previous steps, this guarantees 
                100% completeness by filling with simple statistics.
                
                This is rarely triggered but ensures the pipeline never fails.
                """)
        
        with tabs[3]:
            st.subheader("Validation & Quality Checks")
            
            st.markdown("### Distribution Comparison")
            st.info("Compare original vs. imputed distributions to ensure consistency")
            
            with st.expander("View Validation Steps"):
                st.markdown("""
                1. **Completeness Check**: Verify zero missing values
                2. **Distribution Plots**: Compare before/after histograms
                3. **Correlation Preservation**: Check key relationships maintained
                4. **Constraint Validation**: Verify all logical rules hold
                5. **Outlier Detection**: Flag extreme imputed values
                """)
            
            st.markdown("---")
            st.markdown("### Key Relationship Analysis")
            st.success("**Example:** MeanCycleLength vs. EstimatedDayofOvulation")
            
            st.markdown("""
            The final validation examines the relationship between `MeanCycleLength` 
            and `EstimatedDayofOvulation` using the fully imputed dataset, showing 
            how these menstrual variables correlate after the imputation process.
            
            This helps verify that imputation preserved biological relationships.
            """)
    
    # ==================== HORMONES + SYMPTOMS ====================
    elif dataset == "Hormones + Symptoms":
        tabs = st.tabs([
            "Overview",
            "Encoding",
            "Preparation",
            "Imputation"
        ])
        
        with tabs[0]:
            st.subheader("Hormones + Symptoms Imputation Pipeline")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", "Per-User MICE")
            with col2:
                st.metric("Estimator", "Random Forest")
            with col3:
                st.metric("Max Iterations", "10")
            
            st.markdown("---")
            st.info("""
            **MNAR Detected:** Missing data is not random - when PDG (hormone) was measured,
            symptoms were often not recorded. Missingness is related to testing method.
            """)
            
            st.markdown("### Pipeline Flow")
            st.code("""
Step 1: Ordinal Encoding
├── Menstrual phases (1-4)
├── Symptoms (0-5 scale)
└── Flow variables (multi-level)

Step 2: Data Preparation
├── PDG filled with 0
└── Select numeric columns

Step 3: Per-User Imputation
├── Group by participant ID
├── If <3 records → mean imputation
└── Else → Iterative Imputer (Random Forest)

Step 4: Validation
└── Verify completeness
            """, language="text")
        
        with tabs[1]:
            st.subheader("Step 1: Ordinal Encoding")
            
            st.markdown("### Menstrual Phase Encoding")
            st.code("""
Follicular = 1
Fertility  = 2
Luteal     = 3
Menstrual  = 4
            """, language="text")
            
            st.markdown("---")
            st.markdown("### Symptom & Lifestyle Variables")
            st.info("Variables: appetite, exercise, headaches, cramps, sore breasts, fatigue, sleep, mood swings, stress, food cravings, indigestion, bloating")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("""
                **Encoding Scale:**
                - 0 = Not at all
                - 1 = Very Low
                - 2 = Low
                - 3 = Moderate
                - 4 = High
                - 5 = Very High
                """)
            
            with col2:
                st.code("""
# Example encoding
symptom_mapping = {
    'Not at all': 0,
    'Very Low': 1,
    'Low': 2,
    'Moderate': 3,
    'High': 4,
    'Very High': 5
}

df['headaches_encoded'] = df['headaches'].map(symptom_mapping)
                """, language="python")
            
            st.markdown("---")
            st.markdown("### Flow Variables")
            st.success("Flow volume and color use extended ordinal scales")
            
            st.markdown("""
            Values increase from lightest/mildest to heaviest/most intense category.
            Each original column is transformed into a corresponding `_encoded` column.
            """)
        
        with tabs[2]:
            st.subheader("Step 2: Data Preparation")
            
            st.warning("""
            **MNAR Observed:** Days when PDG was measured often lack symptom data.
            Missingness is related to the hormone testing method.
            """)
            
            st.markdown("### Preparation Steps")
            
            with st.expander("1. Participant Identification", expanded=True):
                st.markdown("""
                Each participant is identified by **`id`** field.
                This allows for group-wise (per-user) imputation.
                """)
            
            with st.expander("2. PDG Hormone Marker", expanded=True):
                st.markdown("""
                Missing values in the hormone marker `pdg` are **filled with 0**.
                
                **Rationale:** Zero PDG indicates hormone was not detected/measured,
                which is meaningful information rather than missing data.
                """)
                
                st.code("""
df['pdg'] = df['pdg'].fillna(0)
                """, language="python")
            
            with st.expander("3. Column Selection", expanded=True):
                st.markdown("""
                Only **numeric columns** (excluding `pdg`) are selected for imputation.
                
                This focuses the imputation process on variables that benefit from
                sophisticated methods while preserving the PDG zeros.
                """)
                
                st.code("""
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputation_cols = [col for col in numeric_cols if col != 'pdg']
                """, language="python")
        
        with tabs[3]:
            st.subheader("Step 3: Per-User Imputation")
            
            st.info("""
            **Strategy:** Impute within each participant to maintain individual consistency.
            Different participants may have different baseline levels for symptoms.
            """)
            
            st.markdown("### Decision Tree")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Few Records (<3)")
                st.code("""
Method: Mean Imputation

participant_mean = user_data.mean()
user_data.fillna(participant_mean)
                """, language="python")
                st.caption("Simple and stable for limited data")
            
            with col2:
                st.markdown("#### Sufficient Records (≥3)")
                st.code("""
Method: Iterative Imputer
Estimator: Random Forest

imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=20,
        max_depth=6
    ),
    max_iter=10
)
                """, language="python")
                st.caption("Captures non-linear relationships")
            
            st.markdown("---")
            st.markdown("### Why Random Forest?")
            
            with st.expander("Advantages", expanded=True):
                st.markdown("""
                1. **Non-linear relationships**: Captures complex interactions between symptoms
                2. **Robust to outliers**: Not sensitive to extreme values
                3. **Feature importance**: Implicitly learns which features predict others
                4. **Efficient**: Optimized parameters balance accuracy and speed
                """)
            
            st.markdown("### Algorithm Parameters")
            
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                st.metric("n_estimators", "20", help="Number of trees in forest")
            with param_col2:
                st.metric("max_depth", "6", help="Maximum tree depth")
            with param_col3:
                st.metric("max_iter", "10", help="Imputation iterations")
            
            st.caption("Parameters optimized for speed while maintaining quality")
            
            st.markdown("---")
            st.markdown("### Complete Implementation")
            
            with st.expander("View Full Code"):
                st.code("""
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def impute_per_user(group):
    if len(group) < 3:
        # Simple mean imputation for small groups
        return group.fillna(group.mean())
    else:
        # Advanced MICE with Random Forest
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=20,
                max_depth=6,
                random_state=42
            ),
            max_iter=10,
            random_state=42
        )
        
        numeric_cols = group.select_dtypes(include=['float', 'int']).columns
        group[numeric_cols] = imputer.fit_transform(group[numeric_cols])
        
        return group

# Apply per-user imputation
final_df = df.groupby('id').apply(impute_per_user).reset_index(drop=True)
                """, language="python")
            
            st.markdown("---")
            st.success("""
             **Final Outcome:**
            - All missing numeric values imputed per user group
            - Ordinal variables encoded for machine learning
            - Logical consistency maintained across all steps
            - Individual participant patterns preserved
            """)
    
    # ==================== HEART RATE MERGED ====================
    else:
        tabs = st.tabs([
            "Overview",
            "HR Loading",
            "Aggregation",
            "Missingness",
            "Imputation",
            "Merging"
        ])
        
        with tabs[0]:
            st.subheader("Heart Rate + Hormones Merged Dataset")
            
            st.warning("""
            **Challenge:** Raw heart rate data is massive (hundreds of thousands of rows 
            per participant). Loading it fully would crash memory.
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", "Time-Series")
            with col2:
                st.metric("Missingness", "15% Simulated")
            with col3:
                st.metric("Memory Savings", "80-90%")
            
            st.markdown("---")
            st.markdown("### Pipeline Overview")
            st.code("""
Step 1: Optimized HR Loading
├── Load in 100k row chunks
├── Filter by participant & date range
└── Memory optimization (float32, uint16)

Step 2: Daily Aggregation
├── Group by participant + day
└── Calculate: mean, min, max, std, count

Step 3: Simulate Missingness (15%)
├── Random dropout
├── Participant bias
├── Night-time missing
└── Multi-day blocks

Step 4: Time-Series Imputation
├── Trend estimation
├── Weekly seasonality
├── Linear interpolation
└── Fallback mean

Step 5: Merge with Hormones
└── Join on: id + day_in_study
            """, language="text")
        
        with tabs[1]:
            st.subheader("Step 1: Heart Rate Loading Process")
            
            st.error("**Problem:** Hundreds of thousands of rows per participant")
            st.success("**Solution:** Chunk-based filtered loading")
            
            st.markdown("### Filtering Before Loading")
            
            with st.expander("Why Filter First?", expanded=True):
                st.markdown("""
                Instead of loading the entire file and then filtering, we:
                1. Load data in **chunks of 100,000 rows**
                2. Filter each chunk immediately
                3. Keep only what matches hormone dataset
                
                **Result:** Only load necessary data, reducing memory by 80-90%
                """)
            
            st.markdown("### Filtering Criteria")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Participants:**
                - Only IDs present in hormone dataset
                - Ignore participants without hormone data
                """)
            with col2:
                st.markdown("""
                **Date Range:**
                - Only days within hormone dataset dates
                - Ignore data outside study period
                """)
            
            st.markdown("---")
            st.markdown("### Processing Steps")
            
            with st.expander("View Implementation", expanded=True):
                st.code("""
# Detect correct HR file name
hr_file = detect_file(['bpm', 'heart_rate', 'heartrate'])

# Get participants and date range from hormone data
valid_ids = hormone_df['id'].unique()
min_day = hormone_df['day_in_study'].min()
max_day = hormone_df['day_in_study'].max()

# Load and filter in chunks
chunks = []
for chunk in pd.read_csv(hr_file, chunksize=100_000):
    # Filter by participant
    chunk = chunk[chunk['id'].isin(valid_ids)]
    
    # Filter by date range
    chunk = chunk[
        (chunk['day_in_study'] >= min_day) & 
        (chunk['day_in_study'] <= max_day)
    ]
    
    # Rename column if needed
    if 'bpm' in chunk.columns:
        chunk.rename(columns={'bpm': 'heart_rate'}, inplace=True)
    
    chunks.append(chunk)

hr_df = pd.concat(chunks, ignore_index=True)
                """, language="python")
            
            st.markdown("---")
            st.markdown("### Memory Optimization")
            
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                st.markdown("**Original Types:**")
                st.code("""
float64  (8 bytes)
int64    (8 bytes)
                """, language="text")
            
            with opt_col2:
                st.markdown("**Optimized Types:**")
                st.code("""
float32  (4 bytes) → 50% savings
uint16   (2 bytes) → 75% savings
int32    (4 bytes) → 50% savings
                """, language="text")
            
            st.code("""
# Apply optimization
hr_df['heart_rate'] = hr_df['heart_rate'].astype('float32')
hr_df['day_in_study'] = hr_df['day_in_study'].astype('uint16')
hr_df['id'] = hr_df['id'].astype('int32')
            """, language="python")
        
        with tabs[2]:
            st.subheader("Step 2: Aggregating Heart Rate (Daily)")
            
            st.info("Original HR dataset may have many readings per day (every minute)")
            
            st.markdown("### Aggregation Strategy")
            st.success("Aggregate per participant, per day into summary statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Metrics Calculated:**")
                st.markdown("""
                - `hr_mean` - Average HR
                - `hr_min` - Minimum HR
                - `hr_max` - Maximum HR
                - `hr_std` - Standard deviation
                - `hr_count` - Number of samples
                """)
            
            with col2:
                st.markdown("**Benefits:**")
                st.markdown("""
                - Reduces noise
                - Removes random fluctuations
                - Decreases data volume
                - Aligns with daily hormones
                - Stable daily summaries
                """)
            
            st.markdown("---")
            st.markdown("### Implementation")
            
            with st.expander("View Aggregation Code", expanded=True):
                st.code("""
# Aggregate heart rate by participant and day
hr_daily = hr_df.groupby(['id', 'day_in_study']).agg({
    'heart_rate': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()

# Flatten column names
hr_daily.columns = [
    'id', 'day_in_study', 
    'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_count'
]

# Example result
# id | day_in_study | hr_mean | hr_std | hr_min | hr_max | hr_count
# 1  | 5           | 72.3    | 8.5    | 58     | 95     | 1423
                """, language="python")
            
            st.markdown("### Example Transformation")
            
            st.markdown("**Before Aggregation:**")
            st.code("""
id | day_in_study | timestamp | heart_rate
1  | 5           | 08:00     | 72
1  | 5           | 08:01     | 73
1  | 5           | 08:02     | 71
... (1,440 rows for one day)
            """, language="text")
            
            st.markdown("**After Aggregation:**")
            st.code("""
id | day_in_study | hr_mean | hr_std | hr_min | hr_max | hr_count
1  | 5           | 72.3    | 8.5    | 58     | 95     | 1423
            """, language="text")
        
        with tabs[3]:
            st.subheader("Step 3: Introducing Realistic Missingness")
            
            st.warning("**Purpose:** Simulate real-world missing data patterns (15% total)")
            
            st.markdown("### Missingness Patterns")
            
            pattern_tabs = st.tabs([
                "Random",
                "Participant Bias",
                "Time-Based",
                "Block Missing"
            ])
            
            with pattern_tabs[0]:
                st.markdown("#### Random Missing")
                st.info("Simulates: General device issues, random sensor failures")
                st.code("""
# Random 5% missingness
random_mask = np.random.random(len(hr_daily)) < 0.05
hr_daily.loc[random_mask, 'hr_mean'] = np.nan
                """, language="python")
            
            with pattern_tabs[1]:
                st.markdown("#### Participant-Biased Missing")
                st.info("Simulates: Some participants forget to wear device more often")
                st.code("""
# Select 20% of participants
biased_participants = np.random.choice(
    hr_daily['id'].unique(), 
    size=int(0.2 * len(hr_daily['id'].unique()))
)

# 15% missingness for these participants
for pid in biased_participants:
    mask = (hr_daily['id'] == pid) & (np.random.random(len(hr_daily)) < 0.15)
    hr_daily.loc[mask, 'hr_mean'] = np.nan
                """, language="python")
            
            with pattern_tabs[2]:
                st.markdown("#### Time-Dependent Missing")
                st.info("Simulates: Sensor issues at night, not wearing device")
                st.code("""
# Higher missingness for certain day ranges
# (e.g., days 20-30 had technical issues)
night_mask = hr_daily['day_in_study'].between(20, 30)
random_night = np.random.random(len(hr_daily)) < 0.10

hr_daily.loc[night_mask & random_night, 'hr_mean'] = np.nan
                """, language="python")
            
            with pattern_tabs[3]:
                st.markdown("#### Block Missing (Multi-Day)")
                st.info("Simulates: Device off for multiple consecutive days")
                st.code("""
# Create 3-5 day missing blocks
for participant in some_participants:
    start_day = np.random.randint(10, 50)
    block_length = np.random.randint(3, 6)
    
    mask = (
        (hr_daily['id'] == participant) &
        (hr_daily['day_in_study'] >= start_day) &
        (hr_daily['day_in_study'] < start_day + block_length)
    )
    
    hr_daily.loc[mask, 'hr_mean'] = np.nan
                """, language="python")
            
            st.markdown("---")
            st.success("**Result:** Realistic missing data that mimics real Fitbit/MCPhases behavior")
        
        with tabs[4]:
            st.subheader("Step 4: Time-Series Imputation")
            
            st.info("Advanced multi-step algorithm for smooth, realistic HR curves")
            
            imputation_steps = st.tabs([
                "Trend",
                "Seasonality", 
                "Interpolation",
                "Fallback"
            ])
            
            with imputation_steps[0]:
                st.markdown("### Trend Estimation")
                st.markdown("""
                Captures long-term HR changes over the study period.
                
                **Method:** Rolling window average (7-14 days)
                """)
                
                st.code("""
# Calculate trend using rolling mean
hr_daily['hr_trend'] = (
    hr_daily.groupby('id')['hr_mean']
    .transform(lambda x: x.rolling(window=7, min_periods=1, center=True).mean())
)
                """, language="python")
                
                st.info("This smooths out day-to-day variations to reveal underlying patterns")
            
            with imputation_steps[1]:
                st.markdown("### Weekly Seasonality")
                st.markdown("""
                Captures biological rhythms (sleep/wake cycles, weekly patterns).
                
                **Method:** Day of week effect (day_in_study % 7)
                """)
                
                st.code("""
# Calculate weekly seasonality
hr_daily['day_of_week'] = hr_daily['day_in_study'] % 7
weekly_pattern = hr_daily.groupby(['id', 'day_of_week'])['hr_mean'].transform('mean')
hr_daily['hr_seasonal'] = weekly_pattern
                """, language="python")
                
                st.success("People often have consistent patterns on the same day of the week")
            
            with imputation_steps[2]:
                st.markdown("### Linear Interpolation")
                st.markdown("""
                Fills gaps between known values with smooth transitions.
                
                **Method:** Linear interpolation per participant
                """)
                
                st.code("""
# Interpolate missing values
hr_daily['hr_interpolated'] = (
    hr_daily.groupby('id')['hr_mean']
    .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
)
                """, language="python")
                
                st.info("Creates smooth curves between data points rather than sharp jumps")
            
            with imputation_steps[3]:
                st.markdown("### Final Fallback Mean")
                st.markdown("""
                Only used if a participant has too few HR records for other methods.
                
                **Method:** Participant mean or global mean
                """)
                
                st.code("""
# Final fallback for any remaining missing values
for participant_id in hr_daily['id'].unique():
    mask = hr_daily['id'] == participant_id
    participant_data = hr_daily[mask]
    
    if participant_data['hr_mean'].isna().all():
        # Use global mean if participant has no HR data
        hr_daily.loc[mask, 'hr_mean'] = hr_daily['hr_mean'].mean()
    else:
        # Use participant mean
        hr_daily.loc[mask, 'hr_mean'] = (
            hr_daily.loc[mask, 'hr_mean'].fillna(participant_data['hr_mean'].mean())
        )
                """, language="python")
                
                st.warning("This is rarely needed but ensures 100% completeness")
            
            st.markdown("---")
            st.success("""
            **Result:** Smooth, realistic HR curves that maintain:
            - Individual participant patterns
            - Biological rhythms
            - Temporal continuity
            - No artificial jumps or outliers
            """)
        
        with tabs[5]:
            st.subheader("Step 5: Merging Heart Rate with Hormones")
            
            st.info("After cleaning HR data, merge it with hormone/symptom dataset")
            
            st.markdown("### Merge Key")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.code("""
Merge on:
- id (participant)
- day_in_study (day)
                """, language="text")
            
            with col2:
                st.code("""
Type: Inner join
(keeps only matching records)
                """, language="text")
            
            st.markdown("---")
            st.markdown("### Implementation")
            
            with st.expander("View Merge Code", expanded=True):
                st.code("""
# Merge HR with hormones
merged_df = pd.merge(
    hormone_symptom_df,
    hr_daily,
    on=['id', 'day_in_study'],
    how='inner'
)

# Verify merge
print(f"Hormone records: {len(hormone_symptom_df)}")
print(f"HR records: {len(hr_daily)}")
print(f"Merged records: {len(merged_df)}")
                """, language="python")
            
            st.markdown("---")
            st.markdown("### Merged Dataset Contains")
            
            content_tabs = st.tabs([
                "HR Metrics",
                "Hormones",
                "Symptoms",
                "Cycle Info"
            ])
            
            with content_tabs[0]:
                st.markdown("""
                **Heart Rate Variables:**
                - hr_mean
                - hr_min
                - hr_max
                - hr_std
                - hr_count
                """)
            
            with content_tabs[1]:
                st.markdown("""
                **Hormone Measurements:**
                - pdg (progesterone metabolite)
                - estrogen markers
                - Other hormonal indicators
                """)
            
            with content_tabs[2]:
                st.markdown("""
                **Self-Reported Symptoms:**
                - headaches, cramps, fatigue
                - mood swings, stress
                - sleep quality
                - appetite, food cravings
                - And more...
                """)
            
            with content_tabs[3]:
                st.markdown("""
                **Cycle Information:**
                - Menstrual phase
                - Day in cycle
                - Flow characteristics
                - Ovulation indicators
                """)
            
            st.markdown("---")
            st.markdown("### Use Cases")
            
            use_col1, use_col2 = st.columns(2)
            
            with use_col1:
                st.success("""
                **Machine Learning:**
                - Heart Rate Prediction
                - Phase Classification
                - Symptom Prediction
                """)
            
            with use_col2:
                st.success("""
                **Analysis:**
                - Feature Engineering
                - Correlation Studies
                - Pattern Discovery
                """)
            
            st.markdown("---")
            st.markdown("### Output File")
            
            st.code("""
final_merged_hr_hormones.csv
            """, language="text")
            
            st.success("""
             **Complete Dataset Ready For:**
            - Machine learning model training
            - Statistical analysis
            - Visualization
            - Research insights
            """)
