import streamlit as st
import pandas as pd

def show():
    st.title("Machine Learning Models")
    st.markdown("---")
    
    st.markdown("""
    This dashboard uses three specialized machine learning models to provide predictions 
    and insights about menstrual cycle patterns. Each model is trained on real data and 
    optimized for accuracy while maintaining ease of use.
    """)
    
    tab1, tab2 = st.tabs(["Model 1: Cycle Phase", "Model 2: Cycle Length"])
    
    # ====================================================================
    # MODEL 1: PHASE PREDICTION
    # ====================================================================
    with tab1:
        st.header("Model 2: Cycle Phase Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Overview")
            st.markdown("""
            Identifies which phase of your menstrual cycle you're currently in based on 
            physiological measurements and hormone levels.
            """)
            
            st.subheader("Purpose")
            st.markdown("""
            - Accurately identify current cycle phase
            - Understand phase-specific symptoms and patterns
            - Plan activities based on cycle phase
            - Track cycle regularity over time
            """)
            
            st.subheader("Model Details")
            st.markdown("""
            - **Algorithm**: XGBoost Classifier
            - **Parameters**: 180 trees, learning rate 0.12
            - **Classes**: Follicular, Fertility, Luteal, Menstrual
            - **Optimization**: High accuracy with minimal user inputs
            """)
        
        with col2:
            st.metric("Model Type", "Classification")
            st.metric("Trees", "180")
            st.metric("Classes", "4")
            st.metric("Accuracy", "43%")
        
        st.subheader("Input Features (Simplified)")
        features_m2 = pd.DataFrame({
            'Feature': [
                'cycle_day', 'normalized_cycle_day', 'estrogen', 'pdg',
                'estrogen_z', 'pdg_z', 'hr_mean', 'hr_lag1', 'hr_rolling_7d', 'phase_lag1'
            ],
            'Description': [
                'Day of cycle (1-28)', 'Normalized cycle position',
                'Estrogen level', 'Progesterone level',
                'Z-scored estrogen (standardized)', 'Z-scored progesterone',
                'Current average heart rate', 'Previous day heart rate',
                '7-day rolling average heart rate', 'Previous phase (temporal context)'
            ],
            'User Provides?': [
                '✅ Yes', '❌ Auto-calculated', '✅ Yes', '✅ Yes',
                '❌ Auto-calculated', '❌ Auto-calculated', '✅ Yes',
                '❌ Auto-calculated', '❌ Auto-calculated', '❌ Auto-calculated'
            ]
        })
        st.dataframe(features_m2, use_container_width=True, hide_index=True)
        
        st.success("**User-Friendly**: Only 4 inputs needed! (cycle_day, estrogen, pdg, hr_mean)")
        
        st.subheader("Predicted Phases")
        phases = pd.DataFrame({
            'Phase': ['Follicular', 'Fertility', 'Luteal', 'Menstrual'],
            'Timing': ['Days 1-13', 'Days 14-17', 'Days 18-28', 'Days 1-5'],
            'Characteristics': [
                'Estrogen rising, preparing for ovulation',
                'Peak estrogen and LH surge, ovulation occurs',
                'Progesterone dominant, preparing for potential pregnancy',
                'Hormone levels drop, menstruation begins'
            ]
        })
        st.dataframe(phases, use_container_width=True, hide_index=True)
    
    # ====================================================================
    # MODEL 2: CYCLE LENGTH PREDICTION
    # ====================================================================
    with tab2:
        st.header("Model 3: Cycle Length Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Overview")
            st.markdown("""
            Predicts the total length of your menstrual cycle based on demographic 
            information and menstrual characteristics. Helps you anticipate when 
            your next period will begin.
            """)
            
            st.subheader("Purpose")
            st.markdown("""
            - Predict next period start date
            - Understand cycle regularity patterns
            - Identify factors affecting cycle length
            - Plan ahead for important events
            """)
            
            st.subheader("Model Details")
            st.markdown("""
            - **Algorithm**: Random Forest Regressor
            - **Parameters**: 200 trees, max depth 10
            - **Training Data**: Kaggle menstrual cycle dataset
            - **Performance**: MAE of 0.45 days, R² of 0.88
            """)
        
        with col2:
            st.metric("Model Type", "Regression")
            st.metric("Trees", "200")
            st.metric("Max Depth", "10")
            st.metric("MAE", "0.45 days")
            st.metric("R²", "0.88")
        
        st.subheader("Input Features")
        features_m3 = pd.DataFrame({
            'Feature': [
                'Age', 'BMI', 'Height', 'Weight',
                'MeanBleedingIntensity', 'TotalNumberofHighDays', 'TotalMensesScore',
                'MensesScoreDay1-5', 'EstimatedDayofOvulation',
                'LengthofLutealPhase', 'LengthofMenses'
            ],
            'Description': [
                'Age in years', 'Body Mass Index', 'Height in cm', 'Weight in kg',
                'Average bleeding intensity across cycle',
                'Number of high fertility days',
                'Sum of all menstruation scores',
                'Bleeding intensity for first 5 days',
                'Estimated day ovulation occurred',
                'Length of the luteal phase in days',
                'Number of days of menstruation'
            ],
            'Why Included?': [
                'Age affects cycle regularity',
                'BMI correlates with hormonal balance',
                'Physical characteristics baseline',
                'Physical characteristics baseline',
                'Bleeding patterns indicate cycle health',
                'Fertility window indicates ovulation timing',
                'Menstrual characteristics',
                'Early menstrual pattern',
                'Strongly linked to overall cycle length',
                'Luteal phase length directly impacts cycle duration',
                'Menses length contributes to total cycle length'
            ]
        })
        st.dataframe(features_m3, use_container_width=True, hide_index=True)

        st.success("**High Performance**: MAE of 0.45 days — predictions within half a day on average.")

        st.subheader("Typical Cycle Lengths")
        st.markdown("""
        - **Short**: 21-24 days
        - **Normal**: 25-30 days (most common)
        - **Long**: 31-35 days
        - **Irregular**: Varies by >7 days between cycles
        """)
    
    # ====================================================================
    # COMPARISON SECTION
    # ====================================================================
    st.markdown("---")
    st.header("Model Comparison")
    
    comparison_df = pd.DataFrame({
        'Model': ['Model 2: Phase', 'Model 3: Cycle Length'],
        'Type': ['Classification', 'Regression'],
        'Algorithm': ['XGBoost', 'Random Forest'],
        'Input Complexity': ['Low (4 user inputs)', 'Low (11 features)'],
        'Output': ['Phase name', 'Cycle length (days)'],
        'Performance': ['43% accuracy', 'MAE 0.45 days, R² 0.88'],
        'Use Case': ['Phase tracking', 'Period prediction']
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ====================================================================
    # HOW TO USE
    # ====================================================================
    st.markdown("---")
    st.header("How to Use These Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Gather Data")
        st.markdown("""
        - Track your cycle day
        - Measure hormone levels (if available)
        - Monitor heart rate with smartwatch
        - Note demographic info
        """)
    
    with col2:
        st.subheader("Make Predictions")
        st.markdown("""
        - Navigate to Predictions page
        - Enter required values
        - Click "Predict"
        - View results instantly
        """)
    
    with col3:
        st.subheader("Understand Results")
        st.markdown("""
        - Review prediction confidence
        - Compare with historical data
        - Track patterns over time
        - Adjust lifestyle accordingly
        """)
    
    # ====================================================================
    # TECHNICAL DETAILS
    # ====================================================================
    st.markdown("---")
    with st.expander("Technical Details & Training Info"):
        st.subheader("Training Process")
        st.markdown("""
        All models were trained using scikit-learn and XGBoost with the following considerations:
        
        **Data Preprocessing:**
        - Missing value imputation using median values
        - Feature scaling with StandardScaler
        - Label encoding for categorical targets
        - Feature engineering for temporal patterns
        
        **Model Selection:**
        - Random Forest for robust regression tasks
        - XGBoost for high-accuracy classification
        - Hyperparameter tuning for optimal performance
        - Cross-validation to prevent overfitting
        
        **Evaluation Metrics:**
        - Regression: MAE 0.45 days, R² 0.88
        - Classification: 43% accuracy across 4 phases
        - All models evaluated on held-out test sets
        
        **Feature Engineering:**
        - Cycle day normalization (0-1 scale)
        - Rolling averages for temporal smoothing
        - Lag features for sequential context
        - Z-score normalization for hormones
        - Log transformations for skewed distributions
        """)
        
        st.subheader("Model Files")
        st.code("""
        models/
        ├── model2_phase_prediction.pkl    # Phase classification model
        ├── model2_scaler.pkl              # Feature scaler
        ├── model2_encoder.pkl             # Label encoder
        ├── model2_features.pkl            # Feature list
        ├── model3_cycle_length.pkl        # Cycle length model
        ├── model3_scaler.pkl              # Feature scaler
        └── model3_features.pkl            # Feature list
        """)
