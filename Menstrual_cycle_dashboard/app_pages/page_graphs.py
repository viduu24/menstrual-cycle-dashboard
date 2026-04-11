import streamlit as st
import pandas as pd
import altair as alt
import os
from Menstrual_cycle_dashboard.utils.data_loader import load_data
from Menstrual_cycle_dashboard.utils.visualization import MenstrualCycleVisualizer


def show(period_1, period_2, period_3):
    dataset = st.selectbox(
        "Select a dataset to view:",
        ["Kaggle", "Hormones+symptoms", "Heart rate Hormones symptoms merged"]
    )
    
    if dataset == "Kaggle":
        st.header("Period 1 Data Visualizations")
        
        with st.sidebar:
            st.subheader("Choose Visualization")
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
            age_df = df.groupby("ClientID")["Age"].first().reset_index()
            
            age_chart = alt.Chart(age_df).mark_bar(
                color='#3b82f6',
                opacity=0.7
            ).encode(
                alt.X('Age:Q', bin=alt.Bin(maxbins=20), title='Age'),
                alt.Y('count()', title='Count'),
                tooltip=[
                    alt.Tooltip('Age:Q', bin=alt.Bin(maxbins=20), title='Age'),
                    alt.Tooltip('count()', title='Count')
                ]
            ).properties(
                width=600,
                height=400,
                title='Distribution of Age'
            ).interactive()
            
            st.altair_chart(age_chart, use_container_width=True)
            st.markdown("The above graph shows the age distribution of the dataset.")
            
        # 2️⃣ BMI Distribution
        elif plot_type == "BMI Distribution":
            st.subheader("BMI Distribution")
            bmi_df = df.groupby('ClientID')["BMI"].first().reset_index()
            
            bmi_chart = alt.Chart(bmi_df).mark_bar(
                color='#10b981',
                opacity=0.7
            ).encode(
                alt.X('BMI:Q', bin=alt.Bin(maxbins=20), title='BMI'),
                alt.Y('count()', title='Count'),
                tooltip=[
                    alt.Tooltip('BMI:Q', bin=alt.Bin(maxbins=20), title='BMI'),
                    alt.Tooltip('count()', title='Count')
                ]
            ).properties(
                width=600,
                height=400,
                title='Distribution of BMI'
            ).interactive()
            
            st.altair_chart(bmi_chart, use_container_width=True)
            st.markdown("The above graph shows the BMI distribution of the dataset.")
            
        # 3️⃣ Cycle Length Distribution
        elif plot_type == "Cycle Length Distribution":
            st.subheader("Cycle Length Distribution")
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
            st.markdown("From the bar chart it can be implied that in most women, the cycle length is about 26-32 days.")
            
        # 4️⃣ Age vs Cycle Length (Box Plot)
        elif plot_type == 'Age vs Cycle Length (Box Plot)':
            st.subheader("Cycle Length by Age Group")
            df_age = df.dropna(subset=['Age', 'LengthofCycle']).copy()
            df_age['AgeGroup'] = pd.cut(
                df_age['Age'],
                bins=[17, 25, 30, 35, 40, 45],
                labels=['18-25', '26-30', '31-35', '36-40', '41-45']
            )
            
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
            st.markdown("""
In this box plot, we see quite a few outliers, but it can be observed that after the age of 35 there is a slight change in the median of the cycle of days.
The presence of outliers show that the gap between subsequent periods can vary quite a bit for different women. So if your cycle is occasionally 
not at the expected time, don't worry!
            """)
            
        # 5️⃣ Luteal Phase Length Distribution
        elif plot_type == "Luteal Phase Length Distribution":
            st.subheader("Luteal Phase Length Distribution")
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
            st.markdown("Most women's luteal phase is about 12-15 days.")
            
        # 6️⃣ Average Bleeding Intensity
        elif plot_type == "Average Bleeding Intensity":
            st.subheader("Average Bleeding Intensity Over Menses Days")
            menses_cols = [f'MensesScoreDay{day}' for day in 
                          ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']]
            menses_cols = [col for col in menses_cols if col in df.columns]
            
            if menses_cols:
                menses_avg = df[menses_cols].mean()
                menses_data = pd.DataFrame({
                    'Day': [f'Day {i+1}' for i in range(len(menses_avg))],
                    'Day_Number': list(range(1, len(menses_avg) + 1)),
                    'Average_Score': menses_avg.values
                })
                
                line_chart = alt.Chart(menses_data).mark_line(
                    color='#ef4444',
                    strokeWidth=3,
                    point=alt.OverlayMarkDef(filled=True, size=100)
                ).encode(
                    x=alt.X('Day:N', title='Day', sort=None),
                    y=alt.Y('Average_Score:Q', title='Average Bleeding Intensity Score'),
                    tooltip=[
                        alt.Tooltip('Day:N', title='Day'),
                        alt.Tooltip('Average_Score:Q', title='Average Score', format='.2f')
                    ]
                ).properties(
                    width=600,
                    height=400,
                    title='Average Bleeding Intensity Over Menses Days'
                ).interactive()
                
                st.altair_chart(line_chart, use_container_width=True)
                st.markdown("""
From this line chart we can tell that the maximum bleeding is on day 2 of period for most women (It is important that you take rest and understand what your body needs). Most women menstruate for up to 6 days.
                """)
            else:
                st.warning("No menses-related columns found for plotting.")
    
    elif dataset == "Hormones+symptoms":
        with st.sidebar:
            st.subheader("Choose Visualization")
            plot_type = st.radio(
                "Select a visualization:",
                [
                    "Estrogen Levels by Cycle Phase",
                    "LH Levels by Cycle Phase",
                    "All Symptoms Across Phases",
                ]
            )
        
        st.title("Cycle Phase Hormone & Symptom Visualizations")
        final_df = period_2.copy()
        
        # 1. Estrogen Levels by Cycle Phase
        if plot_type == "Estrogen Levels by Cycle Phase":
            st.subheader("Estrogen Levels by Cycle Phase")
            
            phase_hormones = final_df.groupby('phase_encoded')['estrogen'].mean().reset_index()
            phase_hormones['phase_name'] = phase_hormones['phase_encoded'].apply(decode_phase)
            
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
            st.markdown("Estrogen levels are highest during the fertility phase and lowest during the menstrual phase.")
        
        # 2. LH Levels by Cycle Phase
        elif plot_type == "LH Levels by Cycle Phase":
            st.subheader("LH Levels by Cycle Phase")
            
            phase_hormones = final_df.groupby('phase_encoded')['lh'].mean().reset_index()
            phase_hormones['phase_name'] = phase_hormones['phase_encoded'].apply(decode_phase)
            
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
            st.markdown("LH levels are highest during the fertility phase and lowest during the luteal phase.")
        
        # 3. All Symptoms Across Phases
        elif plot_type == "All Symptoms Across Phases":
            st.subheader("All Symptoms Across Cycle Phases - Grouped Comparison")
            
            symptom_cols = ['headaches_encoded', 'cramps_encoded', 'sorebreasts_encoded', 
                           'fatigue_encoded', 'sleepissue_encoded', 'moodswing_encoded', 
                           'stress_encoded', 'foodcravings_encoded', 'indigestion_encoded', 
                           'bloating_encoded']
            
            symptom_names = ['Headaches', 'Cramps', 'Sore Breasts', 'Fatigue', 
                            'Sleep Issues', 'Mood Swings', 'Stress', 'Food Cravings', 
                            'Indigestion', 'Bloating']
            
            phase_symptoms = final_df.groupby('phase_encoded')[symptom_cols].mean().reset_index()
            phase_symptoms['phase_name'] = phase_symptoms['phase_encoded'].apply(decode_phase)
            
            symptoms_melted = phase_symptoms.melt(
                id_vars=['phase_encoded', 'phase_name'],
                value_vars=symptom_cols,
                var_name='symptom',
                value_name='score'
            )
            
            symptom_mapping = dict(zip(symptom_cols, symptom_names))
            symptoms_melted['symptom_name'] = symptoms_melted['symptom'].map(symptom_mapping)
            
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
            st.markdown("During the menstrual phase, most symptoms start to or increase (especially cramps)")
    
    else:
        st.header("Merged Dataset Visualizations")
        
        with st.sidebar:
            st.subheader("Choose Visualization (Merged)")
            plot_type = st.radio(
                "Select a visualization:",
                [
                    "Heart Rate by Cycle Phase",
                    "HR & Hormones Over Time",
                    "HR vs Hormones Correlation"
                ]
            )
        
        df_merged = period_3.copy()
        
        # 1️⃣ Heart Rate by Cycle Phase
        if plot_type == "Heart Rate by Cycle Phase":
            st.subheader("Heart Rate Distribution by Cycle Phase")
            
            # Map phase_encoded to phase names
            phase_mapping = {0: 'Follicular', 1: 'Fertility', 2: 'Luteal', 3: 'Menstrual'}
            df_merged['phase_name'] = df_merged['phase_encoded'].map(phase_mapping)
            
            box_plot = alt.Chart(df_merged).mark_boxplot(
                size=60,
                opacity=0.7
            ).encode(
                x=alt.X('phase_name:N', 
                       title='Cycle Phase',
                       sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('hr_mean:Q', 
                       title='Heart Rate (bpm)',
                       scale=alt.Scale(zero=False)),
                color=alt.Color('phase_name:N', 
                              scale=alt.Scale(scheme='category10'),
                              legend=None),
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('min(hr_mean):Q', title='Min', format='.1f'),
                    alt.Tooltip('q1(hr_mean):Q', title='Q1', format='.1f'),
                    alt.Tooltip('median(hr_mean):Q', title='Median', format='.1f'),
                    alt.Tooltip('q3(hr_mean):Q', title='Q3', format='.1f'),
                    alt.Tooltip('max(hr_mean):Q', title='Max', format='.1f')
                ]
            ).properties(
                width=700,
                height=400
            ).interactive()
            
            st.altair_chart(box_plot, use_container_width=True)
            st.markdown("Heart rate patterns vary across menstrual cycle phases, with notable changes during the fertility phase.")
        
        # 2️⃣ HR & Hormones Over Time
        elif plot_type == "HR & Hormones Over Time":
            st.subheader("Heart Rate & Hormones Time Series")
            
            # Select a participant
            available_ids = df_merged['id'].unique()
            selected_id = st.selectbox("Select Participant:", available_ids)
            
            df_participant = df_merged[df_merged['id'] == selected_id].sort_values('day_in_study')
            
            # HR Chart
            hr_chart = alt.Chart(df_participant).mark_line(
                color='#ef4444',
                strokeWidth=3,
                point=alt.OverlayMarkDef(filled=True, size=60)
            ).encode(
                x=alt.X('day_in_study:Q', title='Day in Study'),
                y=alt.Y('hr_mean:Q', 
                       title='Heart Rate (bpm)',
                       scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('day_in_study:Q', title='Day'),
                    alt.Tooltip('hr_mean:Q', title='Heart Rate', format='.1f')
                ]
            ).properties(
                width=800,
                height=250,
                title='Heart Rate Over Time'
            )
            
            # Estrogen Chart
            estrogen_chart = alt.Chart(df_participant).mark_line(
                color='#3b82f6',
                strokeWidth=3,
                point=alt.OverlayMarkDef(filled=True, size=60)
            ).encode(
                x=alt.X('day_in_study:Q', title='Day in Study'),
                y=alt.Y('estrogen:Q', 
                       title='Estrogen Level',
                       scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('day_in_study:Q', title='Day'),
                    alt.Tooltip('estrogen:Q', title='Estrogen', format='.2f')
                ]
            ).properties(
                width=800,
                height=250,
                title='Estrogen Levels'
            )
            
            # LH Chart
            lh_chart = alt.Chart(df_participant).mark_line(
                color='#8b5cf6',
                strokeWidth=3,
                point=alt.OverlayMarkDef(filled=True, size=60)
            ).encode(
                x=alt.X('day_in_study:Q', title='Day in Study'),
                y=alt.Y('lh:Q', 
                       title='LH Level',
                       scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('day_in_study:Q', title='Day'),
                    alt.Tooltip('lh:Q', title='LH', format='.2f')
                ]
            ).properties(
                width=800,
                height=250,
                title='LH (Luteinizing Hormone) Levels'
            )
            
            combined = alt.vconcat(hr_chart, estrogen_chart, lh_chart).resolve_scale(x='shared')
            st.altair_chart(combined, use_container_width=True)
            st.markdown("These synchronized time series show how heart rate and hormones fluctuate throughout the menstrual cycle.")
        
        # 3️⃣ Hormone Levels by Phase
        elif plot_type == "Hormone Levels by Phase":
            st.subheader("Average Hormone Levels by Cycle Phase")
            
            # Map phase_encoded to phase names
            phase_mapping = {1: 'Follicular', 2: 'Fertility', 3: 'Luteal', 4: 'Menstrual'}
            df_merged['phase_name'] = df_merged['phase_encoded'].map(phase_mapping)
            
            # Calculate averages
            phase_hormones = df_merged.groupby('phase_name')[['estrogen', 'lh', 'pdg']].mean().reset_index()
            
            # Melt for grouped bar chart
            hormones_melted = phase_hormones.melt(
                id_vars=['phase_name'],
                value_vars=['estrogen', 'lh', 'pdg'],
                var_name='hormone',
                value_name='level'
            )
            
            # Rename hormones
            hormone_names = {'estrogen': 'Estrogen', 'lh': 'LH', 'pdg': 'PDG'}
            hormones_melted['hormone_name'] = hormones_melted['hormone'].map(hormone_names)
            
            chart = alt.Chart(hormones_melted).mark_bar(
                opacity=0.8
            ).encode(
                x=alt.X('phase_name:N', 
                       title='Cycle Phase',
                       sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('level:Q', title='Hormone Level'),
                color=alt.Color('hormone_name:N', 
                              title='Hormone',
                              scale=alt.Scale(scheme='category10')),
                xOffset='hormone_name:N',
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('hormone_name:N', title='Hormone'),
                    alt.Tooltip('level:Q', title='Level', format='.2f')
                ]
            ).properties(
                width=800,
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            st.markdown("Estrogen peaks during fertility phase, while LH surges before ovulation. PDG (progesterone) is highest during luteal phase.")
        
        # 4️⃣ Symptoms by Phase
        elif plot_type == "Symptoms by Phase":
            st.subheader("Common Symptoms Across Cycle Phases")
            
            # Map phase_encoded to phase names
            phase_mapping = {0: 'Follicular', 1: 'Fertility', 2: 'Luteal', 3: 'Menstrual'}
            df_merged['phase_name'] = df_merged['phase_encoded'].map(phase_mapping)
            
            symptom_cols = ['headaches_encoded', 'cramps_encoded', 'fatigue_encoded', 
                           'moodswing_encoded', 'bloating_encoded']
            symptom_names = ['Headaches', 'Cramps', 'Fatigue', 'Mood Swings', 'Bloating']
            
            # Calculate averages
            phase_symptoms = df_merged.groupby('phase_name')[symptom_cols].mean().reset_index()
            
            # Melt data
            symptoms_melted = phase_symptoms.melt(
                id_vars=['phase_name'],
                value_vars=symptom_cols,
                var_name='symptom',
                value_name='score'
            )
            
            symptom_mapping = dict(zip(symptom_cols, symptom_names))
            symptoms_melted['symptom_name'] = symptoms_melted['symptom'].map(symptom_mapping)
            
            # Heatmap
            heatmap = alt.Chart(symptoms_melted).mark_rect().encode(
                x=alt.X('phase_name:N', 
                       title='Cycle Phase',
                       sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('symptom_name:N', title='Symptom'),
                color=alt.Color('score:Q',
                              scale=alt.Scale(scheme='reds'),
                              title='Severity'),
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('symptom_name:N', title='Symptom'),
                    alt.Tooltip('score:Q', title='Average Score', format='.2f')
                ]
            ).properties(
                width=700,
                height=400
            ).interactive()
            
            st.altair_chart(heatmap, use_container_width=True)
            st.markdown("Symptoms like cramps and bloating are typically highest during the menstrual phase, while other symptoms vary throughout the cycle.")
        
        # 5️⃣ Heart Rate Variability
        elif plot_type == "Heart Rate Variability":
            st.subheader("Heart Rate Variability (HRV) Analysis")
            
            # HRV distribution
            hrv_hist = alt.Chart(df_merged).mark_bar(
                color='#0d9488',
                opacity=0.7
            ).encode(
                alt.X('hr_std:Q', 
                     bin=alt.Bin(maxbins=40), 
                     title='Heart Rate Standard Deviation (bpm)'),
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
            
            # Mean line
            mean_hrv = df_merged['hr_std'].mean()
            mean_line = alt.Chart(pd.DataFrame({'mean': [mean_hrv]})).mark_rule(
                color='red',
                strokeWidth=2,
                strokeDash=[5, 5]
            ).encode(
                x='mean:Q'
            )
            
            hrv_chart = (hrv_hist + mean_line).interactive()
            
            # HRV by phase
            phase_mapping = {0: 'Follicular', 1: 'Fertility', 2: 'Luteal', 3: 'Menstrual'}
            df_merged['phase_name'] = df_merged['phase_encoded'].map(phase_mapping)
            
            hrv_violin = alt.Chart(df_merged).mark_boxplot(
                size=50
            ).encode(
                x=alt.X('phase_name:N', 
                       title='Cycle Phase',
                       sort=['Follicular', 'Fertility', 'Luteal', 'Menstrual']),
                y=alt.Y('hr_std:Q', title='HRV (Std Dev)', scale=alt.Scale(zero=False)),
                color=alt.Color('phase_name:N', scale=alt.Scale(scheme='set2'), legend=None),
                tooltip=[
                    alt.Tooltip('phase_name:N', title='Phase'),
                    alt.Tooltip('median(hr_std):Q', title='Median HRV', format='.2f')
                ]
            ).properties(
                width=700,
                height=300,
                title='HRV Distribution by Cycle Phase'
            ).interactive()
            
            combined_hrv = alt.vconcat(hrv_chart, hrv_violin)
            st.altair_chart(combined_hrv, use_container_width=True)
            st.markdown("Higher heart rate variability generally indicates better cardiovascular health and stress resilience. HRV patterns can vary across cycle phases.")
        
        # 6️⃣ HR vs Hormones Correlation
        elif plot_type == "HR vs Hormones Correlation":
            st.subheader("Heart Rate vs Hormone Levels")
            
            # Map phase for color
            phase_mapping = {0: 'Follicular', 1: 'Fertility', 2: 'Luteal', 3: 'Menstrual'}
            df_merged['phase_name'] = df_merged['phase_encoded'].map(phase_mapping)
            df_plot = df_merged.dropna(subset=['phase_encoded'])
            # HR vs Estrogen scatter
            scatter_estrogen = alt.Chart(df_merged).mark_circle(
                size=60,
                opacity=0.6
            ).encode(
                x=alt.X('estrogen:Q', title='Estrogen Level', scale=alt.Scale(zero=False)),
                y=alt.Y('hr_mean:Q', title='Heart Rate (bpm)', scale=alt.Scale(zero=False)),
                color=alt.Color('phase_name:N', 
                              title='Cycle Phase',
                              scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('estrogen:Q', title='Estrogen', format='.2f'),
                    alt.Tooltip('hr_mean:Q', title='Heart Rate', format='.1f'),
                    alt.Tooltip('phase_name:N', title='Phase')
                ]
            ).properties(
                width=350,
                height=350,
                title='Heart Rate vs Estrogen'
            ).interactive()
            
            # HR vs LH scatter
            scatter_lh = alt.Chart(df_merged).mark_circle(
                size=60,
                opacity=0.6
            ).encode(
                x=alt.X('lh:Q', title='LH Level', scale=alt.Scale(zero=False)),
                y=alt.Y('hr_mean:Q', title='Heart Rate (bpm)', scale=alt.Scale(zero=False)),
                color=alt.Color('phase_name:N', 
                              title='Cycle Phase',
                              scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('lh:Q', title='LH', format='.2f'),
                    alt.Tooltip('hr_mean:Q', title='Heart Rate', format='.1f'),
                    alt.Tooltip('phase_name:N', title='Phase')
                ]
            ).properties(
                width=350,
                height=350,
                title='Heart Rate vs LH'
            ).interactive()
            
            combined_scatter = alt.hconcat(scatter_estrogen, scatter_lh)
            st.altair_chart(combined_scatter, use_container_width=True)
            st.markdown("These scatter plots reveal potential correlations between heart rate and hormone levels, colored by cycle phase to show patterns.")


def decode_phase(phase_code):
    """Helper function to decode phase numbers to phase names"""
    phase_map = {
        1: 'Follicular',
        2: 'Fertility',
        3: 'Luteal',
        4: 'Menstrual'
    }
    if pd.isna(phase_code):
        return 'Unknown'
    return phase_map.get(int(phase_code), 'Unknown')
  
