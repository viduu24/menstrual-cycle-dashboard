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

class MenstrualCycleVisualizer:
    """
    Comprehensive visualization suite for heart rate and hormone data
    aligned with menstrual cycle phases.
    """
    
    def __init__(self, data_path):
        """
        Initialize visualizer with merged HR + hormone dataset.
        
        Args:
            data_path: Path to final_merged_hr_hormones.csv
        """
        self.df = pd.read_csv(data_path)
        self.setup_style()
        print(f"✓ Loaded dataset: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
    def setup_style(self):
        """Set publication-quality plot style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    # =====================================================
    # VISUALIZATION 1: Heart Rate Across Menstrual Phases
    # =====================================================
    
    def plot_hr_by_cycle_phase(self, save_path='hr_by_phase.png'):
        """
        Box plots + violin plots showing HR distribution across cycle phases.
        Demonstrates time series patterns (RUBRIC: Specialized Data 5%)
        """
        print("\n📊 Creating HR by Cycle Phase visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Detect HR column name
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        # Define cycle phases (customize based on your data)
        if 'cycle_phase' in self.df.columns:
            phase_col = 'cycle_phase'
        elif 'phase' in self.df.columns:
            phase_col = 'phase'
        else:
            # Create phases based on day_in_study if available
            if 'day_in_study' in self.df.columns:
                self.df['cycle_phase'] = pd.cut(
                    self.df['day_in_study'] % 28,
                    bins=[0, 5, 14, 28],
                    labels=['Menstrual', 'Follicular', 'Luteal']
                )
                phase_col = 'cycle_phase'
            else:
                print("⚠ No cycle phase information found")
                return
        
        # 1. Box Plot
        sns.boxplot(data=self.df, x=phase_col, y=hr_col, ax=axes[0, 0],
                   palette='Set2', linewidth=2)
        axes[0, 0].set_title('Heart Rate Distribution by Cycle Phase', 
                            fontweight='bold', fontsize=14)
        axes[0, 0].set_ylabel('Heart Rate (bpm)', fontsize=12)
        axes[0, 0].set_xlabel('Cycle Phase', fontsize=12)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Violin Plot with quartiles
        sns.violinplot(data=self.df, x=phase_col, y=hr_col, ax=axes[0, 1],
                      palette='muted', inner='quartile')
        axes[0, 1].set_title('Heart Rate Density by Cycle Phase',
                            fontweight='bold', fontsize=14)
        axes[0, 1].set_ylabel('Heart Rate (bpm)', fontsize=12)
        axes[0, 1].set_xlabel('Cycle Phase', fontsize=12)
        
        # 3. Mean ± SEM by phase
        phase_stats = self.df.groupby(phase_col)[hr_col].agg(['mean', 'sem'])
        x_pos = range(len(phase_stats))
        axes[1, 0].bar(x_pos, phase_stats['mean'], yerr=phase_stats['sem'],
                      capsize=8, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(phase_stats.index, rotation=0)
        axes[1, 0].set_title('Mean Heart Rate ± SEM by Phase',
                            fontweight='bold', fontsize=14)
        axes[1, 0].set_ylabel('Heart Rate (bpm)', fontsize=12)
        axes[1, 0].set_xlabel('Cycle Phase', fontsize=12)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Statistical significance (ANOVA)
        phases = self.df[phase_col].unique()
        phase_groups = [self.df[self.df[phase_col] == p][hr_col].dropna() 
                       for p in phases]
        
        if len(phase_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*phase_groups)
            
            axes[1, 1].axis('off')
            stats_text = f"""
            STATISTICAL ANALYSIS
            {'='*40}
            
            Test: One-Way ANOVA
            Groups: {', '.join(map(str, phases))}
            
            F-statistic: {f_stat:.3f}
            P-value: {p_value:.4f}
            
            Interpretation:
            {'Significant difference' if p_value < 0.05 else 'No significant difference'}
            between phases (α = 0.05)
            
            Sample Sizes:
            """
            for phase in phases:
                n = len(self.df[self.df[phase_col] == phase])
                stats_text += f"\n  {phase}: n={n}"
            
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                          fontsize=11, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
    
    # =====================================================
    # VISUALIZATION 2: Time Series of HR + Hormones
    # =====================================================
    
    def plot_hr_hormone_timeseries(self, participant_id=None, save_path='hr_hormone_timeseries.png'):
        """
        Dual-axis time series showing HR and hormone levels together.
        Publication-quality with confidence intervals.
        """
        print("\n📊 Creating HR + Hormone Time Series...")
        
        if participant_id is None:
            participant_id = self.df['id'].iloc[0]
        
        # Filter for one participant
        df_p = self.df[self.df['id'] == participant_id].sort_values('day_in_study')
        
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        # Detect hormone columns
        hormone_cols = [col for col in df_p.columns 
                       if any(x in col.lower() for x in ['estrogen', 'progesterone', 'lh', 'fsh'])]
        
        if not hormone_cols:
            print("⚠ No hormone columns found")
            return
        
        fig, axes = plt.subplots(len(hormone_cols) + 1, 1, 
                                figsize=(14, 4 * (len(hormone_cols) + 1)),
                                sharex=True)
        
        if len(hormone_cols) == 0:
            axes = [axes]
        
        # Plot HR
        ax1 = axes[0]
        ax1.plot(df_p['day_in_study'], df_p[hr_col], 
                linewidth=2.5, color='crimson', label='Heart Rate', marker='o', markersize=4)
        
        # Add confidence band if hr_std exists
        if 'hr_std' in df_p.columns:
            ax1.fill_between(df_p['day_in_study'],
                            df_p[hr_col] - df_p['hr_std'],
                            df_p[hr_col] + df_p['hr_std'],
                            alpha=0.2, color='crimson', label='±1 SD')
        
        ax1.set_ylabel('Heart Rate (bpm)', fontsize=13, fontweight='bold', color='crimson')
        ax1.tick_params(axis='y', labelcolor='crimson')
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_title(f'Participant {participant_id}: Heart Rate & Hormone Dynamics',
                     fontsize=15, fontweight='bold', pad=15)
        
        # Plot each hormone
        colors = ['blue', 'green', 'purple', 'orange']
        for i, (hormone_col, color) in enumerate(zip(hormone_cols, colors), 1):
            ax = axes[i]
            
            # Remove missing values for cleaner plot
            valid_data = df_p[['day_in_study', hormone_col]].dropna()
            
            ax.plot(valid_data['day_in_study'], valid_data[hormone_col],
                   linewidth=2.5, color=color, marker='s', markersize=5,
                   label=hormone_col.replace('_', ' ').title())
            
            ax.set_ylabel(hormone_col.replace('_', ' ').title(),
                         fontsize=13, fontweight='bold', color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper left')
        
        axes[-1].set_xlabel('Day in Study', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
    
    # =====================================================
    # VISUALIZATION 3: Correlation Heatmap
    # =====================================================
    
    def plot_correlation_matrix(self, save_path='correlation_heatmap.png'):
        """
        Advanced correlation heatmap with hierarchical clustering.
        Shows relationships between HR metrics and hormones.
        """
        print("\n📊 Creating Correlation Matrix...")
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Focus on HR and hormone columns
        hr_cols = [col for col in numeric_cols 
                  if any(x in col.lower() for x in ['hr_', 'heart_rate', 'bpm'])]
        hormone_cols = [col for col in numeric_cols 
                       if any(x in col.lower() for x in ['estrogen', 'progesterone', 'lh', 'fsh', 'testosterone'])]
        
        selected_cols = hr_cols + hormone_cols
        
        if len(selected_cols) < 2:
            print("⚠ Not enough numeric columns for correlation")
            return
        
        # Compute correlation matrix
        corr_matrix = self.df[selected_cols].corr()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Regular heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
                   cbar_kws={'label': 'Correlation Coefficient'})
        axes[0].set_title('Correlation Matrix: HR & Hormones',
                         fontsize=15, fontweight='bold', pad=15)
        
        # 2. Clustered heatmap
        sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                      center=0, vmin=-1, vmax=1, figsize=(10, 10),
                      cbar_kws={'label': 'Correlation Coefficient'})
        plt.savefig(save_path.replace('.png', '_clustered.png'), dpi=300, bbox_inches='tight')
        
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'See separate clustered heatmap figure →',
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=14, style='italic')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        print(f"✓ Saved clustered version to {save_path.replace('.png', '_clustered.png')}")
        plt.show()
    
    # =====================================================
    # VISUALIZATION 4: Heart Rate Variability Analysis
    # =====================================================
    
    def plot_hrv_analysis(self, save_path='hrv_analysis.png'):
        """
        Heart Rate Variability (standard deviation) across cycle.
        Shows autonomic nervous system activity.
        """
        print("\n📊 Creating HRV Analysis...")
        
        if 'hr_std' not in self.df.columns:
            print("⚠ No hr_std column for variability analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. HRV distribution
        axes[0, 0].hist(self.df['hr_std'].dropna(), bins=50, 
                       color='teal', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.df['hr_std'].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].axvline(self.df['hr_std'].median(), color='orange',
                          linestyle='--', linewidth=2, label='Median')
        axes[0, 0].set_xlabel('Heart Rate Std Dev (bpm)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Distribution of Heart Rate Variability',
                            fontweight='bold', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. HRV over time (rolling average)
        if 'day_in_study' in self.df.columns:
            daily_hrv = self.df.groupby('day_in_study')['hr_std'].mean()
            rolling_hrv = daily_hrv.rolling(window=7, center=True).mean()
            
            axes[0, 1].plot(daily_hrv.index, daily_hrv.values, 
                           alpha=0.3, color='gray', label='Daily HRV')
            axes[0, 1].plot(rolling_hrv.index, rolling_hrv.values,
                           linewidth=3, color='darkblue', label='7-day Rolling Avg')
            axes[0, 1].set_xlabel('Day in Study', fontsize=12)
            axes[0, 1].set_ylabel('HRV (Std Dev)', fontsize=12)
            axes[0, 1].set_title('Heart Rate Variability Over Time',
                                fontweight='bold', fontsize=14)
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. HRV by cycle phase
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
        
        if phase_col in self.df.columns:
            sns.violinplot(data=self.df, x=phase_col, y='hr_std',
                          ax=axes[1, 0], palette='Set3', inner='box')
            axes[1, 0].set_xlabel('Cycle Phase', fontsize=12)
            axes[1, 0].set_ylabel('HRV (Std Dev)', fontsize=12)
            axes[1, 0].set_title('HRV Distribution by Cycle Phase',
                                fontweight='bold', fontsize=14)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. HR mean vs HRV scatter
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        axes[1, 1].scatter(self.df[hr_col], self.df['hr_std'],
                          alpha=0.5, s=30, color='purple')
        
        # Add regression line
        valid_data = self.df[[hr_col, 'hr_std']].dropna()
        z = np.polyfit(valid_data[hr_col], valid_data['hr_std'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data[hr_col].min(), valid_data[hr_col].max(), 100)
        axes[1, 1].plot(x_line, p(x_line), "r--", linewidth=2, 
                       label=f'y={z[0]:.3f}x+{z[1]:.2f}')
        
        axes[1, 1].set_xlabel('Mean Heart Rate (bpm)', fontsize=12)
        axes[1, 1].set_ylabel('HRV (Std Dev)', fontsize=12)
        axes[1, 1].set_title('Heart Rate vs Variability',
                            fontweight='bold', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
    
    # =====================================================
    # VISUALIZATION 5: Multi-Participant Comparison
    # =====================================================
    
    def plot_participant_comparison(self, n_participants=6, save_path='participant_comparison.png'):
        """
        Small multiples showing HR patterns across multiple participants.
        Demonstrates inter-individual variability.
        """
        print("\n📊 Creating Participant Comparison...")
        
        participants = self.df['id'].unique()[:n_participants]
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        n_cols = 3
        n_rows = int(np.ceil(len(participants) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows), sharex=True)
        axes = axes.flatten() if n_participants > 1 else [axes]
        
        for i, pid in enumerate(participants):
            df_p = self.df[self.df['id'] == pid].sort_values('day_in_study')
            
            axes[i].plot(df_p['day_in_study'], df_p[hr_col],
                        linewidth=2, marker='o', markersize=3, alpha=0.8)
            
            # Add shaded region for variability if available
            if 'hr_std' in df_p.columns:
                axes[i].fill_between(df_p['day_in_study'],
                                    df_p[hr_col] - df_p['hr_std'],
                                    df_p[hr_col] + df_p['hr_std'],
                                    alpha=0.2)
            
            axes[i].set_title(f'Participant {pid}', fontweight='bold', fontsize=12)
            axes[i].set_ylabel('Heart Rate (bpm)', fontsize=11)
            axes[i].grid(alpha=0.3)
            
            # Add mean line
            mean_hr = df_p[hr_col].mean()
            axes[i].axhline(mean_hr, color='red', linestyle='--', 
                           alpha=0.5, label=f'Mean={mean_hr:.1f}')
            axes[i].legend(loc='upper right', fontsize=9)
        
        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.text(0.5, 0.02, 'Day in Study', ha='center', fontsize=14, fontweight='bold')
        fig.suptitle('Heart Rate Patterns Across Participants',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
    
    # =====================================================
    # VISUALIZATION 6: Circadian Rhythm Analysis
    # =====================================================
    
    def plot_circadian_patterns(self, save_path='circadian_patterns.png'):
        """
        Analyzes hourly HR patterns if hour_of_day exists.
        Shows daily rhythm variations across cycle phases.
        """
        print("\n📊 Creating Circadian Pattern Analysis...")
        
        if 'hour_of_day' not in self.df.columns:
            print("⚠ No hour_of_day column found - skipping circadian analysis")
            return
        
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average HR by hour
        hourly_hr = self.df.groupby('hour_of_day')[hr_col].agg(['mean', 'sem'])
        axes[0, 0].plot(hourly_hr.index, hourly_hr['mean'], 
                       linewidth=3, marker='o', markersize=8, color='darkblue')
        axes[0, 0].fill_between(hourly_hr.index,
                               hourly_hr['mean'] - hourly_hr['sem'],
                               hourly_hr['mean'] + hourly_hr['sem'],
                               alpha=0.3)
        axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
        axes[0, 0].set_ylabel('Mean Heart Rate (bpm)', fontsize=12)
        axes[0, 0].set_title('24-Hour Heart Rate Pattern (Average)',
                            fontweight='bold', fontsize=14)
        axes[0, 0].set_xticks(range(0, 24, 2))
        axes[0, 0].grid(alpha=0.3)
        
        # Add day/night shading
        axes[0, 0].axvspan(0, 6, alpha=0.1, color='blue', label='Sleep')
        axes[0, 0].axvspan(22, 24, alpha=0.1, color='blue')
        axes[0, 0].legend()
        
        # 2. Heatmap of HR by hour and day
        if 'day_in_study' in self.df.columns:
            pivot_data = self.df.pivot_table(
                values=hr_col,
                index='hour_of_day',
                columns='day_in_study',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, cmap='YlOrRd', ax=axes[0, 1],
                       cbar_kws={'label': 'Heart Rate (bpm)'})
            axes[0, 1].set_title('Heart Rate: Hour × Day Heatmap',
                                fontweight='bold', fontsize=14)
            axes[0, 1].set_xlabel('Day in Study', fontsize=12)
            axes[0, 1].set_ylabel('Hour of Day', fontsize=12)
        
        # 3. HR by hour, colored by cycle phase
        if 'cycle_phase' in self.df.columns or 'phase' in self.df.columns:
            phase_col = 'cycle_phase' if 'cycle_phase' in self.df.columns else 'phase'
            
            for phase in self.df[phase_col].unique():
                phase_data = self.df[self.df[phase_col] == phase]
                hourly = phase_data.groupby('hour_of_day')[hr_col].mean()
                axes[1, 0].plot(hourly.index, hourly.values,
                               linewidth=2, marker='o', label=phase)
            
            axes[1, 0].set_xlabel('Hour of Day', fontsize=12)
            axes[1, 0].set_ylabel('Mean Heart Rate (bpm)', fontsize=12)
            axes[1, 0].set_title('Circadian Patterns by Cycle Phase',
                                fontweight='bold', fontsize=14)
            axes[1, 0].legend()
            axes[1, 0].set_xticks(range(0, 24, 2))
            axes[1, 0].grid(alpha=0.3)
        
        # 4. Peak detection
        hourly_avg = self.df.groupby('hour_of_day')[hr_col].mean()
        peaks, properties = find_peaks(hourly_avg.values, prominence=1)
        
        axes[1, 1].plot(hourly_avg.index, hourly_avg.values,
                       linewidth=3, color='darkgreen', label='HR Pattern')
        axes[1, 1].plot(hourly_avg.index[peaks], hourly_avg.values[peaks],
                       'ro', markersize=12, label=f'Peaks (n={len(peaks)})')
        axes[1, 1].set_xlabel('Hour of Day', fontsize=12)
        axes[1, 1].set_ylabel('Mean Heart Rate (bpm)', fontsize=12)
        axes[1, 1].set_title('Peak Heart Rate Hours',
                            fontweight='bold', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].set_xticks(range(0, 24, 2))
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
    
    # =====================================================
    # MASTER FUNCTION: Generate All Visualizations
    # =====================================================
    
    def generate_all_visualizations(self, output_dir='visualizations'):
        """
        Generate all visualizations at once.
        Perfect for meeting project requirements!
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATION SUITE")
        print("="*60)
        
        viz_functions = [
            (self.plot_hr_by_cycle_phase, f'{output_dir}/01_hr_by_phase.png'),
            (self.plot_hr_hormone_timeseries, f'{output_dir}/02_hr_hormone_timeseries.png'),
            (self.plot_correlation_matrix, f'{output_dir}/03_correlation_heatmap.png'),
            (self.plot_hrv_analysis, f'{output_dir}/04_hrv_analysis.png'),
            (self.plot_participant_comparison, f'{output_dir}/05_participant_comparison.png'),
            (self.plot_circadian_patterns, f'{output_dir}/06_circadian_patterns.png'),
        ]
        
        for i, (func, path) in enumerate(viz_functions, 1):
            try:
                print(f"\n[{i}/{len(viz_functions)}] ", end='')
                func(save_path=path)
            except Exception as e:
                print(f"⚠ Error: {str(e)}")
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
        "m1_model": load_pickle("model1_hr_prediction.pkl"),
        "m1_scaler": load_pickle("model1_scaler.pkl"),
        "m1_features": load_pickle("model1_features.pkl"),

        # Model 2 – Phase prediction (LightGBM)
        "m2_model": load_pickle("model2_phase_prediction_lgbm.pkl"),
        "m2_scaler": load_pickle("model2_scaler.pkl"),
        "m2_encoder": load_pickle("model2_encoder.pkl"),
        "m2_features": load_pickle("model2_features.pkl"),

        # Model 3 – Regularity prediction (RandomForestClassifier)
        "m3_model": load_pickle("model3_regularity.pkl"),
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
page = st.sidebar.radio("Go to", ["README", "Data Description", "Missingness","Cleaning Process", "Information","Graphs", "ML models"])

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
        
        
