import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
        
    def plot_hr_by_cycle_phase(self, save_path='hr_by_phase.png'):
        """Box plots + violin plots showing HR distribution across cycle phases."""
        print("\n📊 Creating HR by Cycle Phase visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
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
                return
        
        # 1. Box Plot
        sns.boxplot(data=self.df, x=phase_col, y=hr_col, ax=axes[0, 0],
                   palette='Set2', linewidth=2)
        axes[0, 0].set_title('Heart Rate Distribution by Cycle Phase', 
                            fontweight='bold', fontsize=14)
        axes[0, 0].set_ylabel('Heart Rate (bpm)', fontsize=12)
        axes[0, 0].set_xlabel('Cycle Phase', fontsize=12)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Violin Plot
        sns.violinplot(data=self.df, x=phase_col, y=hr_col, ax=axes[0, 1],
                      palette='muted', inner='quartile')
        axes[0, 1].set_title('Heart Rate Density by Cycle Phase',
                            fontweight='bold', fontsize=14)
        axes[0, 1].set_ylabel('Heart Rate (bpm)', fontsize=12)
        axes[0, 1].set_xlabel('Cycle Phase', fontsize=12)
        
        # 3. Mean ± SEM
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
        
        # 4. Statistical test
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
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                          fontsize=11, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
    
    def plot_hr_hormone_timeseries(self, participant_id=None, save_path='hr_hormone_timeseries.png'):
        """Dual-axis time series showing HR and hormone levels together."""
        print("\n📊 Creating HR + Hormone Time Series...")
        
        if participant_id is None:
            participant_id = self.df['id'].iloc[0]
        
        df_p = self.df[self.df['id'] == participant_id].sort_values('day_in_study')
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        
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
        
        # Plot hormones
        colors = ['blue', 'green', 'purple', 'orange']
        for i, (hormone_col, color) in enumerate(zip(hormone_cols, colors), 1):
            ax = axes[i]
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
    
    def plot_correlation_matrix(self, save_path='correlation_heatmap.png'):
        """Advanced correlation heatmap with hierarchical clustering."""
        print("\n📊 Creating Correlation Matrix...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        hr_cols = [col for col in numeric_cols 
                  if any(x in col.lower() for x in ['hr_', 'heart_rate', 'bpm'])]
        hormone_cols = [col for col in numeric_cols 
                       if any(x in col.lower() for x in ['estrogen', 'progesterone', 'lh', 'fsh', 'testosterone'])]
        
        selected_cols = hr_cols + hormone_cols
        
        if len(selected_cols) < 2:
            print("⚠ Not enough numeric columns for correlation")
            return
        
        corr_matrix = self.df[selected_cols].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
                   cbar_kws={'label': 'Correlation Coefficient'})
        axes[0].set_title('Correlation Matrix: HR & Hormones',
                         fontsize=15, fontweight='bold', pad=15)
        
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
        plt.show()
    
    def plot_hrv_analysis(self, save_path='hrv_analysis.png'):
        """Heart Rate Variability analysis across cycle."""
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
        
        # 2. HRV over time
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
        
        # 3. HRV by phase
        phase_col = None
        if 'cycle_phase' in self.df.columns:
            phase_col = 'cycle_phase'
        elif 'phase' in self.df.columns:
            phase_col = 'phase'
        elif 'day_in_study' in self.df.columns:
            self.df['cycle_phase'] = pd.cut(
                self.df['day_in_study'] % 28,
                bins=[0, 5, 14, 28],
                labels=['Menstrual', 'Follicular', 'Luteal']
            )
            phase_col = 'cycle_phase'
        
        if phase_col:
            sns.violinplot(data=self.df, x=phase_col, y='hr_std',
                          ax=axes[1, 0], palette='Set3', inner='box')
            axes[1, 0].set_xlabel('Cycle Phase', fontsize=12)
            axes[1, 0].set_ylabel('HRV (Std Dev)', fontsize=12)
            axes[1, 0].set_title('HRV Distribution by Cycle Phase',
                                fontweight='bold', fontsize=14)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. HR vs HRV
        hr_col = 'hr_mean' if 'hr_mean' in self.df.columns else 'heart_rate'
        axes[1, 1].scatter(self.df[hr_col], self.df['hr_std'],
                          alpha=0.5, s=30, color='purple')
        
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
    
    def plot_participant_comparison(self, n_participants=6, save_path='participant_comparison.png'):
        """Small multiples showing HR patterns across participants."""
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
            
            if 'hr_std' in df_p.columns:
                axes[i].fill_between(df_p['day_in_study'],
                                    df_p[hr_col] - df_p['hr_std'],
                                    df_p[hr_col] + df_p['hr_std'],
                                    alpha=0.2)
            
            axes[i].set_title(f'Participant {pid}', fontweight='bold', fontsize=12)
            axes[i].set_ylabel('Heart Rate (bpm)', fontsize=11)
            axes[i].grid(alpha=0.3)
            
            mean_hr = df_p[hr_col].mean()
            axes[i].axhline(mean_hr, color='red', linestyle='--', 
                           alpha=0.5, label=f'Mean={mean_hr:.1f}')
            axes[i].legend(loc='upper right', fontsize=9)
        
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.text(0.5, 0.02, 'Day in Study', ha='center', fontsize=14, fontweight='bold')
        fig.suptitle('Heart Rate Patterns Across Participants',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.show()
