"""
Exploratory Data Analysis (EDA) Module
Generates comprehensive visualizations and business insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EDAAnalyzer:
    """Performs comprehensive exploratory data analysis."""
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize EDA analyzer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.insights = {}
        
    def analyze_churn_distribution(self, df: pd.DataFrame) -> None:
        """
        Analyze and visualize churn distribution.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing churn distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        churn_counts = df['Churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(churn_counts.index, churn_counts.values, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xlabel('Churn', fontsize=12)
        
        # Add count labels on bars
        for i, (idx, val) in enumerate(churn_counts.items()):
            axes[0].text(i, val + 50, str(val), ha='center', fontsize=11, fontweight='bold')
        
        # Pie chart
        churn_pct = df['Churn'].value_counts(normalize=True) * 100
        axes[1].pie(churn_pct.values, labels=churn_pct.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, explode=(0.05, 0), shadow=True)
        axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_churn_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store insights
        churn_rate = (df['Churn'] == 'Yes').sum() / len(df) * 100
        self.insights['churn_rate'] = f"{churn_rate:.2f}%"
        logger.info(f"Churn rate: {churn_rate:.2f}%")
    
    def analyze_tenure(self, df: pd.DataFrame) -> None:
        """
        Analyze tenure distribution by churn status.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing tenure patterns...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution by churn
        df[df['Churn'] == 'No']['tenure'].hist(bins=30, alpha=0.6, label='No Churn', ax=axes[0], color='#2ecc71')
        df[df['Churn'] == 'Yes']['tenure'].hist(bins=30, alpha=0.6, label='Churn', ax=axes[0], color='#e74c3c')
        axes[0].set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Tenure (months)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        
        # Box plot
        data_to_plot = [df[df['Churn'] == 'No']['tenure'], df[df['Churn'] == 'Yes']['tenure']]
        bp = axes[1].boxplot(data_to_plot, labels=['No Churn', 'Churn'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title('Tenure by Churn Status (Box Plot)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Tenure (months)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_tenure_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        avg_tenure_churn = df[df['Churn'] == 'Yes']['tenure'].mean()
        avg_tenure_no_churn = df[df['Churn'] == 'No']['tenure'].mean()
        self.insights['avg_tenure_churn'] = f"{avg_tenure_churn:.1f} months"
        self.insights['avg_tenure_no_churn'] = f"{avg_tenure_no_churn:.1f} months"
        logger.info(f"Avg tenure (Churn): {avg_tenure_churn:.1f} months | (No Churn): {avg_tenure_no_churn:.1f} months")
    
    def analyze_monthly_charges(self, df: pd.DataFrame) -> None:
        """
        Analyze monthly charges distribution by churn status.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing monthly charges...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution by churn
        df[df['Churn'] == 'No']['MonthlyCharges'].hist(bins=30, alpha=0.6, label='No Churn', ax=axes[0], color='#2ecc71')
        df[df['Churn'] == 'Yes']['MonthlyCharges'].hist(bins=30, alpha=0.6, label='Churn', ax=axes[0], color='#e74c3c')
        axes[0].set_title('Monthly Charges Distribution by Churn Status', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Monthly Charges ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        
        # Box plot
        data_to_plot = [df[df['Churn'] == 'No']['MonthlyCharges'], 
                        df[df['Churn'] == 'Yes']['MonthlyCharges']]
        bp = axes[1].boxplot(data_to_plot, labels=['No Churn', 'Churn'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title('Monthly Charges by Churn Status (Box Plot)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Monthly Charges ($)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_monthly_charges_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        avg_charges_churn = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
        avg_charges_no_churn = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
        self.insights['avg_monthly_charges_churn'] = f"${avg_charges_churn:.2f}"
        self.insights['avg_monthly_charges_no_churn'] = f"${avg_charges_no_churn:.2f}"
        logger.info(f"Avg monthly charges (Churn): ${avg_charges_churn:.2f} | (No Churn): ${avg_charges_no_churn:.2f}")
    
    def analyze_contract_type(self, df: pd.DataFrame) -> None:
        """
        Analyze contract type vs churn.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing contract types...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        contract_churn.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[0].set_title('Churn by Contract Type', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Contract Type', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].legend(title='Churn')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Churn rate by contract
        churn_rate_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
        axes[1].bar(churn_rate_contract.index, churn_rate_contract.values, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Contract Type', fontsize=12)
        axes[1].set_ylabel('Churn Rate (%)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for i, v in enumerate(churn_rate_contract.values):
            axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_contract_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.insights['highest_churn_contract'] = churn_rate_contract.idxmax()
        self.insights['highest_churn_rate'] = f"{churn_rate_contract.max():.1f}%"
        logger.info(f"Highest churn rate: {churn_rate_contract.max():.1f}% ({churn_rate_contract.idxmax()})")
    
    def analyze_internet_service(self, df: pd.DataFrame) -> None:
        """
        Analyze internet service type vs churn.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing internet service...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
        internet_churn.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[0].set_title('Churn by Internet Service Type', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Internet Service Type', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].legend(title='Churn')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Churn rate by internet service
        churn_rate_internet = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
        axes[1].bar(churn_rate_internet.index, churn_rate_internet.values, color='#3498db', alpha=0.7, edgecolor='black')
        axes[1].set_title('Churn Rate by Internet Service Type', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Internet Service Type', fontsize=12)
        axes[1].set_ylabel('Churn Rate (%)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for i, v in enumerate(churn_rate_internet.values):
            axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_internet_service_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_demographic_features(self, df: pd.DataFrame) -> None:
        """
        Analyze demographic features vs churn.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing demographic features...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        demographic_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        
        for idx, feature in enumerate(demographic_features):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            churn_by_feature = pd.crosstab(df[feature], df['Churn'])
            churn_rate = df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
            
            ax.bar(churn_rate.index, churn_rate.values, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Churn Rate (%)', fontsize=11)
            
            # Add percentage labels
            for i, v in enumerate(churn_rate.values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_demographic_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_services(self, df: pd.DataFrame) -> None:
        """
        Analyze additional services impact on churn.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing service usage...")
        
        service_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        service_features = [f for f in service_features if f in df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(service_features):
            ax = axes[idx]
            
            churn_rate = df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
            
            ax.bar(churn_rate.index, churn_rate.values, color='#f39c12', alpha=0.7, edgecolor='black')
            ax.set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Churn Rate (%)', fontsize=11)
            
            # Add percentage labels
            for i, v in enumerate(churn_rate.values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_services_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_numerical_features(self, df: pd.DataFrame) -> None:
        """
        Generate correlation heatmap for numerical features.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Analyzing numerical correlations...")
        
        # Prepare data for correlation
        df_temp = df.copy()
        df_temp['Churn'] = (df_temp['Churn'] == 'Yes').astype(int)
        
        # Select numerical columns
        numerical_cols = df_temp.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_temp[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '08_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary of insights.
        
        Returns:
            Summary report as string
        """
        logger.info("Generating summary report...")
        
        report = "\n" + "="*70 + "\n"
        report += "EXPLORATORY DATA ANALYSIS - SUMMARY REPORT\n"
        report += "="*70 + "\n\n"
        
        for key, value in self.insights.items():
            report += f"{key.replace('_', ' ').title()}: {value}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
    
    def run_all_analyses(self, df: pd.DataFrame) -> str:
        """
        Run all EDA analyses.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Summary report
        """
        logger.info("Running comprehensive EDA analysis...")
        
        self.analyze_churn_distribution(df)
        self.analyze_tenure(df)
        self.analyze_monthly_charges(df)
        self.analyze_contract_type(df)
        self.analyze_internet_service(df)
        self.analyze_demographic_features(df)
        self.analyze_services(df)
        self.analyze_numerical_features(df)
        
        report = self.generate_summary_report()
        
        logger.info("EDA analysis complete!")
        return report


if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_sample_data()
    
    analyzer = EDAAnalyzer(output_dir='outputs')
    report = analyzer.run_all_analyses(df)
    print(report)
