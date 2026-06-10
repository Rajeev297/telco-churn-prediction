"""
Data Loading Module
Downloads and prepares the IBM Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from multiple sources."""
    
    @staticmethod
    def load_from_csv(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    @staticmethod
    def load_sample_data() -> pd.DataFrame:
        """
        Generate or load sample IBM Telco Customer Churn dataset.
        Uses a direct download from a public source if available.
        
        Returns:
            DataFrame with customer churn data
        """
        logger.info("Loading IBM Telco Customer Churn dataset...")
        
        # Try to download from a public source
        try:
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            logger.info(f"Downloading from: {url}")
            df = pd.read_csv(url)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"Could not download from public URL: {e}")
            logger.info("Creating synthetic dataset based on Telco Customer Churn schema...")
            return DataLoader._create_synthetic_data()
    
    @staticmethod
    def _create_synthetic_data() -> pd.DataFrame:
        """
        Create synthetic dataset matching IBM Telco Customer Churn schema
        for demonstration purposes.
        
        Returns:
            Synthetic DataFrame
        """
        np.random.seed(42)
        n_samples = 7043
        
        data = {
            'customerID': [f'ID-{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.55, 0.34, 0.11]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.37, 0.52, 0.11]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.55, 0.11]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.55, 0.11]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.59, 0.11]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.40, 0.49, 0.11]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.50, 0.11]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.22, 0.23]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.76, 0.24]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                              n_samples, p=[0.34, 0.23, 0.22, 0.21]),
            'MonthlyCharges': np.random.uniform(18, 120, n_samples).round(2),
            'TotalCharges': np.random.uniform(18, 8684, n_samples).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Created synthetic dataset with shape: {df.shape}")
        return df


def main():
    """Main function to test data loading."""
    # Load data
    loader = DataLoader()
    df = loader.load_sample_data()
    
    # Display basic info
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nChurn Distribution:\n{df['Churn'].value_counts()}")
    
    return df


if __name__ == "__main__":
    df = main()
