"""
Data Preprocessing and Validation Pipeline
Handles data cleaning, validation, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity."""
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate dataset for quality issues.
        
        Args:
            df: Input DataFrame
            verbose: Print validation results
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        if verbose:
            logger.info("="*60)
            logger.info("DATA VALIDATION REPORT")
            logger.info("="*60)
            logger.info(f"Total rows: {validation_results['total_rows']}")
            logger.info(f"Total columns: {validation_results['total_columns']}")
            logger.info(f"Duplicate rows: {validation_results['duplicate_rows']}")
            
            missing = {k: v for k, v in validation_results['missing_values'].items() if v > 0}
            if missing:
                logger.warning(f"Missing values found: {missing}")
            else:
                logger.info("No missing values detected")
        
        return validation_results


class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self, target_column: str = 'Churn'):
        """
        Initialize preprocessor.
        
        Args:
            target_column: Name of target column
        """
        self.target_column = target_column
        self.categorical_features = []
        self.numerical_features = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Define known numeric columns (to handle string-encoded numbers in datasets like TotalCharges)
        self.known_numeric_cols = {'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'}
    
    def convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert known numeric columns from string to numeric types.
        Handles datasets like IBM Telco where TotalCharges might be stored as strings.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with numeric columns properly typed
        """
        df = df.copy()
        
        for col in self.known_numeric_cols:
            if col in df.columns:
                # Check if the column is not already numeric
                if not is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        n_missing = df[col].isnull().sum()
                        if n_missing > 0:
                            logger.warning(f"Column {col}: {n_missing} non-numeric values converted to NaN. Filling with 0.")
                            df[col] = df[col].fillna(0)
                        logger.info(f"Converted {col} to numeric (dtype: {df[col].dtype})")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to numeric: {e}")
        
        return df
    
    def identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        # Exclude target and ID columns
        exclude_cols = {self.target_column, 'customerID'}
        
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            # Properly detect numeric dtypes using pandas function (handles custom dtypes)
            if is_numeric_dtype(df[col]):
                numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        logger.info(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        return categorical_features, numerical_features
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'forward_fill', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            logger.info("No missing values to handle")
            return df
        
        logger.info(f"Handling missing values in columns: {missing_cols}")
        
        for col in missing_cols:
            if col in self.numerical_features:
                if strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding and One-Hot Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if fit:
            logger.info("Fitting categorical encoders...")
        else:
            logger.info("Transforming with existing categorical encoders...")
        
        # Handle target variable encoding (binary)
        if self.target_column in df.columns:
            if df[self.target_column].dtype == 'object':
                if fit:
                    self.label_encoders[self.target_column] = LabelEncoder()
                    df[self.target_column] = self.label_encoders[self.target_column].fit_transform(df[self.target_column])
                else:
                    df[self.target_column] = self.label_encoders[self.target_column].transform(df[self.target_column])
        
        # One-hot encode categorical features
        cat_cols = [col for col in self.categorical_features if col in df.columns]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        num_cols = [col for col in self.numerical_features if col in df.columns]
        
        if not num_cols:
            return df
        
        if fit:
            logger.info(f"Fitting scaler for {len(num_cols)} numerical features...")
            scaled_values = self.scaler.fit_transform(df[num_cols])
            df[num_cols] = scaled_values
        else:
            logger.info(f"Transforming {len(num_cols)} numerical features...")
            scaled_values = self.scaler.transform(df[num_cols])
            df[num_cols] = scaled_values
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for improved model performance."""
        df = df.copy()

        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(
                df['tenure'], bins=[-1, 6, 12, 24, 48, 72],
                labels=['0-6mo', '7-12mo', '13-24mo', '25-48mo', '49-72mo']
            )

        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['avg_charge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
            max_charge = df['MonthlyCharges'].max() * 1.5 if 'MonthlyCharges' in df.columns else 200
            df['avg_charge'] = df['avg_charge'].clip(upper=max_charge)

        service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        present_services = [c for c in service_cols if c in df.columns]
        if present_services:
            df['services_count'] = sum(df[c].astype(str).str.lower().isin(['yes', '1', 'true']) for c in present_services)

        if 'Contract' in df.columns:
            df['has_contract'] = (df['Contract'].astype(str) != 'Month-to-month').astype(int)

        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['high_value_new'] = ((df['MonthlyCharges'] > 80) & (df['tenure'] < 12)).astype(int)

        logger.info(f"Engineered features added. Columns now: {list(df.columns)}")
        return df

    def fit_transform(self, df: pd.DataFrame, use_smote: bool = False,
                      y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit and transform the entire preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            use_smote: Whether to apply SMOTE for class balancing
            y: Target series (required if use_smote=True)
            
        Returns:
            Tuple of (processed DataFrame, feature names)
        """
        logger.info("="*60)
        logger.info("PREPROCESSING PIPELINE - FIT TRANSFORM")
        logger.info("="*60)
        
        # Remove ID column if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Convert numeric columns that may be stored as strings
        df = self.convert_numeric_columns(df)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Identify features
        self.identify_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Apply SMOTE (before scaling, after encoding)
        if use_smote and y is not None:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                # Convert target to binary before SMOTE if it's string
                if pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
                    y_binary = y
                else:
                    y_binary = (np.array(y) == 'Yes').astype(int)
                X_res, y_res = smote.fit_resample(df, y_binary)
                df = pd.DataFrame(X_res, columns=df.columns)
                self._smote_y = y_res
                self._smote_applied = True
                logger.info(f"SMOTE applied. Class distribution: {pd.Series(y_res).value_counts().to_dict()}")
            except ImportError:
                logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Skipping.")
        
        # Scale numerical features
        df = self.scale_features(df, fit=True)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        if self.target_column in self.feature_names:
            self.feature_names.remove(self.target_column)
        
        self.is_fitted = True
        logger.info(f"Preprocessing complete. Total features: {len(self.feature_names)}")
        
        return df, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Remove ID column if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Convert numeric columns that may be stored as strings
        df = self.convert_numeric_columns(df)
        
        # Engineer features (same as during fit)
        df = self._engineer_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Scale numerical features
        df = self.scale_features(df, fit=False)
        
        return df
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor state to file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """
        Load preprocessor state from file.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                 use_smote: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    Prepare data for modeling: split and preprocess.
    
    Args:
        df: Raw DataFrame
        test_size: Test set size ratio
        random_state: Random seed
        use_smote: Whether to apply SMOTE for class balancing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    from sklearn.model_selection import train_test_split
    
    # Validate data
    validator = DataValidator()
    validator.validate_dataset(df)
    
    # Separate features and target
    y = df['Churn'].copy()
    X = df.drop('Churn', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train/Test split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Preprocess training data
    preprocessor = DataPreprocessor(target_column='Churn')
    
    # Store pre-SMOTE data for SHAP background
    X_train_processed, feature_names = preprocessor.fit_transform(X_train, use_smote=use_smote, y=y_train)
    preprocessor.pre_smote_X = X_train_processed.copy()
    
    # Transform test data with fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    
    # Ensure same columns
    X_test_processed = X_test_processed[X_train_processed.columns]
    
    # Convert target to binary
    y_train_binary = (y_train == 'Yes').astype(int)
    y_test_binary = (y_test == 'Yes').astype(int)
    
    # If SMOTE was applied, use the SMOTE-resampled target
    if use_smote and getattr(preprocessor, '_smote_applied', False):
        y_train_binary = preprocessor._smote_y
    
    logger.info(f"Final feature count: {X_train_processed.shape[1]}")
    train_counts = pd.Series(y_train_binary).value_counts().to_dict() if hasattr(y_train_binary, 'value_counts') else {int(k): int(v) for k, v in zip(*np.unique(y_train_binary, return_counts=True))}
    logger.info(f"Churn distribution - Train: {train_counts}")
    logger.info(f"Churn distribution - Test: {y_test_binary.value_counts().to_dict()}")
    
    return X_train_processed, X_test_processed, y_train_binary, y_test_binary, preprocessor


if __name__ == "__main__":
    # Test preprocessing pipeline
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_sample_data()
    
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    print("\n" + "="*60)
    print("PREPROCESSING RESULTS")
    print("="*60)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Feature names: {X_train.columns.tolist()[:10]}... (total: {len(X_train.columns)})")
