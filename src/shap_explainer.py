"""
SHAP Explainability Module
Provides model interpretability using SHAP (SHapley Additive exPlanations).
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Tuple, List, Any, Dict
import shap
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Provides SHAP-based model explanations."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data for SHAP
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names else X_train.columns.tolist()
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        self._initialize_explainer()
        
    def _initialize_explainer(self) -> None:
        """Initialize the SHAP explainer based on model type."""
        logger.info("Initializing SHAP explainer...")
        
        model_class_name = self.model.__class__.__name__
        
        try:
            if model_class_name == 'LogisticRegression':
                # Use KernelExplainer for Logistic Regression
                logger.info("Using KernelExplainer for Logistic Regression")
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(self.X_train, min(100, len(self.X_train)))
                )
            elif 'RandomForest' in model_class_name or 'XGB' in model_class_name or 'LGBM' in model_class_name:
                # Use TreeExplainer for tree-based models (RF, XGBoost, LightGBM)
                logger.info(f"Using TreeExplainer for {model_class_name}")
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer
                logger.info("Using KernelExplainer as fallback")
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(self.X_train, min(100, len(self.X_train)))
                )
            
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            logger.info("Using KernelExplainer as fallback")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X_train, min(100, len(self.X_train)))
            )
    
    def calculate_shap_values(self, X_data: pd.DataFrame, max_samples: int = None) -> np.ndarray:
        """
        Calculate SHAP values for given data.
        
        Args:
            X_data: Data to calculate SHAP values for
            max_samples: Maximum samples to process (for performance)
            
        Returns:
            SHAP values array
        """
        logger.info(f"Calculating SHAP values for {len(X_data)} samples...")
        
        if max_samples and len(X_data) > max_samples:
            X_data = X_data.iloc[:max_samples]
            logger.info(f"Using {max_samples} samples for SHAP calculation")
        
        try:
            shap_values = self.explainer.shap_values(X_data)
            
            # Handle output format from different explainers
            if isinstance(shap_values, list):
                # For binary classification: shap_values[1] is for class 1
                self.shap_values = np.asarray(shap_values[1])
                self.base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
            else:
                self.shap_values = np.asarray(shap_values)
                expected_value = self.explainer.expected_value
                # For TreeExplainer with binary XGBoost, expected_value may be a list
                if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                    self.base_value = expected_value[1]
                else:
                    self.base_value = expected_value
            
            # Ensure 2D shape (n_samples, n_features)
            if self.shap_values.ndim > 2:
                self.shap_values = self.shap_values.reshape(self.shap_values.shape[0], -1)
            
            # Handle case where shap_values has 2x features (both classes returned side-by-side)
            n_expected = len(self.feature_names)
            if self.shap_values.shape[1] > n_expected:
                # Take the second half (class 1 contributions)
                self.shap_values = self.shap_values[:, -n_expected:]
            
            logger.info(f"SHAP values calculated successfully. Shape: {self.shap_values.shape}")
            return self.shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            raise
    
    def get_feature_importance(self, X_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X_data: Data to calculate importance for (uses training data if None)
            
        Returns:
            DataFrame with feature importance ranking
        """
        if self.shap_values is None:
            if X_data is not None:
                self.calculate_shap_values(X_data)
            else:
                self.calculate_shap_values(self.X_train.iloc[:100])
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance calculated")
        return importance_df
    
    def plot_feature_importance(self, X_data: pd.DataFrame = None, top_n: int = 15,
                               output_path: str = None) -> None:
        """
        Plot feature importance based on SHAP values.
        
        Args:
            X_data: Data to calculate importance for
            top_n: Number of top features to display
            output_path: Path to save plot
        """
        logger.info("Creating feature importance plot...")
        
        importance_df = self.get_feature_importance(X_data)
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                       color='#3498db', alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        ax.set_title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_features['importance'].values):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        plt.close()
    
    def explain_prediction(self, X_sample: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.
        
        Args:
            X_sample: Sample to explain
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with explanation data
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_sample)
        
        prediction = self.model.predict(X_sample)[sample_idx]
        prediction_proba = self.model.predict_proba(X_sample)[sample_idx]
        
        # Get SHAP values for the sample and ensure 1D
        sample_shap_values = np.asarray(self.shap_values[sample_idx]).ravel()
        
        # Get feature values and ensure 1D
        feature_values = np.asarray(X_sample.iloc[sample_idx].values).ravel()
        
        # Trim to match lengths
        n_features = min(len(sample_shap_values), len(self.feature_names), len(feature_values))
        
        # Sort features by absolute SHAP value
        feature_impact = pd.DataFrame({
            'feature': self.feature_names[:n_features],
            'shap_value': sample_shap_values[:n_features],
            'feature_value': feature_values[:n_features],
            'abs_shap': np.abs(sample_shap_values[:n_features])
        }).sort_values('abs_shap', ascending=False)
        
        explanation = {
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'base_value': self.base_value,
            'feature_impact': feature_impact,
            'top_features': feature_impact.head(10).to_dict('records')
        }
        
        logger.info(f"Prediction explanation generated")
        return explanation
    
    def plot_prediction_explanation(self, X_sample: pd.DataFrame, sample_idx: int = 0,
                                   output_path: str = None) -> None:
        """
        Plot explanation for a single prediction.
        
        Args:
            X_sample: Sample to explain
            sample_idx: Index of sample to explain
            output_path: Path to save plot
        """
        logger.info("Creating prediction explanation plot...")
        
        explanation = self.explain_prediction(X_sample, sample_idx)
        top_features = explanation['feature_impact'].head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['shap_value'].values]
        bars = ax.barh(range(len(top_features)), top_features['shap_value'].values,
                       color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
        ax.set_title(f'Local Feature Importance - Sample {sample_idx}\nPrediction: {explanation["prediction"]}',
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_features['shap_value'].values):
            ax.text(v + (0.01 if v > 0 else -0.01), i, f'{v:.4f}', 
                   va='center', ha='left' if v > 0 else 'right', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction explanation plot saved to {output_path}")
        
        plt.close()
    
    def save_explainer(self, filepath: str) -> None:
        """
        Save explainer to file.
        
        Args:
            filepath: Path to save explainer
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Explainer saved to {filepath}")
    
    @staticmethod
    def load_explainer(filepath: str) -> 'SHAPExplainer':
        """
        Load explainer from file.
        
        Args:
            filepath: Path to load explainer from
            
        Returns:
            Loaded SHAPExplainer instance
        """
        with open(filepath, 'rb') as f:
            explainer = pickle.load(f)
        logger.info(f"Explainer loaded from {filepath}")
        return explainer


def generate_shap_report(explainer: SHAPExplainer, X_data: pd.DataFrame,
                        output_dir: str = 'outputs') -> str:
    """
    Generate a comprehensive SHAP report.
    
    Args:
        explainer: SHAPExplainer instance
        X_data: Data to analyze
        output_dir: Directory to save plots
        
    Returns:
        Report as string
    """
    logger.info("Generating SHAP report...")
    
    importance_df = explainer.get_feature_importance(X_data)
    
    report = "\n" + "="*70 + "\n"
    report += "SHAP EXPLAINABILITY REPORT\n"
    report += "="*70 + "\n\n"
    
    report += "TOP 10 MOST IMPORTANT FEATURES (by mean |SHAP value|):\n"
    report += "-" * 70 + "\n"
    for idx, row in importance_df.head(10).iterrows():
        report += f"{row['feature']:40s}: {row['importance']:.6f}\n"
    
    report += "\n" + "="*70 + "\n"
    
    # Create visualizations
    explainer.plot_feature_importance(X_data, top_n=15,
                                     output_path=os.path.join(output_dir, '11_shap_feature_importance.png'))
    
    # Example prediction explanation
    if len(X_data) > 0:
        explainer.plot_prediction_explanation(X_data, sample_idx=0,
                                             output_path=os.path.join(output_dir, '12_shap_prediction_explanation.png'))
    
    return report


if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import prepare_data
    from model_trainer import ModelTrainer
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_sample_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # Train model
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, tune_hyperparams=False)
    trainer.evaluate_all_models(X_test, y_test)
    trainer.select_best_model()
    
    # Create SHAP explainer
    explainer = SHAPExplainer(trainer.best_model, X_train.iloc[:100], X_train.columns.tolist())
    
    # Generate report
    report = generate_shap_report(explainer, X_test.iloc[:50], output_dir='outputs')
    print(report)
