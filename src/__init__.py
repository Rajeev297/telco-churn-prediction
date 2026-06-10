"""
Customer Churn Prediction ML Project
Complete machine learning system for predicting customer churn with SHAP explanability.

This package includes:
- Data loading and preprocessing
- Exploratory data analysis
- Model training and evaluation
- SHAP-based model explanations
- Streamlit web application
"""

__version__ = "1.0.0"
__author__ = "ML Engineer"
__description__ = "Customer Churn Prediction with SHAP Explainability"

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor, DataValidator, prepare_data
from src.eda import EDAAnalyzer
from src.model_trainer import ModelTrainer, create_evaluation_plots
from src.shap_explainer import SHAPExplainer, generate_shap_report

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'DataValidator',
    'prepare_data',
    'EDAAnalyzer',
    'ModelTrainer',
    'create_evaluation_plots',
    'SHAPExplainer',
    'generate_shap_report'
]
