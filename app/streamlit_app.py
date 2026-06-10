"""
Customer Churn Prediction Streamlit Application
Interactive web app for customer churn prediction with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor
from src.shap_explainer import SHAPExplainer

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .churn-high {
        background-color: #ffcccc;
    }
    .churn-medium {
        background-color: #ffffcc;
    }
    .churn-low {
        background-color: #ccffcc;
    }
</style>
""", unsafe_allow_html=True)


class ChurnPredictionApp:
    """Streamlit application for churn prediction."""
    
    def __init__(self):
        """Initialize the application."""
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.metadata = None
        self.load_artifacts()
    
    def load_artifacts(self) -> None:
        """Load trained model, preprocessor, and explainer."""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        try:
            # Load preprocessor
            preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                st.success("✓ Preprocessor loaded successfully")
            else:
                st.warning("Preprocessor not found. Please train the model first.")
            
            # Load best model
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'preprocessor' not in f and 'explainer' not in f and 'metadata' not in f]
            if model_files:
                # Get the most recent model file
                latest_model = max(model_files, key=lambda f: os.path.getctime(os.path.join(models_dir, f)))
                model_path = os.path.join(models_dir, latest_model)
                with open(model_path, 'rb') as f:
                    loaded_obj = pickle.load(f)
                    # Handle both direct model objects and wrapped model dicts
                    if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                        self.model = loaded_obj['model']
                    else:
                        self.model = loaded_obj
                st.success(f"✓ Model loaded successfully ({latest_model})")
            else:
                st.warning("No trained model found. Please train the model first.")
            
            # Load SHAP explainer
            explainer_path = os.path.join(models_dir, 'shap_explainer.pkl')
            if os.path.exists(explainer_path):
                with open(explainer_path, 'rb') as f:
                    self.explainer = pickle.load(f)
                st.success("✓ SHAP explainer loaded successfully")
            
            # Load metadata
            metadata_path = os.path.join(models_dir, 'pipeline_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
        
        except Exception as e:
            st.error(f"Error loading artifacts: {e}")
    
    def get_risk_category(self, churn_probability: float) -> Tuple[str, str]:
        """
        Determine risk category based on churn probability.
        
        Args:
            churn_probability: Predicted churn probability
            
        Returns:
            Tuple of (risk_category, color)
        """
        if churn_probability < 0.33:
            return "LOW RISK", "#2ecc71"
        elif churn_probability < 0.67:
            return "MEDIUM RISK", "#f39c12"
        else:
            return "HIGH RISK", "#e74c3c"
    
    def predict_churn(self, customer_data: pd.DataFrame) -> Tuple[int, float]:
        """
        Make churn prediction for customer.
        
        Args:
            customer_data: Customer feature data
            
        Returns:
            Tuple of (prediction, probability)
        """
        # Preprocess data
        processed_data = self.preprocessor.transform(customer_data)
        
        # Align columns to match training data
        # Get feature names from preprocessor
        expected_features = self.preprocessor.feature_names
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        # Reorder columns to match training data
        processed_data = processed_data[expected_features]
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0][1]
        
        return prediction, probability
    
    def get_prediction_explanation(self, customer_data: pd.DataFrame) -> Dict:
        """
        Get SHAP explanation for prediction.
        
        Args:
            customer_data: Customer feature data
            
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            return None
        
        try:
            processed_data = self.preprocessor.transform(customer_data)
            
            # Align columns to match training data
            expected_features = self.preprocessor.feature_names
            for feature in expected_features:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0
            processed_data = processed_data[expected_features]
            
            explanation = self.explainer.explain_prediction(processed_data, sample_idx=0)
            return explanation
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
            return None


def create_input_form() -> pd.DataFrame:
    """Create form for customer data input."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    
    with col3:
        st.subheader("Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 0.0, 8700.0, 500.0)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    
    with col5:
        st.subheader("Online Services")
        online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
    
    with col6:
        st.subheader("Additional Services")
        tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
    
    # Create DataFrame
    customer_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    return customer_data


def display_prediction_results(app: ChurnPredictionApp, customer_data: pd.DataFrame) -> None:
    """Display prediction results and explanations."""
    
    # Make prediction
    prediction, probability = app.predict_churn(customer_data)
    risk_category, risk_color = app.get_risk_category(probability)
    
    # Display prediction
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.1%}",
            f"{(probability - 0.27):.1%}",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Prediction",
            "CHURN" if prediction == 1 else "NO CHURN",
            help="Model prediction: Will customer churn?"
        )
    
    with col3:
        st.markdown(f"""
        <div class="metric-box" style="background-color: {risk_color}; color: white;">
        <h3 style="margin-top: 0; text-align: center;">Risk Category</h3>
        <h2 style="text-align: center; margin: 0;">{risk_category}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Display interpretation
    st.markdown("---")
    st.subheader("📋 Interpretation")
    
    if probability > 0.67:
        st.error(f"""
        ⚠️ **HIGH CHURN RISK** - This customer has a high probability ({probability:.1%}) of churning.
        
        **Recommended Actions:**
        - Contact customer immediately with retention offer
        - Review service quality and satisfaction
        - Offer personalized incentives based on usage patterns
        - Consider service bundling or discounts
        """)
    elif probability > 0.33:
        st.warning(f"""
        ⚠️ **MEDIUM CHURN RISK** - This customer has a moderate probability ({probability:.1%}) of churning.
        
        **Recommended Actions:**
        - Monitor customer satisfaction metrics
        - Prepare retention campaigns
        - Optimize customer service touchpoints
        - Consider proactive engagement
        """)
    else:
        st.success(f"""
        ✅ **LOW CHURN RISK** - This customer has a low probability ({probability:.1%}) of churning.
        
        **Recommended Actions:**
        - Focus on maintaining current satisfaction
        - Continue regular check-ins
        - Monitor for any service issues
        - Consider upselling opportunities
        """)
    
    # Display SHAP explanation if available
    st.markdown("---")
    st.subheader("🔍 Feature Importance (SHAP Explanation)")
    
    explanation = app.get_prediction_explanation(customer_data)
    
    if explanation:
        # Display top features
        st.write("**Top Features Influencing This Prediction:**")
        
        top_features = explanation['feature_impact'].head(10)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['shap_value'].values]
        ax.barh(range(len(top_features)), top_features['shap_value'].values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Importance for This Prediction')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.invert_yaxis()
        
        st.pyplot(fig)
        
        # Display feature details table
        st.write("**Feature Details:**")
        display_df = top_features[['feature', 'feature_value', 'shap_value']].copy()
        display_df.columns = ['Feature', 'Customer Value', 'SHAP Impact']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("SHAP explanation not available. Please train the model with SHAP support.")
    
    # Display model information
    if app.metadata:
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Model Type:** {app.metadata.get('best_model', 'Unknown')}")
        
        with col2:
            st.info(f"**ROC-AUC Score:** {app.metadata.get('best_model_roc_auc', 'N/A'):.4f}")
        
        with col3:
            st.info(f"**Training Date:** {app.metadata.get('timestamp', 'N/A').split('T')[0]}")


def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150?text=Telco+Logo", width=150)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Prediction", "Analytics", "About"])
    
    # Main content
    st.title("📊 Customer Churn Prediction System")
    st.markdown("Predict customer churn with ML and interpretable explanations")
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = ChurnPredictionApp()
    
    app = st.session_state.app
    
    # Page navigation
    if page == "Home":
        st.markdown("""
        ## Welcome to the Customer Churn Prediction System
        
        This application uses machine learning to predict customer churn and provide interpretable explanations
        for each prediction using SHAP (SHapley Additive exPlanations).
        
        ### Features:
        - 🎯 **Accurate Predictions** - Multiple ML models trained on Telco customer data
        - 🔍 **Explainability** - SHAP-based local and global explanations
        - 📈 **Risk Assessment** - Automatic risk categorization (Low/Medium/High)
        - 💡 **Actionable Insights** - Business recommendations based on risk level
        
        ### How to Use:
        1. Go to the **Prediction** page
        2. Fill in customer information
        3. View churn prediction and risk assessment
        4. Review feature importance explanations
        
        ### About the Model:
        The model is trained on historical customer data using Logistic Regression, Random Forest, 
        and XGBoost algorithms. The best performing model is deployed for predictions.
        
        **Start by navigating to the Prediction page!**
        """)
    
    elif page == "Prediction":
        st.header("🔮 Make a Prediction")
        
        if app.model is None or app.preprocessor is None:
            st.error("❌ Model not loaded. Please train the model first.")
        else:
            st.write("Enter customer information below:")
            
            customer_data = create_input_form()
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🎯 Predict Churn", use_container_width=True):
                    display_prediction_results(app, customer_data)
    
    elif page == "Analytics":
        st.header("📈 Analytics & Insights")
        
        st.write("""
        This page displays overall analytics from the model training process.
        
        Check the `outputs/` directory for detailed visualizations:
        - EDA plots (01-08): Data exploration and patterns
        - Model comparison (09-10): Performance metrics
        - SHAP explanations (11-12): Global and local feature importance
        """)
        
        # List available outputs
        outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        if os.path.exists(outputs_dir):
            files = sorted([f for f in os.listdir(outputs_dir) if f.endswith('.png')])
            if files:
                st.write(f"**Found {len(files)} visualization files:**")
                for file in files:
                    st.write(f"- {file}")
            else:
                st.info("No visualization files found yet. Run the training pipeline first.")
        else:
            st.warning("Outputs directory not found.")
    
    elif page == "About":
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ## Customer Churn Prediction ML Project
        
        ### Problem Statement
        Customer churn is a critical metric for telecommunications companies. 
        This project aims to predict which customers are likely to churn, enabling 
        proactive retention strategies.
        
        ### Dataset
        - **Source**: IBM Telco Customer Churn Dataset
        - **Samples**: ~7,000 customer records
        - **Features**: 20 features including demographics, services, and charges
        - **Target**: Binary churn indicator (Yes/No)
        
        ### Methodology
        
        **1. Exploratory Data Analysis (EDA)**
        - Analyze churn distribution
        - Identify key features correlated with churn
        - Visualize patterns by customer segments
        
        **2. Preprocessing**
        - Handle missing values
        - Encode categorical features
        - Scale numerical features
        - Create reproducible pipeline
        
        **3. Model Training**
        - Train multiple models:
          - Logistic Regression
          - Random Forest
          - XGBoost (if available)
        - Cross-validation for robust evaluation
        - Hyperparameter tuning
        
        **4. Evaluation**
        - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
        - Select best model based on ROC-AUC
        - Generate confusion matrices
        
        **5. Explainability**
        - Use SHAP for model interpretability
        - Global feature importance
        - Local prediction explanations
        
        ### Key Insights
        - Contract type strongly influences churn
        - New customers (low tenure) have higher churn rates
        - Internet service type matters (Fiber optic has higher churn)
        - Monthly charges correlate with churn risk
        
        ### Business Impact
        - Identify high-risk customers for targeted retention
        - Understand key drivers of churn
        - Optimize service offerings based on segments
        - Reduce customer acquisition costs by improving retention
        
        ### Project Structure
        ```
        customer_churn_ml/
        ├── data/              # Datasets
        ├── src/               # Source code
        ├── models/            # Trained models
        ├── outputs/           # Visualizations
        ├── reports/           # Analysis reports
        ├── app/               # Streamlit app
        └── train_pipeline.py  # Training orchestrator
        ```
        """)
        
        st.markdown("---")
        st.write("**Built with:** Python, Scikit-learn, XGBoost, SHAP, Streamlit")


if __name__ == "__main__":
    main()
