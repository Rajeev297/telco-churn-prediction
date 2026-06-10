# Customer Churn Prediction - ML Project

A complete machine learning project for predicting customer churn in telecom using Scikit-learn, XGBoost, SHAP, and Streamlit.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Documentation](#documentation)

## 🎯 Overview

This project builds a machine learning system to predict which customers are likely to churn (discontinue service) in a telecommunications company. The system includes:

- **Data preprocessing** with validation and encoding pipelines
- **Exploratory data analysis** with comprehensive visualizations
- **Multiple ML models** (Logistic Regression, Random Forest, XGBoost)
- **Model evaluation** with multiple metrics
- **SHAP-based explainability** for model interpretability
- **Interactive Streamlit dashboard** for predictions and explanations

## ✨ Features

- ✅ **Reproducible Pipeline**: Automated end-to-end ML workflow
- ✅ **Data Validation**: Comprehensive data quality checks
- ✅ **EDA Visualizations**: 8+ detailed analysis plots
- ✅ **Model Comparison**: Train and compare 3+ ML models
- ✅ **SHAP Explanations**: Global and local feature importance
- ✅ **Risk Assessment**: Automatic customer risk categorization (Low/Medium/High)
- ✅ **Interactive Web App**: Streamlit-based prediction interface
- ✅ **Bulk CSV Prediction**: Upload thousands of customers for batch analysis
- ✅ **Executive Dashboard**: KPI cards, risk distribution charts, contract/internet analysis
- ✅ **Model Performance Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC (when ground truth provided)
- ✅ **Downloadable Results**: Export predictions CSV with risk levels
- ✅ **Business Insights Engine**: Auto-generated retention recommendations
- ✅ **Production-Ready Code**: Clean, modular, well-documented

## 📁 Project Structure

```
customer_churn_ml/
│
├── data/                          # Datasets
│   └── telco_customer_churn_raw.csv
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading
│   ├── preprocessing.py          # Data preprocessing pipeline
│   ├── eda.py                    # Exploratory data analysis
│   ├── model_trainer.py          # Model training & evaluation
│   └── shap_explainer.py         # SHAP explainability
│
├── models/                        # Trained models & artifacts
│   ├── logistic_regression_*.pkl
│   ├── random_forest_*.pkl
│   ├── xgboost_*.pkl
│   ├── preprocessor.pkl          # Preprocessing pipeline
│   ├── shap_explainer.pkl        # SHAP explainer
│   └── pipeline_metadata.pkl     # Pipeline metadata
│
├── outputs/                       # Generated visualizations
│   ├── 01_churn_distribution.png
│   ├── 02_tenure_analysis.png
│   ├── 03_monthly_charges_analysis.png
│   ├── 04_contract_analysis.png
│   ├── 05_internet_service_analysis.png
│   ├── 06_demographic_analysis.png
│   ├── 07_services_analysis.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_model_comparison.png
│   ├── 10_confusion_matrices.png
│   ├── 11_shap_feature_importance.png
│   └── 12_shap_prediction_explanation.png
│
├── reports/                       # Generated reports
│   ├── eda_summary.txt           # EDA insights
│   ├── model_evaluation.txt      # Model performance
│   ├── shap_explainability.txt   # SHAP analysis
│   └── pipeline_summary.txt      # Pipeline overview
│
├── app/                           # Streamlit applications
│   ├── streamlit_app.py          # Original interactive web app (tech-oriented)
│   └── client_app.py             # Client-facing Churn Analytics Platform
│
├── train_pipeline.py             # Main training script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── project_report.md            # Detailed project report
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone or Download Project

```bash
cd customer_churn_ml
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n churn-ml python=3.8
conda activate churn-ml
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with XGBoost installation:

```bash
# Windows
pip install xgboost

# Mac (may need additional dependencies)
pip install xgboost
```

## 💻 Usage

### 1. Train the Model

Run the complete training pipeline:

```bash
python train_pipeline.py
```

This will:
- Download the dataset
- Perform EDA and generate visualizations
- Preprocess data
- Train multiple ML models
- Evaluate and select the best model
- Generate SHAP explanations
- Save all artifacts

**Expected output:**
- Models saved in `models/`
- Visualizations saved in `outputs/`
- Reports saved in `reports/`

### 2. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- 📊 Home: Overview and instructions
- 🔮 Prediction: Enter customer data and get churn prediction with risk assessment
- 📈 Analytics: View training visualizations
- ℹ️ About: Project details and methodology

### 3. Run the Client-Facing Analytics Platform

```bash
streamlit run app/client_app.py --server.port 8502
```

A business-ready churn analytics dashboard opens at `http://localhost:8502`.

### 4. Bulk CSV Prediction Workflow

1. **Prepare a CSV** with the 19 required feature columns (download the sample template from the app)
2. **Upload** the CSV in Tab 2 (Bulk CSV Analysis)
3. **Validate**: The app checks for missing columns, invalid values, negative tenure/charges
4. **Run Prediction**: Click "Run Bulk Prediction" — processes data in batches of 1000 with a progress bar
5. **Review Results**: Interactive table with Churn Probability, Predicted Churn, and Risk Level (Low/Medium/High)
6. **Download**: Export results as CSV with all original data plus predictions
7. **Dashboard**: View KPI metrics, risk distribution charts, SHAP global importance, and business insights

#### CSV Format Requirements

| Column | Type | Valid Values |
|--------|------|-------------|
| `gender` | text | Male, Female |
| `SeniorCitizen` | int | 0, 1 |
| `Partner` | text | Yes, No |
| `Dependents` | text | Yes, No |
| `tenure` | int | 0–72 |
| `PhoneService` | text | Yes, No |
| `MultipleLines` | text | Yes, No, No phone service |
| `InternetService` | text | DSL, Fiber optic, No |
| `OnlineSecurity` | text | Yes, No, No internet service |
| `OnlineBackup` | text | Yes, No, No internet service |
| `DeviceProtection` | text | Yes, No, No internet service |
| `TechSupport` | text | Yes, No, No internet service |
| `StreamingTV` | text | Yes, No, No internet service |
| `StreamingMovies` | text | Yes, No, No internet service |
| `Contract` | text | Month-to-month, One year, Two year |
| `PaperlessBilling` | text | Yes, No |
| `PaymentMethod` | text | Electronic check, Mailed check, Bank transfer, Credit card |
| `MonthlyCharges` | float | 18.0–120.0 |
| `TotalCharges` | float | 0–8684.0 |

Optional ground truth column for model performance metrics:

| Column | Type | Valid Values |
|--------|------|-------------|
| `Churn` | text | Yes, No |

#### Risk Level Thresholds

| Risk Level | Probability | Color | Action |
|------------|-----------|-------|--------|
| **Low Risk** | 0–30% | Green (#34d399) | Monitor, loyalty rewards |
| **Medium Risk** | 30–70% | Yellow (#fbbf24) | Proactive outreach, retention incentive |
| **High Risk** | 70–100% | Red (#f87171) | Immediate retention specialist |

#### Executive Dashboard

After bulk prediction, the dashboard displays:

- **6 KPI Cards**: Total Customers, High/Medium/Low Risk counts, Avg Churn Probability, Estimated Churn Rate
- **Model Performance Metrics** (if CSV includes `Churn` column): Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **4 Interactive Charts**: Risk Distribution (pie), Churn Probability Distribution (histogram), Contract Type vs Risk (stacked bar), Internet Service vs Risk (stacked bar)
- **Global Feature Importance**: SHAP-based top 15 churn drivers across the uploaded dataset
- **Business Insights**: Auto-generated retention recommendations and risk analysis

### 5. Make Individual Predictions

```python
# Example: Making predictions programmatically
from src.data_loader import DataLoader
from src.preprocessing import prepare_data
from src.model_trainer import ModelTrainer
import pickle

# Load artifacts
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/<best_model>.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare customer data
customer_data = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    # ... other features
})

# Preprocess and predict
processed_data = preprocessor.transform(customer_data)
prediction = model.predict(processed_data)
probability = model.predict_proba(processed_data)[0][1]

print(f"Churn Prediction: {prediction}")
print(f"Churn Probability: {probability:.2%}")
```

## 📊 Dataset

### IBM Telco Customer Churn Dataset

**Source**: Automatically downloaded from public repository

**Statistics:**
- **Total Records**: ~7,000 customers
- **Total Features**: 20
- **Target Variable**: Churn (Yes/No)
- **Class Distribution**: ~27% Churn, ~73% No Churn

**Features:**
- Demographics: gender, age, partner status, dependents
- Account: tenure, contract type, billing method
- Services: internet, phone, online security, tech support, etc.
- Charges: monthly and total charges

### Data Preprocessing

The preprocessing pipeline includes:

1. **Missing Value Handling**: Mean/median imputation for numerical, mode for categorical
2. **Categorical Encoding**: One-hot encoding for categorical features
3. **Feature Scaling**: StandardScaler for numerical features
4. **Binary Encoding**: Target variable converted to 0/1
5. **Train-Test Split**: 80-20 split with stratification

All preprocessing is fitted on training data and applied consistently to test data.

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

**Analyses Performed:**
- Churn distribution (overall churn rate: ~27%)
- Tenure analysis (new customers show higher churn)
- Monthly charges analysis (higher charges → higher churn)
- Contract type impact (month-to-month highest risk)
- Internet service type (fiber optic highest risk)
- Demographic patterns (senior citizens, partners, dependents)
- Service adoption (online security reduces churn)
- Correlations between features

**Output**: 8 detailed visualizations + insights summary

### 2. Model Training

**Models Trained:**

1. **Logistic Regression**
   - Simple, interpretable baseline
   - Linear decision boundaries
   - Fast training and prediction

2. **Random Forest**
   - Ensemble of decision trees
   - Captures non-linear patterns
   - Feature importance built-in

3. **XGBoost**
   - Gradient boosting framework
   - State-of-the-art performance
   - Handles class imbalance well

**Training Process:**
- 5-fold cross-validation for robust evaluation
- Hyperparameter tuning with GridSearchCV
- Train-test split (80-20) with stratification
- Evaluation on held-out test set

### 3. Model Evaluation

**Metrics Calculated:**

| Metric | Description | Best Value |
|--------|-------------|-----------|
| Accuracy | Correctly classified / Total | 1.0 |
| Precision | TP / (TP + FP) | 1.0 |
| Recall | TP / (TP + FN) | 1.0 |
| F1 Score | Harmonic mean of Precision & Recall | 1.0 |
| ROC-AUC | Area under ROC curve | 1.0 |

**Selection Criterion**: Best model selected based on ROC-AUC score (best for imbalanced data)

### 4. Model Explainability (SHAP)

**SHAP Analysis Includes:**

- **Global Importance**: Which features matter most overall?
- **Local Explanations**: Why was this prediction made?
- **Feature Impact**: How does each feature push prediction up/down?
- **Contribution Plots**: Visual breakdown of prediction components

**Benefits:**
- Understand model decisions
- Build stakeholder trust
- Identify model biases
- Comply with regulatory requirements

## 📈 Model Performance

Expected performance metrics (typical results):

```
Model Comparison:
┌─────────────────────┬──────────┬───────────┬────────┬──────────┬─────────┐
│ Model               │ Accuracy │ Precision │ Recall │ F1 Score │ ROC-AUC │
├─────────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ Logistic Regression │  0.800   │   0.650   │ 0.520  │  0.578   │ 0.848   │
│ Random Forest       │  0.820   │  0.720    │ 0.580  │  0.642   │ 0.875   │
│ XGBoost (BEST)      │  0.835   │  0.755    │ 0.635  │  0.688   │ 0.892   │
└─────────────────────┴──────────┴───────────┴────────┴──────────┴─────────┘
```

## 🎨 Deployment

### Running the Applications

Two Streamlit apps are available for different audiences:

#### Tech-Oriented App (Original)

```bash
streamlit run app/streamlit_app.py   # http://localhost:8501
```

Features: Prediction form, SHAP explanations, training analytics, EDA visualizations.

#### Client-Facing Analytics Platform

```bash
streamlit run app/client_app.py --server.port 8502  # http://localhost:8502
```

Features: Tab 1 — Single Customer Prediction with SHAP; Tab 2 — Bulk CSV upload, prediction, executive dashboard, SHAP global importance, business insights, downloadable results.

### Risk Categories

| Risk Level | Probability Range | Color | Recommendation |
|-----------|------------------|-------|-----------------|
| **LOW** | 0–30% | Green | Monitor, loyalty rewards, upsell opportunities |
| **MEDIUM** | 30–70% | Yellow | Proactive outreach, retention incentives |
| **HIGH** | 70–100% | Red | Immediate retention specialist intervention |

## 📚 Documentation

### File Descriptions

#### `src/data_loader.py`
- **Purpose**: Load and validate data
- **Classes**: `DataLoader`
- **Key Methods**: `load_sample_data()`, `load_from_csv()`

#### `src/preprocessing.py`
- **Purpose**: Data preprocessing and validation
- **Classes**: `DataValidator`, `DataPreprocessor`
- **Key Methods**: `fit_transform()`, `transform()`, `handle_missing_values()`

#### `src/eda.py`
- **Purpose**: Exploratory data analysis
- **Classes**: `EDAAnalyzer`
- **Key Methods**: `run_all_analyses()`, `analyze_churn_distribution()`

#### `src/model_trainer.py`
- **Purpose**: Model training and evaluation
- **Classes**: `ModelTrainer`
- **Key Methods**: `train_all_models()`, `evaluate_all_models()`, `select_best_model()`

#### `src/shap_explainer.py`
- **Purpose**: SHAP-based model explainability
- **Classes**: `SHAPExplainer`
- **Key Methods**: `explain_prediction()`, `plot_feature_importance()`

#### `train_pipeline.py`
- **Purpose**: Orchestrate the complete ML workflow
- **Classes**: `ChurnPredictionPipeline`
- **Key Methods**: `run_full_pipeline()`

#### `app/streamlit_app.py`
- **Purpose**: Original interactive web application for predictions
- **Features**: Prediction form, SHAP explanations, risk assessment, analytics page

#### `app/client_app.py`
- **Purpose**: Business-ready Churn Analytics Platform
- **Features**: Single customer prediction with SHAP, bulk CSV upload & validation, executive dashboard with Plotly charts, global SHAP importance, business insights engine, model performance metrics (Accuracy/Precision/Recall/F1/ROC-AUC), CSV template download, results export

### Reports Generated

1. **eda_summary.txt**: Key insights from exploratory analysis
2. **model_evaluation.txt**: Detailed model performance metrics
3. **shap_explainability.txt**: Feature importance analysis
4. **pipeline_summary.txt**: Overall pipeline summary

## 🔧 Configuration & Customization

### Adjusting Hyperparameters

Edit `src/model_trainer.py`:

```python
# Random Forest parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}
```

### Changing Train-Test Split

Edit `src/preprocessing.py`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # Change this value
    random_state=42
)
```

### Modifying SHAP Samples

Edit `src/shap_explainer.py`:

```python
explainer = SHAPExplainer(
    model,
    X_train.iloc[:200],  # Change number of background samples
    feature_names=feature_names
)
```

## 🎓 Key Insights from Analysis

### Top Churn Drivers:

1. **Contract Type**: Month-to-month contracts have 42% churn rate
2. **Tenure**: New customers (0-12 months) have 50% churn rate
3. **Internet Service**: Fiber optic customers have 42% churn rate
4. **Total Charges**: Inversely correlated with churn
5. **Online Security**: Customers with this service have 15% churn rate
6. **Tech Support**: Reduces churn from 27% to 17%

### Business Recommendations:

1. **Promote Long-term Contracts**: Offer incentives for annual/2-year contracts
2. **Focus on New Customer Retention**: Implement onboarding programs
3. **Improve Fiber Optic Experience**: Address service quality issues
4. **Bundle Services**: Include security and support to reduce churn
5. **Proactive Support**: Contact high-risk customers before they leave

## 🧪 Testing & Validation

### Validating Results

1. **Check Output Files**:
   ```bash
   ls -la outputs/        # Should have 12 PNG files
   ls -la models/         # Should have .pkl files
   ls -la reports/        # Should have 4 text files
   ```

2. **Review Visualizations**:
   - Open PNG files in outputs/ to verify quality
   - Check reports/ for detailed insights

3. **Test Predictions**:
   ```bash
   streamlit run app/streamlit_app.py
   # Enter test data and verify reasonable predictions
   ```

## 🐛 Troubleshooting

### Issue: XGBoost Installation Fails
**Solution**:
```bash
pip install --upgrade setuptools wheel
pip install xgboost
```

### Issue: SHAP Calculation is Slow
**Solution**: Reduce number of background samples in `shap_explainer.py`

### Issue: Streamlit App Not Loading Models
**Solution**: Ensure training pipeline was completed successfully:
```bash
python train_pipeline.py
```

### Issue: Out of Memory Errors
**Solution**: Process smaller batches or reduce training data:
```python
# In train_pipeline.py
self.df_raw = self.df_raw.sample(n=5000)  # Use 5000 samples instead
```

## 📝 License

This project is open-source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Optimize code
- Add new features

## 📧 Questions & Support

For issues or questions:
1. Check the troubleshooting section
2. Review project_report.md for detailed methodology
3. Inspect log files: `training_pipeline.log`

## 🎉 Summary

This complete ML project demonstrates:

✅ **Production-ready code** with proper structure and documentation  
✅ **Comprehensive pipeline** from data loading to deployment  
✅ **Model explainability** using SHAP for transparency  
✅ **Interactive interface** for real-time predictions  
✅ **Best practices** in ML development and evaluation  
✅ **Business value** with actionable insights  

Perfect for learning ML development or deploying in production!

---

**Happy Predicting! 🚀**
