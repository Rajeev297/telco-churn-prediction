"""
Main Training Pipeline
Orchestrates the complete customer churn prediction workflow.
"""

import os
import sys
import logging
from datetime import datetime
import pickle

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.preprocessing import prepare_data, DataPreprocessor
from src.eda import EDAAnalyzer
from src.model_trainer import ModelTrainer, create_evaluation_plots
from src.shap_explainer import SHAPExplainer, generate_shap_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChurnPredictionPipeline:
    """Orchestrates the complete churn prediction ML pipeline."""
    
    def __init__(self, project_root: str = '.'):
        """
        Initialize pipeline.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, 'data')
        self.models_dir = os.path.join(project_root, 'models')
        self.outputs_dir = os.path.join(project_root, 'outputs')
        self.reports_dir = os.path.join(project_root, 'reports')
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.outputs_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Pipeline artifacts
        self.df_raw = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.trainer = None
        self.explainer = None
    
    def run_full_pipeline(self, tune_hyperparams: bool = True, create_plots: bool = True,
                          use_smote: bool = False) -> None:
        """
        Run the complete pipeline.
        
        Args:
            tune_hyperparams: Whether to perform hyperparameter tuning
            create_plots: Whether to create visualization plots
            use_smote: Whether to apply SMOTE for class balancing
        """
        logger.info("="*70)
        logger.info("STARTING CUSTOMER CHURN PREDICTION PIPELINE")
        logger.info("="*70)
        
        try:
            # Step 1: Load data
            logger.info("\n[STEP 1/6] LOADING DATA")
            self._load_data()
            
            # Step 2: Exploratory Data Analysis
            if create_plots:
                logger.info("\n[STEP 2/6] PERFORMING EDA")
                self._perform_eda()
            
            # Step 3: Prepare data (preprocessing)
            logger.info("\n[STEP 3/6] PREPARING DATA")
            self._use_smote = use_smote
            self._prepare_data()
            
            # Step 4: Train models
            logger.info("\n[STEP 4/6] TRAINING MODELS")
            self._train_models(tune_hyperparams)
            
            # Step 5: Evaluate models
            logger.info("\n[STEP 5/6] EVALUATING MODELS")
            self._evaluate_models(create_plots)
            
            # Step 6: Generate SHAP explanations
            logger.info("\n[STEP 6/6] GENERATING SHAP EXPLANATIONS")
            self._generate_explanations(create_plots)
            
            # Save artifacts
            self._save_artifacts()
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
    
    def _load_data(self) -> None:
        """Load dataset."""
        logger.info("Loading customer churn dataset...")
        
        loader = DataLoader()
        self.df_raw = loader.load_sample_data()
        
        # Save raw data
        csv_path = os.path.join(self.data_dir, 'telco_customer_churn_raw.csv')
        self.df_raw.to_csv(csv_path, index=False)
        logger.info(f"Raw data saved to {csv_path}")
        logger.info(f"Dataset shape: {self.df_raw.shape}")
    
    def _perform_eda(self) -> None:
        """Perform exploratory data analysis."""
        logger.info("Running EDA analysis...")
        
        analyzer = EDAAnalyzer(output_dir=self.outputs_dir)
        report = analyzer.run_all_analyses(self.df_raw)
        
        # Save EDA report
        eda_report_path = os.path.join(self.reports_dir, 'eda_summary.txt')
        with open(eda_report_path, 'w') as f:
            f.write(report)
        logger.info(f"EDA report saved to {eda_report_path}")
    
    def _prepare_data(self) -> None:
        """Prepare data for modeling."""
        logger.info("Preparing data for modeling...")
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor = prepare_data(
            self.df_raw,
            test_size=0.2,
            random_state=42,
            use_smote=self._use_smote if hasattr(self, '_use_smote') else False
        )
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def _train_models(self, tune_hyperparams: bool = True) -> None:
        """Train machine learning models."""
        logger.info("Training machine learning models...")
        
        self.trainer = ModelTrainer(output_dir=self.models_dir)
        self.trainer.train_all_models(self.X_train, self.y_train, tune_hyperparams=tune_hyperparams)
        
        logger.info("Model training completed")
    
    def _evaluate_models(self, create_plots: bool = True) -> None:
        """Evaluate trained models."""
        logger.info("Evaluating models...")
        
        self.trainer.evaluate_all_models(self.X_test, self.y_test)
        self.trainer.select_best_model()
        
        # Generate evaluation report
        report = self.trainer.generate_evaluation_report()
        eval_report_path = os.path.join(self.reports_dir, 'model_evaluation.txt')
        with open(eval_report_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to {eval_report_path}")
        
        # Save best model
        best_model_path = self.trainer.save_best_model()
        logger.info(f"Best model saved to {best_model_path}")
        
        # Create comparison plots
        if create_plots:
            create_evaluation_plots(self.trainer.results, output_dir=self.outputs_dir)
            from src.model_trainer import generate_threshold_comparison_plot
            generate_threshold_comparison_plot(self.trainer.results, output_dir=self.outputs_dir)
    
    def _generate_explanations(self, create_plots: bool = True) -> None:
        """Generate SHAP-based model explanations."""
        logger.info("Generating SHAP explanations...")
        
        try:
            # Use pre-SMOTE training data for SHAP background if SMOTE was applied
            shap_background = self.X_train.iloc[:200]
            if hasattr(self.preprocessor, 'pre_smote_X') and self.preprocessor.pre_smote_X is not None:
                pre_smote = self.preprocessor.pre_smote_X.drop(columns=['Churn'], errors='ignore')
                shap_background = pre_smote.iloc[:200]
                logger.info("Using pre-SMOTE data for SHAP background sampling")

            self.explainer = SHAPExplainer(
                self.trainer.best_model,
                shap_background,
                feature_names=self.X_train.columns.tolist()
            )
            
            # Generate report
            report = generate_shap_report(self.explainer, self.X_test.iloc[:100], self.outputs_dir)
            
            shap_report_path = os.path.join(self.reports_dir, 'shap_explainability.txt')
            with open(shap_report_path, 'w') as f:
                f.write(report)
            logger.info(f"SHAP report saved to {shap_report_path}")
            
            # Save explainer
            explainer_path = os.path.join(self.models_dir, 'shap_explainer.pkl')
            self.explainer.save_explainer(explainer_path)
            logger.info(f"SHAP explainer saved to {explainer_path}")
        
        except Exception as e:
            logger.warning(f"SHAP explanation generation failed: {e}")
            logger.info("Continuing pipeline without SHAP explanations")
    
    def _save_artifacts(self) -> None:
        """Save pipeline artifacts and metadata."""
        logger.info("Saving pipeline artifacts...")
        
        best_metrics = self.trainer.results[self.trainer.best_model_name]['test_metrics']
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.trainer.best_model_name,
            'best_model_roc_auc': best_metrics['roc_auc'],
            'best_model_recall': best_metrics['recall'],
            'best_model_precision': best_metrics['precision'],
            'best_model_f1': best_metrics['f1'],
            'best_model_accuracy': best_metrics['accuracy'],
            'optimal_threshold': best_metrics.get('optimal_threshold', 0.5),
            'optimal_recall': best_metrics.get('optimal_recall', best_metrics['recall']),
            'optimal_precision': best_metrics.get('optimal_precision', best_metrics['precision']),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'n_features': len(self.X_train.columns),
            'feature_names': self.X_train.columns.tolist(),
            'use_smote': getattr(self, '_use_smote', False),
        }
        
        metadata_path = os.path.join(self.models_dir, 'pipeline_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Pipeline metadata saved to {metadata_path}")
        
        # Create summary
        summary = f"""
CUSTOMER CHURN PREDICTION PIPELINE - SUMMARY
{'='*70}

Pipeline Execution Date: {metadata['timestamp']}

BEST MODEL: {metadata['best_model']}
- ROC-AUC Score:        {metadata['best_model_roc_auc']:.4f}
- Test Recall:          {metadata['best_model_recall']:.4f}  (default threshold 0.50)
- Test Precision:       {metadata['best_model_precision']:.4f}
- Test F1 Score:        {metadata['best_model_f1']:.4f}
- Test Accuracy:        {metadata['best_model_accuracy']:.4f}

THRESHOLD TUNING:
- Optimal Threshold:    {metadata['optimal_threshold']:.2f}
- Recall @ Optimal:     {metadata['optimal_recall']:.4f}
- Precision @ Optimal:  {metadata['optimal_precision']:.4f}

DATA STATISTICS:
- Training samples: {metadata['train_size']}
- Test samples: {metadata['test_size']}
- Total features: {metadata['n_features']}
- SMOTE applied: {metadata['use_smote']}

OUTPUTS GENERATED:
- EDA visualizations: outputs/01_*.png - outputs/08_*.png
- Model comparison plots: outputs/09_*.png, outputs/10_*.png
- Threshold tuning curves: outputs/13_threshold_tuning.png
- SHAP explanations: outputs/11_*.png, outputs/12_*.png
- Model files: models/*.pkl
- Reports: reports/*.txt

{'='*70}
"""
        
        summary_path = os.path.join(self.reports_dir, 'pipeline_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
        logger.info(f"Pipeline summary saved to {summary_path}")


def main():
    """Main function to execute the pipeline."""
    # Get project root from command line or use current directory
    project_root = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # Check if directories exist
    if not os.path.exists(os.path.join(project_root, 'src')):
        logger.error(f"Project structure not found in {project_root}")
        sys.exit(1)
    
    # Run pipeline (tuning enabled, SMOTE enabled for balanced training)
    pipeline = ChurnPredictionPipeline(project_root=project_root)
    pipeline.run_full_pipeline(tune_hyperparams=True, create_plots=True, use_smote=True)


if __name__ == "__main__":
    main()
