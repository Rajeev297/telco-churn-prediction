"""
Model Training and Evaluation Module
Trains, evaluates, and compares multiple ML models optimized for recall and ROC-AUC.
Supports class weighting, SMOTE, threshold tuning, and LightGBM.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and manages multiple ML models with recall/ROC-AUC optimization."""

    def __init__(self, output_dir: str = 'models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = 0.5

    def _compute_scale_pos_weight(self, y_train: pd.Series) -> float:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        return neg / pos if pos > 0 else 1.0

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  tune_hyperparams: bool = True) -> Tuple[LogisticRegression, Dict]:
        logger.info("Training Logistic Regression...")
        if tune_hyperparams:
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'max_iter': [200, 500],
                'class_weight': [None, 'balanced'],
            }
            base = LogisticRegression(random_state=42, solver='lbfgs')
            grid = GridSearchCV(base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
        else:
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return model, {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores}

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            tune_hyperparams: bool = True) -> Tuple[RandomForestClassifier, Dict]:
        logger.info("Training Random Forest...")
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': [None, 'balanced'],
            }
            base = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid = GridSearchCV(base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return model, {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores}

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      tune_hyperparams: bool = True) -> Tuple[Optional[Any], Dict]:
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available. Skipping.")
            return None, {}
        logger.info("Training XGBoost...")
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'scale_pos_weight': [1, scale_pos_weight],
            }
            base = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            grid = GridSearchCV(base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
        else:
            model = xgb.XGBClassifier(
                n_estimators=100, random_state=42, use_label_encoder=False,
                eval_metric='logloss', scale_pos_weight=scale_pos_weight
            )
            model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return model, {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores}

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       tune_hyperparams: bool = True) -> Tuple[Optional[Any], Dict]:
        if not LGB_AVAILABLE:
            logger.warning("LightGBM not available. Skipping.")
            return None, {}
        logger.info("Training LightGBM...")
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [15, 31],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1, scale_pos_weight],
            }
            base = lgb.LGBMClassifier(random_state=42, verbose=-1)
            grid = GridSearchCV(base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
        else:
            model = lgb.LGBMClassifier(
                n_estimators=100, random_state=42, verbose=-1,
                scale_pos_weight=scale_pos_weight
            )
            model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return model, {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores}

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         tune_hyperparams: bool = True) -> None:
        logger.info("=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)

        lr_model, lr_scores = self.train_logistic_regression(X_train, y_train, tune_hyperparams)
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = {'cv_scores': lr_scores}

        rf_model, rf_scores = self.train_random_forest(X_train, y_train, tune_hyperparams)
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {'cv_scores': rf_scores}

        if XGB_AVAILABLE:
            xgb_model, xgb_scores = self.train_xgboost(X_train, y_train, tune_hyperparams)
            self.models['XGBoost'] = xgb_model
            self.results['XGBoost'] = {'cv_scores': xgb_scores}

        if LGB_AVAILABLE:
            lgb_model, lgb_scores = self.train_lightgbm(X_train, y_train, tune_hyperparams)
            self.models['LightGBM'] = lgb_model
            self.results['LightGBM'] = {'cv_scores': lgb_scores}

        logger.info("All models trained successfully!")

    def tune_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        return tune_threshold(y_true, y_prob)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        results.append({
            'threshold': t,
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        })
    df_r = pd.DataFrame(results)

    best_f1_idx = df_r['f1'].idxmax()
    optimal_f1 = df_r.iloc[best_f1_idx].to_dict()

    mask = df_r['precision'] >= 0.40
    if mask.any():
        best_recall_idx = df_r[mask]['recall'].idxmax()
        optimal_recall = df_r.iloc[best_recall_idx].to_dict()
    else:
        optimal_recall = optimal_f1

    default = df_r[df_r['threshold'] == 0.50]
    default_metrics = default.iloc[0].to_dict() if len(default) > 0 else optimal_f1

    logger.info(f"Threshold tuning - Optimal for recall: {optimal_recall['threshold']:.2f} "
                f"(recall={optimal_recall['recall']:.4f}, precision={optimal_recall['precision']:.4f})")
    logger.info(f"Threshold tuning - Optimal for F1: {optimal_f1['threshold']:.2f} "
                f"(recall={optimal_f1['recall']:.4f}, f1={optimal_f1['f1']:.4f})")

    return {
        'curve': df_r,
        'optimal_f1': optimal_f1,
        'optimal_recall': optimal_recall,
        'default': default_metrics,
    }

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str) -> Dict:
        logger.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'y_pred': y_pred,
            'y_pred_proba': y_prob,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }

        threshold_results = self.tune_threshold(y_test, y_prob)
        metrics['threshold_results'] = threshold_results
        metrics['optimal_threshold'] = threshold_results['optimal_recall']['threshold']
        metrics['optimal_recall'] = threshold_results['optimal_recall']['recall']
        metrics['optimal_precision'] = threshold_results['optimal_recall']['precision']

        logger.info(f"Results for {model_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Optimal threshold: {metrics['optimal_threshold']:.2f} "
                    f"(recall={metrics['optimal_recall']:.4f}, precision={metrics['optimal_precision']:.4f})")

        return metrics

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        logger.info("=" * 70)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 70)
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            self.results[model_name]['test_metrics'] = metrics
        return self.results

    def select_best_model(self) -> Tuple[str, Any]:
        logger.info("Selecting best model based on ROC-AUC...")
        best_roc_auc = -1
        best_model_name = None
        for model_name, result in self.results.items():
            if 'test_metrics' in result:
                roc_auc = result['test_metrics']['roc_auc']
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_name = model_name
        if best_model_name:
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            self.best_threshold = self.results[best_model_name]['test_metrics'].get('optimal_threshold', 0.5)
            logger.info(f"Best model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
            logger.info(f"  Optimal threshold: {self.best_threshold:.2f}")
            logger.info(f"  Recall at optimal threshold: {self.results[best_model_name]['test_metrics'].get('optimal_recall', 'N/A'):.4f}")
        return best_model_name, self.best_model

    def save_model(self, model_data: Dict, model_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(' ', '_')}_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
        return filepath

    def save_best_model(self) -> str:
        if self.best_model is None:
            raise ValueError("No best model selected. Run evaluate_all_models first.")
        metrics = self.results[self.best_model_name]['test_metrics']
        model_data = {
            'model': self.best_model,
            'threshold': self.best_threshold,
            'model_name': self.best_model_name,
            'threshold_info': {
                'optimal_f1': metrics.get('threshold_results', {}).get('optimal_f1', {}),
                'optimal_recall': metrics.get('threshold_results', {}).get('optimal_recall', {}),
            },
        }
        return self.save_model(model_data, self.best_model_name)

    def generate_evaluation_report(self) -> str:
        report = "\n" + "=" * 80 + "\n"
        report += "MODEL EVALUATION REPORT (Recall & ROC-AUC Focus)\n"
        report += "=" * 80 + "\n\n"

        for model_name, result in self.results.items():
            report += f"\n{model_name}:\n"
            report += "-" * 60 + "\n"
            if 'cv_scores' in result and result['cv_scores']:
                report += f"  Cross-Val ROC-AUC: {result['cv_scores']['mean']:.4f} (+/- {result['cv_scores']['std']:.4f})\n"
            if 'test_metrics' in result:
                m = result['test_metrics']
                report += f"  Test Accuracy:        {m['accuracy']:.4f}\n"
                report += f"  Test Precision:       {m['precision']:.4f}\n"
                report += f"  Test Recall:          {m['recall']:.4f}\n"
                report += f"  Test F1 Score:        {m['f1']:.4f}\n"
                report += f"  Test ROC-AUC:         {m['roc_auc']:.4f}\n"
                report += f"  Optimal Threshold:    {m.get('optimal_threshold', 0.5):.2f}\n"
                report += f"  Recall@OptimalThresh: {m.get('optimal_recall', 'N/A'):.4f}\n"
                report += f"  Precision@OptThresh:  {m.get('optimal_precision', 'N/A'):.4f}\n"

        report += "\n" + "=" * 80 + "\n"
        report += f"BEST MODEL: {self.best_model_name}\n"
        report += "=" * 80 + "\n"

        if self.best_model_name:
            bm = self.results[self.best_model_name]['test_metrics']
            report += f"\nRECOMMENDATION:\n"
            report += f"  The best model is {self.best_model_name} with ROC-AUC of {bm['roc_auc']:.4f}.\n"
            report += f"  Using the optimal threshold of {bm.get('optimal_threshold', 0.5):.2f},\n"
            report += f"  recall improves from {bm['recall']:.4f} to {bm.get('optimal_recall', 'N/A'):.4f},\n"
            report += f"  while precision adjusts from {bm['precision']:.4f} to {bm.get('optimal_precision', 'N/A'):.4f}.\n"
            report += f"  This configuration captures more at-risk customers for proactive retention.\n"

        return report


def create_evaluation_plots(results: Dict[str, Dict], output_dir: str = 'outputs') -> None:
    os.makedirs(output_dir, exist_ok=True)

    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    opt_recalls = []

    for model_name, result in results.items():
        if 'test_metrics' in result:
            models.append(model_name)
            m = result['test_metrics']
            accuracies.append(m['accuracy'])
            precisions.append(m['precision'])
            recalls.append(m['recall'])
            f1_scores.append(m['f1'])
            roc_aucs.append(m['roc_auc'])
            opt_recalls.append(m.get('optimal_recall', m['recall']))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].bar(models, accuracies, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)

    axes[0, 1].bar(models, precisions, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Precision', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)

    axes[0, 2].bar(models, recalls, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Recall', fontsize=12, fontweight='bold')
    axes[0, 2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(recalls):
        axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)

    axes[1, 0].bar(models, f1_scores, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_scores):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)

    axes[1, 1].bar(models, roc_aucs, color='#f39c12', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('ROC-AUC', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(roc_aucs):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)

    axes[1, 2].bar(models, opt_recalls, color='#e67e22', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Recall @ Optimal Threshold', fontsize=12, fontweight='bold')
    axes[1, 2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(opt_recalls):
        axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Model comparison plot saved to {output_dir}/09_model_comparison.png")

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for idx, (model_name, result) in enumerate(results.items()):
        if 'test_metrics' in result:
            cm = result['test_metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(f'Confusion Matrix\n{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrices saved to {output_dir}/10_confusion_matrices.png")


def generate_threshold_comparison_plot(results: Dict[str, Dict], output_dir: str = 'outputs') -> None:
    """Plot threshold tuning curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    idx = 0
    for model_name, result in results.items():
        if idx >= 4:
            break
        if 'test_metrics' in result and 'threshold_results' in result['test_metrics']:
            tr = result['test_metrics']['threshold_results']
            df_c = tr['curve']
            ax = axes_flat[idx]
            ax.plot(df_c['threshold'], df_c['recall'], label='Recall', color='#2ecc71', linewidth=2)
            ax.plot(df_c['threshold'], df_c['precision'], label='Precision', color='#e74c3c', linewidth=2)
            ax.plot(df_c['threshold'], df_c['f1'], label='F1', color='#3498db', linewidth=2, linestyle='--')
            ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default 0.5')
            ax.axvline(x=tr['optimal_recall']['threshold'], color='green', linestyle='--', alpha=0.7,
                       label=f"Optimal {tr['optimal_recall']['threshold']:.2f}")
            ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            idx += 1
    plt.suptitle('Threshold Tuning Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_threshold_tuning.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Threshold tuning plot saved to {output_dir}/13_threshold_tuning.png")


if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import prepare_data

    loader = DataLoader()
    df = loader.load_sample_data()

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    trainer = ModelTrainer(output_dir='models')
    trainer.train_all_models(X_train, y_train, tune_hyperparams=True)
    trainer.evaluate_all_models(X_test, y_test)
    trainer.select_best_model()

    report = trainer.generate_evaluation_report()
    print(report)
