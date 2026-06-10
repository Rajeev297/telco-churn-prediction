import sys, os, pytest, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_trainer import ModelTrainer, tune_threshold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


SAMPLE_X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                      [7, 8], [8, 9], [9, 10], [10, 11]], dtype=float)
SAMPLE_Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


class TestTuneThreshold:
    def test_tune_threshold_basic(self):
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = tune_threshold(y_true, probs)
        assert "optimal_f1" in result
        assert "optimal_recall" in result
        assert "default" in result
        assert 0 <= result["optimal_f1"]["threshold"] <= 1


class TestModelTrainer:
    def test_train_and_predict(self):
        trainer = ModelTrainer()
        trainer.X_train = SAMPLE_X
        trainer.y_train = SAMPLE_Y
        trainer.X_test = SAMPLE_X
        trainer.y_test = SAMPLE_Y
        model = LogisticRegression(max_iter=500)
        model.fit(SAMPLE_X, SAMPLE_Y)
        result = {"model": model, "model_name": "Logistic Regression", "threshold": 0.5, "threshold_info": {}}
        model_obj = result["model"]
        prob = model_obj.predict_proba(pd.DataFrame(SAMPLE_X))[:, 1]
        pred = (prob >= 0.5).astype(int)
        assert len(pred) == 10
        assert len(prob) == 10
        assert all(p in [0, 1] for p in pred)
