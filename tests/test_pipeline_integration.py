import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, get


class TestPipelineIntegration:
    def test_config_loaded(self):
        cfg = load_config()
        assert get("model.threshold") is not None

    def test_models_dir_exists(self):
        assert os.path.isdir(get("paths.models_dir"))

    def test_model_file_present(self):
        models_dir = get("paths.models_dir")
        files = [f for f in os.listdir(models_dir) if f.endswith(".pkl") and "preprocessor" not in f and "explainer" not in f and "metadata" not in f]
        assert len(files) >= 1

    def test_preprocessor_present(self):
        assert os.path.isfile(os.path.join(get("paths.models_dir"), "preprocessor.pkl"))

    def test_evaluation_report_present(self):
        path = os.path.join(get("paths.reports_dir"), "model_evaluation.txt")
        assert os.path.isfile(path)
