import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, get, reload_config


class TestConfig:
    def test_load_config(self):
        cfg = load_config()
        assert "app" in cfg
        assert cfg["app"]["name"] == "Churn Analytics Platform"

    def test_get_existing_key(self):
        val = get("app.name")
        assert val == "Churn Analytics Platform"

    def test_get_nested_key(self):
        val = get("model.threshold")
        assert isinstance(val, float)

    def test_get_default(self):
        val = get("nonexistent.key", "fallback")
        assert val == "fallback"

    def test_reload(self):
        cfg = reload_config()
        assert "app" in cfg
