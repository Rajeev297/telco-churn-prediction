import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealth:
    def test_health_endpoint(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
