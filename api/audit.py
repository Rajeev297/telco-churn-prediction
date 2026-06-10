import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional


AUDIT_LOGGER = logging.getLogger("audit")
PREDICTION_LOGGER = logging.getLogger("prediction")


def setup_audit_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)

    for name, filename in [("audit", "audit.log"), ("prediction", "predictions.log")]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(log_dir, filename))
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        if not logger.handlers:
            logger.addHandler(handler)


def log_api_event(
    action: str,
    client_ip: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    user: str = "anonymous",
):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "client_ip": client_ip,
        "user": user,
        "status": status,
        "details": details or {},
    }
    AUDIT_LOGGER.info(json.dumps(record))


def log_prediction(
    client_ip: str,
    num_customers: int,
    num_high_risk: int,
    avg_probability: float,
    response_time_ms: float,
):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": client_ip,
        "num_customers": num_customers,
        "num_high_risk": num_high_risk,
        "avg_probability": round(avg_probability, 4),
        "response_time_ms": round(response_time_ms, 1),
    }
    PREDICTION_LOGGER.info(json.dumps(record))
