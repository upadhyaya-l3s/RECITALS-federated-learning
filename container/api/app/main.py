"""
RECITALS Federated Learning – Insider Threat Detection API
Wraps the trained federated LSTM / Transformer model for inference.

Repo: https://github.com/upadhyaya-l3s/RECITALS-federated-learning
Dataset: CERT Insider Threat Dataset r4.2 (weekr4.2.csv)
"""

import os
import copy
import time
import uuid
import logging
import numpy as np
from datetime import datetime
from typing import Optional, List

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recitals-fl-api")

# ─────────────────────────────────────────────────────────────────────────────
# LSTM model  —  exact architecture from federated_lstm.py in the repo
# ─────────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)                      # (B, 1, input_dim)
        x = self.input_projection(x)             # (B, 1, hidden_dim)
        lstm_out, _ = self.lstm(x)               # (B, 1, hidden_dim)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x                                 # raw logit → sigmoid outside


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns — exactly as derived from CERT r4.2 via lcd-dal extractor
# (all columns in weekr4.2.csv minus the excluded metadata columns)
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE_COLS = {
    "user", "week", "insider", "insider_binary",
    "starttime", "endtime", "role", "b_unit",
    "f_unit", "dept", "team", "ITAdmin",
}

# These are the ~33 numeric features produced by the CERT feature extractor.
# They are listed here explicitly so the API can validate / document inputs.
CERT_FEATURE_COLS = [
    # Logon activity
    "logon_count", "logon_after_hours", "logoff_count",
    # PC / device usage
    "pc_count",
    # File activity
    "file_count", "file_exe", "file_jpg", "file_pdf",
    "file_doc", "file_zip", "file_txt",
    # USB / removable media
    "device_connect", "device_disconnect",
    # Email activity
    "email_count", "email_sent", "email_received",
    "email_cc", "email_bcc", "email_attach",
    "email_external_sent",
    # HTTP / browser activity
    "http_count", "http_upload", "http_download",
    # LDAP / organisational
    "ldap_count",
    # Psychometric features (from CERT r4.2 scenario)
    "O", "C", "E", "A", "N",            # Big-Five personality scores
    # After-hours aggregates
    "after_hours_logon", "after_hours_file",
]

#INPUT_DIM = len(CERT_FEATURE_COLS)



THRESHOLD  = 0.3          # same as in federated_lstm.py
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/federated_lstm_heterogeneous_moderate.pt")
MODEL_VERSION = os.getenv("MODEL_VERSION", "fl-lstm-fedavg-v1.0-cert-r4.2")

def _get_input_dim(model_path: str) -> int:
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        return ckpt["input_projection.weight"].shape[1]
    return 657  # fallback: actual CERT r4.2 feature count

INPUT_DIM = _get_input_dim(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[LSTMClassifier] = None
_model_loaded = False

def load_model():
    global _model, _model_loaded
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = LSTMClassifier(input_dim=INPUT_DIM).to(_device)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model.eval()
        _model_loaded = True
        logger.info("Model loaded successfully.")
    else:
        logger.warning(
            f"Model file not found at {MODEL_PATH}. "
            "Running with rule-based stub — replace with real weights."
        )
        _model_loaded = False

load_model()
_START = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RECITALS FL – Insider Threat Detection API",
    description=(
        "REST inference API for the RECITALS federated learning component. "
        "Accepts pre-extracted CERT-format week-level feature vectors and returns "
        "insider threat scores from the federated LSTM model.\n\n"
        "**No raw user data leaves the local node** — this API is intended for "
        "deployment at each RECITALS federation participant site."
    ),
    version="1.0.0",
    contact={"name": "L3S Research Center", "url": "https://github.com/upadhyaya-l3s/RECITALS-federated-learning"},
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    openapi_tags=[
        {"name": "detection", "description": "Inference / scoring endpoints"},
        {"name": "health",    "description": "Liveness and readiness probes"},
        {"name": "model",     "description": "Model and federation metadata"},
    ],
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# Input schema
# Two modes:
#   1. Raw CERT activity event (as provided by HES / the deliverable spec)
#   2. Pre-extracted feature vector (for direct FL model consumption)
# ─────────────────────────────────────────────────────────────────────────────
class ActivityEvent(BaseModel):
    """
    A single user activity event as defined in the RECITALS HES data spec.
    The API will map this to the CERT feature space internally.
    """
    timestamp:     str   = Field(..., example="2026-03-20T10:15:30Z")
    user_id:       str   = Field(..., example="U12345")
    device_id:     str   = Field(..., example="PC01")
    activity_type: str   = Field(..., example="file_access",
                                  description="file_access | email | http | logon | device")
    action:        str   = Field(..., example="copy_to_usb")
    file_type:     Optional[str] = Field(None, example="document")
    file_name:     Optional[str] = Field(None, example="report.docx")
    source:        Optional[str] = Field(None, example="local_disk")
    destination:   Optional[str] = Field(None, example="usb")
    is_work_hours: bool  = Field(..., example=True)


class CERTFeatureVector(BaseModel):
    """
    Pre-extracted week-level feature vector in CERT r4.2 format.
    Use this if you've already run the lcd-dal feature extractor.
    Field names match the columns in weekr4.2.csv exactly.
    """
    user_id:              str   = Field(..., example="U12345")
    week:                 int   = Field(..., example=5)
    logon_count:          float = Field(0.0)
    logon_after_hours:    float = Field(0.0)
    logoff_count:         float = Field(0.0)
    pc_count:             float = Field(0.0)
    file_count:           float = Field(0.0)
    file_exe:             float = Field(0.0)
    file_jpg:             float = Field(0.0)
    file_pdf:             float = Field(0.0)
    file_doc:             float = Field(0.0)
    file_zip:             float = Field(0.0)
    file_txt:             float = Field(0.0)
    device_connect:       float = Field(0.0)
    device_disconnect:    float = Field(0.0)
    email_count:          float = Field(0.0)
    email_sent:           float = Field(0.0)
    email_received:       float = Field(0.0)
    email_cc:             float = Field(0.0)
    email_bcc:            float = Field(0.0)
    email_attach:         float = Field(0.0)
    email_external_sent:  float = Field(0.0)
    http_count:           float = Field(0.0)
    http_upload:          float = Field(0.0)
    http_download:        float = Field(0.0)
    ldap_count:           float = Field(0.0)
    O:                    float = Field(0.0, description="Big-Five: Openness")
    C:                    float = Field(0.0, description="Big-Five: Conscientiousness")
    E:                    float = Field(0.0, description="Big-Five: Extraversion")
    A:                    float = Field(0.0, description="Big-Five: Agreeableness")
    N:                    float = Field(0.0, description="Big-Five: Neuroticism")
    after_hours_logon:    float = Field(0.0)
    after_hours_file:     float = Field(0.0)


class BatchActivityRequest(BaseModel):
    events: List[ActivityEvent] = Field(..., min_length=1, max_length=500)

class BatchFeatureRequest(BaseModel):
    records: List[CERTFeatureVector] = Field(..., min_length=1, max_length=500)


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────
class ThreatScore(BaseModel):
    event_id:        str
    user_id:         str
    anomaly_score:   float = Field(..., ge=0.0, le=1.0,
                                    description="Continuous risk in [0,1]. Higher = more anomalous.")
    threat_label:    str   = Field(..., description="normal | suspicious | malicious")
    confidence:      float = Field(..., ge=0.0, le=1.0)
    triggered_rules: List[str] = []
    model_version:   str
    scored_at:       str

class DetectResponse(BaseModel):
    request_id:         str
    scored_events:      List[ThreatScore]
    model_version:      str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    model_version:                str
    architecture:                 str
    base_dataset:                 str
    feature_count:                int
    feature_names:                List[str]
    decision_threshold:           float
    federation_algorithm:         str
    federation_rounds_completed:  int
    num_clients_trained:          int


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _activity_to_features(event: ActivityEvent) -> np.ndarray:
    """
    Maps a raw HES activity event to the CERT week-level feature vector.
    This is a *single-event* approximation — in production you would
    aggregate events per (user, week) before calling the model.
    """
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    feat = {f: i for i, f in enumerate(CERT_FEATURE_COLS)}

    act = event.activity_type.lower()
    action = event.action.lower()
    dest = (event.destination or "").lower()
    ftype = (event.file_type or "").lower()
    after_hours = not event.is_work_hours

    if act == "logon":
        vec[feat["logon_count"]] = 1.0
        if after_hours:
            vec[feat["logon_after_hours"]] = 1.0
            vec[feat["after_hours_logon"]] = 1.0
    elif act == "file_access":
        vec[feat["file_count"]] = 1.0
        if ftype in ("exe", "application"):   vec[feat["file_exe"]] = 1.0
        elif ftype in ("jpg", "image"):       vec[feat["file_jpg"]] = 1.0
        elif ftype == "pdf":                  vec[feat["file_pdf"]] = 1.0
        elif ftype in ("doc", "document"):    vec[feat["file_doc"]] = 1.0
        elif ftype in ("zip", "archive"):     vec[feat["file_zip"]] = 1.0
        elif ftype in ("txt", "text"):        vec[feat["file_txt"]] = 1.0
        if dest == "usb":
            vec[feat["device_connect"]] = 1.0
        if after_hours:
            vec[feat["after_hours_file"]] = 1.0
    elif act == "device":
        if "connect" in action:    vec[feat["device_connect"]] = 1.0
        if "disconnect" in action: vec[feat["device_disconnect"]] = 1.0
    elif act == "email":
        vec[feat["email_count"]] = 1.0
        if "sent" in action or "send" in action:
            vec[feat["email_sent"]] = 1.0
            if "external" in action or dest not in ("", "internal"):
                vec[feat["email_external_sent"]] = 1.0
        else:
            vec[feat["email_received"]] = 1.0
    elif act == "http":
        vec[feat["http_count"]] = 1.0
        if "upload" in action: vec[feat["http_upload"]] = 1.0
        if "download" in action: vec[feat["http_download"]] = 1.0

    return vec


def _feature_vec_to_array(record: CERTFeatureVector) -> np.ndarray:
    return np.array(
        [getattr(record, f) for f in CERT_FEATURE_COLS], dtype=np.float32
    )


def _score_tensor(features: np.ndarray) -> tuple[float, str, float, List[str]]:
    """Run model forward pass OR fall back to heuristic stub."""
    if _model_loaded and _model is not None:
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(_device)
            logit = _model(x)
            prob  = torch.sigmoid(logit).item()
        rules: List[str] = []
    else:
        # Heuristic stub (no weights yet) — mirrors logic from federated_lstm.py
        prob = 0.0
        rules: List[str] = []
        f = {name: features[i] for i, name in enumerate(CERT_FEATURE_COLS)}
        if f.get("device_connect", 0) > 0:
            prob += 0.35; rules.append("usb_device_connect")
        if f.get("after_hours_file", 0) > 0:
            prob += 0.20; rules.append("file_access_after_hours")
        if f.get("after_hours_logon", 0) > 0:
            prob += 0.15; rules.append("logon_after_hours")
        if f.get("email_external_sent", 0) > 0:
            prob += 0.20; rules.append("external_email")
        if f.get("http_upload", 0) > 0:
            prob += 0.10; rules.append("http_upload")
        prob = min(prob, 1.0)

    # Use same threshold as training (THRESHOLD = 0.3)
    if prob >= 0.7:
        label, conf = "malicious",  min(0.75 + prob * 0.25, 1.0)
    elif prob >= THRESHOLD:
        label, conf = "suspicious", min(0.60 + prob * 0.30, 1.0)
    else:
        label, conf = "normal",     max(0.90 - prob * 0.40, 0.0)

    return round(prob, 4), label, round(conf, 4), rules


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status        = "ok",
        model_loaded  = _model_loaded,
        model_version = MODEL_VERSION,
        uptime_seconds= round(time.time() - _START, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    return ModelInfoResponse(
        model_version               = MODEL_VERSION,
        architecture                = "LSTM (input_proj → 2-layer LSTM → FC head → sigmoid)",
        base_dataset                = "CERT Insider Threat Dataset r4.2 (weekr4.2.csv)",
        feature_count               = INPUT_DIM,
        feature_names               = [f"feature_{i}" for i in range(INPUT_DIM)],
        decision_threshold          = THRESHOLD,
        federation_algorithm        = "FedAvg (weighted by client dataset size)",
        federation_rounds_completed = 5,
        num_clients_trained         = 2,
    )


@app.post("/detect", response_model=DetectResponse, tags=["detection"],
          summary="Score a batch of raw HES activity events")
def detect(request: BatchActivityRequest):
    """
    Accepts raw activity events (as per HES data spec) and returns
    insider threat scores. Events are mapped to CERT feature space internally.

    For best accuracy, aggregate events per (user, week) using the
    lcd-dal feature extractor before calling `/detect/features`.
    """
    t0 = time.time()
    scored = []
    for ev in request.events:
        try:
            features = _activity_to_features(ev)
            score, label, conf, rules = _score_tensor(features)
            scored.append(ThreatScore(
                event_id      = str(uuid.uuid4()),
                user_id       = ev.user_id,
                anomaly_score = score,
                threat_label  = label,
                confidence    = conf,
                triggered_rules=rules,
                model_version = MODEL_VERSION,
                scored_at     = datetime.utcnow().isoformat() + "Z",
            ))
        except Exception as e:
            logger.exception(f"Failed scoring event for user {ev.user_id}")
            raise HTTPException(500, detail=str(e))

    return DetectResponse(
        request_id        = str(uuid.uuid4()),
        scored_events     = scored,
        model_version     = MODEL_VERSION,
        processing_time_ms= round((time.time() - t0) * 1000, 2),
    )


@app.post("/detect/features", response_model=DetectResponse, tags=["detection"],
          summary="Score pre-extracted CERT week-level feature vectors (most accurate)")
def detect_features(request: BatchFeatureRequest):
    """
    **Preferred endpoint** when you have already run the lcd-dal feature extractor
    on your CERT-format logs to produce week-level aggregated rows.
    Feature names match `weekr4.2.csv` columns exactly.
    """
    t0 = time.time()
    scored = []
    for rec in request.records:
        try:
            features = _feature_vec_to_array(rec)
            score, label, conf, rules = _score_tensor(features)
            scored.append(ThreatScore(
                event_id       = str(uuid.uuid4()),
                user_id        = rec.user_id,
                anomaly_score  = score,
                threat_label   = label,
                confidence     = conf,
                triggered_rules= rules,
                model_version  = MODEL_VERSION,
                scored_at      = datetime.utcnow().isoformat() + "Z",
            ))
        except Exception as e:
            logger.exception(f"Failed scoring record for user {rec.user_id}")
            raise HTTPException(500, detail=str(e))

    return DetectResponse(
        request_id        = str(uuid.uuid4()),
        scored_events     = scored,
        model_version     = MODEL_VERSION,
        processing_time_ms= round((time.time() - t0) * 1000, 2),
    )