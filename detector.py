"""
detector.py — shared scoring used by both detect.py (CLI) and app.py (portal).

Keeps the "no identifiers as features" contract in ONE place: score_pcap() only
ever feeds FEATURE_COLS to the model. It also knows how to bootstrap a model
from the fixtures on a fresh machine, so the web portal works out of the box.
"""

from __future__ import annotations

import os
import subprocess
import sys

import joblib
import pandas as pd

from features import FEATURE_COLS, extract_features

MODEL_PATH = "model.joblib"
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_model() -> None:
    """Guarantee model.joblib exists. If missing, bootstrap it from fixtures.

    This is what lets the portal "run on any machine": on first launch with no
    trained model, we synthesize the demo pcaps, extract features, and train —
    all via the SAME interpreter that's running us (sys.executable).
    """
    model_path = os.path.join(_PROJECT_DIR, MODEL_PATH)
    if os.path.exists(model_path):
        return

    def run(*args):
        subprocess.run([sys.executable, *args], cwd=_PROJECT_DIR, check=True)

    fixtures = os.path.join(_PROJECT_DIR, "fixtures")
    if not os.path.exists(os.path.join(fixtures, "ai_sample.pcap")):
        run("make_fixtures.py")

    # Extract the two labeled captures and train.
    ai_csv = os.path.join(_PROJECT_DIR, "ai.csv")
    normal_csv = os.path.join(_PROJECT_DIR, "normal.csv")
    if not os.path.exists(ai_csv):
        run("extract_features.py", os.path.join(fixtures, "ai_sample.pcap"), ai_csv)
    if not os.path.exists(normal_csv):
        run("extract_features.py", os.path.join(fixtures, "normal_sample.pcap"), normal_csv)
    run("train.py", ai_csv, normal_csv)


_bundle_cache = None


def _load_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        ensure_model()
        _bundle_cache = joblib.load(os.path.join(_PROJECT_DIR, MODEL_PATH))
        assert _bundle_cache["feature_cols"] == FEATURE_COLS, \
            "model was trained on different feature columns"
    return _bundle_cache


def feature_importances() -> list[dict]:
    """Return [{feature, importance}] sorted desc — for the portal to display."""
    model = _load_bundle()["model"]
    pairs = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    return [{"feature": f, "importance": float(w)} for f, w in pairs]


def score_pcap(pcap_path: str, threshold: float = 0.5) -> dict:
    """Parse + score one pcap. Returns a JSON-friendly dict of per-flow results.

    The model sees FEATURE_COLS ONLY. `server`/`client` are carried into the
    output for the operator's reference — they were NOT inputs to the decision.
    """
    bundle = _load_bundle()
    model, cols = bundle["model"], bundle["feature_cols"]

    df = extract_features(pcap_path)
    flows = []
    if not df.empty:
        proba = model.predict_proba(df[cols].to_numpy(dtype=float))[:, 1]
        for (_, row), p in zip(df.iterrows(), proba):
            flows.append({
                "server": row["server"],          # reference only, not a feature
                "client": row["client"],          # reference only, not a feature
                "p_ai": float(p),
                "flagged": bool(p >= threshold),
                "duration": float(row["duration"]),
                "n_packets": int(row["n_packets"]),
                "n_down": int(row["n_down"]),
                "n_up": int(row["n_up"]),
                "down_pps": float(row["down_pps"]),
                "frac_small_down": float(row["frac_small_down"]),
                "down_size_mean": float(row["down_size_mean"]),
                "down_size_std": float(row["down_size_std"]),
                "down_up_byte_ratio": float(row["down_up_byte_ratio"]),
                "down_iat_mean_ms": float(row["down_iat_mean"]) * 1000.0,
                "down_iat_std_ms": float(row["down_iat_std"]) * 1000.0,
                "down_iat_max_ms": float(row["down_iat_max"]) * 1000.0,
                "down_iat_cv": float(row["down_iat_cv"]),
            })

    # Loudest (most confident) flows first.
    flows.sort(key=lambda f: f["p_ai"], reverse=True)
    return {
        "n_flows": len(flows),
        "n_flagged": sum(1 for f in flows if f["flagged"]),
        "flows": flows,
    }
