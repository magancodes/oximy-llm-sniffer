"""
app.py — the llm-shape-detector web portal.

Runs a local web server (Flask) that lets you, from a browser:

  1. Fire a REAL streaming LLM request and watch it get detected purely from
     its traffic shape (llm_probe reconstructs the on-wire flow from the real
     token timing/sizes — no packet-capture driver needed).
  2. Run a SIMULATED test on synthetic AI / normal / mixed flows (no API key).
  3. Analyze an uploaded .pcap or one of the bundled fixtures.

Every path runs the identical pcap-parse + model pipeline and the model only
ever sees FEATURE_COLS: no hostname, SNI, IP, or port. The UI keeps saying so.

Start it:  python app.py   (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

import detector
import llm_probe
from features import FEATURE_COLS

app = Flask(__name__)
# Serve the latest index.html on every request (it's a local dev tool; editing
# the UI shouldn't require a server restart).
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIXTURES_DIR = os.path.join(_PROJECT_DIR, "fixtures")


# ---------------------------------------------------------------------------
# Page + status
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Engine + model readiness for the UI to render, plus env defaults."""
    try:
        importances = detector.feature_importances()
    except Exception:
        importances = []

    fixtures = []
    if os.path.isdir(_FIXTURES_DIR):
        fixtures = sorted(f for f in os.listdir(_FIXTURES_DIR) if f.endswith(".pcap"))

    return jsonify({
        # Pure-Python pcap parsing (scapy). No tshark / system binary needed,
        # which is what lets this run serverless.
        "engine": "scapy (pure Python)",
        "model_ready": bool(importances),
        "feature_cols": FEATURE_COLS,
        "importances": importances,
        "fixtures": fixtures,
        # Prefill hints for the real-LLM form (values NEVER leaked, only presence).
        "env_defaults": {
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
    })


# ---------------------------------------------------------------------------
# Simulated test — synthetic flows, no API key required
# ---------------------------------------------------------------------------
@app.route("/api/simulate", methods=["POST"])
def simulate():
    body = request.get_json(force=True, silent=True) or {}
    kind = body.get("kind", "mixed")
    n_flows = int(body.get("n_flows", 6))
    n_flows = max(1, min(n_flows, 40))

    with tempfile.TemporaryDirectory() as d:
        pcap = os.path.join(d, "sim.pcap")
        llm_probe.synthesize_demo_pcap(kind, pcap, n_flows=n_flows,
                                       seed_offset=int(time.time()) % 1000)
        result = detector.score_pcap(pcap)
    result["source"] = f"simulated:{kind}"
    return jsonify(result)


# ---------------------------------------------------------------------------
# Analyze — uploaded pcap or a bundled fixture
# ---------------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    # Bundled fixture chosen by name?
    fixture = request.form.get("fixture") or (request.args.get("fixture"))
    if fixture:
        safe = os.path.basename(fixture)  # no path traversal
        path = os.path.join(_FIXTURES_DIR, safe)
        if not os.path.exists(path):
            return jsonify({"error": f"fixture not found: {safe}"}), 404
        result = detector.score_pcap(path)
        result["source"] = f"fixture:{safe}"
        return jsonify(result)

    # Otherwise an uploaded file.
    if "pcap" not in request.files:
        return jsonify({"error": "no pcap uploaded and no fixture chosen"}), 400
    up = request.files["pcap"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, os.path.basename(up.filename) or "upload.pcap")
        up.save(path)
        result = detector.score_pcap(path)
    result["source"] = f"upload:{up.filename}"
    return jsonify(result)


# ---------------------------------------------------------------------------
# Real LLM test — stream tokens live, then reconstruct + detect
# ---------------------------------------------------------------------------
def _ndjson(obj):
    return json.dumps(obj) + "\n"


@app.route("/api/real-llm", methods=["POST"])
def real_llm():
    body = request.get_json(force=True, silent=True) or {}
    provider = body.get("provider", "openai")
    base_url = body.get("base_url") or "https://api.openai.com/v1"
    model = body.get("model") or "gpt-4o-mini"
    prompt = body.get("prompt") or "Write a detailed paragraph about the ocean."
    max_tokens = int(body.get("max_tokens", 220))

    # Resolve the API key: explicit field wins, else fall back to env.
    api_key = body.get("api_key") or (
        os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic"
        else os.environ.get("OPENAI_API_KEY")
    )

    def generate():
        events = []
        try:
            yield _ndjson({"type": "start", "provider": provider, "model": model})
            req_start = time.time()
            first_t = None
            for ev in llm_probe.stream_llm(provider, base_url, api_key, model,
                                           prompt, max_tokens=max_tokens):
                if first_t is None:
                    first_t = ev["t"]
                events.append(ev)
                yield _ndjson({"type": "token", "text": ev["text"]})

            if not events:
                yield _ndjson({"type": "error",
                               "message": "stream returned no tokens"})
                return

            # Reconstruct the flow shape from the REAL token timing/sizes.
            with tempfile.TemporaryDirectory() as d:
                pcap = os.path.join(d, "real.pcap")
                stats = llm_probe.synthesize_pcap_from_events(events, pcap)
                stats["ttft_seconds"] = first_t - req_start
                result = detector.score_pcap(pcap)

            result["source"] = f"real-llm:{provider}/{model}"
            yield _ndjson({"type": "result", "stream": stats, "detection": result})
        except Exception as e:
            yield _ndjson({
                "type": "error",
                "message": f"{type(e).__name__}: {e}",
                "detail": traceback.format_exc().splitlines()[-3:],
            })

    return Response(stream_with_context(generate()),
                    mimetype="application/x-ndjson")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"llm-shape-detector portal -> http://127.0.0.1:{port}")
    # threaded so the streaming endpoint doesn't block status/simulate calls.
    app.run(host="127.0.0.1", port=port, threaded=True)
