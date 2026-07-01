"""
detect.py <pcap>

Load the trained model and run it over a fresh capture. Print every flow the
model flags as an LLM/AI streaming flow, with its confidence, and make the
core claim explicit: the decision used ONLY packet timing and sizes — no
hostname, no SNI, no IP address, no port number.
"""

import sys

import joblib
import pandas as pd

from features import FEATURE_COLS, extract_features

MODEL_PATH = "model.joblib"
FLAG_THRESHOLD = 0.5  # P(AI) at/above this -> flag the flow


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python detect.py <pcap>", file=sys.stderr)
        return 2

    pcap_path = sys.argv[1]

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    # Sanity: the model was trained on exactly the shape features, in order.
    assert feature_cols == FEATURE_COLS, "feature column mismatch vs training"

    df = extract_features(pcap_path)
    if df.empty:
        print(f"No usable TCP flows (>= min packets) found in {pcap_path}.")
        return 0

    # Predict using FEATURE_COLS ONLY. The df also holds client/server columns,
    # but they are not selected here — the model literally cannot see them.
    X = df[feature_cols].to_numpy(dtype=float)
    proba_ai = model.predict_proba(X)[:, 1]
    df = df.assign(p_ai=proba_ai)

    flagged = df[df["p_ai"] >= FLAG_THRESHOLD].sort_values("p_ai", ascending=False)

    print("=" * 70)
    print(f"LLM-shape scan of {pcap_path}")
    print(f"{len(df)} TCP flow(s) analyzed, {len(flagged)} flagged as LLM/AI.")
    print("=" * 70)

    for _, row in flagged.iterrows():
        # `server` is shown ONLY so an operator knows which connection to look
        # at. It was recovered from the packet headers for reporting; it was
        # NOT an input to the classifier.
        print(f"\n[FLAGGED LLM FLOW]  confidence P(AI) = {row['p_ai']:.3f}")
        print(f"  server (for your reference only, NOT used to decide): {row['server']}")
        print( "  shape that triggered it (this is all the model saw):")
        print(f"    duration            = {row['duration']:.2f} s")
        print(f"    downstream packets  = {int(row['n_down'])}  "
              f"({row['down_pps']:.1f}/s)")
        print(f"    small(<300B) frac   = {row['frac_small_down']:.2f}")
        print(f"    down pkt size mean  = {row['down_size_mean']:.0f} B "
              f"(std {row['down_size_std']:.0f})")
        print(f"    down inter-arrival  = mean {row['down_iat_mean']*1000:.0f} ms, "
              f"max {row['down_iat_max']*1000:.0f} ms, cv {row['down_iat_cv']:.2f}")

    if flagged.empty:
        print("\nNo flows matched the LLM streaming shape.")

    # ---- The point, stated plainly ----
    print("\n" + "-" * 70)
    print("NOTE ON METHOD:")
    print("  This detection used ONLY traffic shape - packet timings and sizes.")
    print("  It did NOT use hostname, TLS SNI, IP address, or port number to")
    print("  decide anything. IP/port were read solely to split each flow into")
    print("  upstream vs downstream directions; their values never reached the")
    print("  model. That is why this approach keeps working after Encrypted")
    print("  Client Hello (ECH) hides the SNI: there is no identifier to encrypt,")
    print("  only the shape, and the shape is what we classify on.")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
