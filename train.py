"""
train.py <ai.csv> <normal.csv>

Train the LLM-shape classifier.

  - ai.csv     : flows captured during an AI-only session   -> label 1
  - normal.csv : flows captured during a normal-only session -> label 0

Both CSVs come from extract_features.py. We label by *which capture the flow
came from* (see README "limitations": labels come from capturing classes
separately, not from per-flow ground truth).

We train a GradientBoostingClassifier on FEATURE_COLS ONLY — the shape
features. The identifier columns (client/server) sitting in the CSV are never
selected. The reported confusion matrix / classification report are
OUT-OF-SAMPLE: they come from 5-fold cross_val_predict, so every prediction
scored was made by a model that did NOT train on that flow.
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

from features import FEATURE_COLS

MODEL_PATH = "model.joblib"


def _load(csv_path: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing feature columns: {missing}")
    df = df.copy()
    df["label"] = label
    return df


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python train.py <ai.csv> <normal.csv>", file=sys.stderr)
        return 2

    ai_csv, normal_csv = sys.argv[1], sys.argv[2]

    ai = _load(ai_csv, 1)
    normal = _load(normal_csv, 0)
    data = pd.concat([ai, normal], ignore_index=True)

    # X is FEATURE_COLS ONLY. This single line is the enforcement point for the
    # whole thesis: the identifier columns (client/server) exist in `data` but
    # are never handed to the model.
    X = data[FEATURE_COLS].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=int)

    n_ai, n_normal = int((y == 1).sum()), int((y == 0).sum())
    print(f"Loaded {len(data)} flows: {n_ai} AI (label 1), {n_normal} normal (label 0)")
    print(f"Training on {len(FEATURE_COLS)} shape features (no host/SNI/IP/port):")
    print(f"  {FEATURE_COLS}\n")

    clf = GradientBoostingClassifier(random_state=0)

    # ---- OUT-OF-SAMPLE evaluation via 5-fold cross-validation ----
    # cross_val_predict returns, for each flow, the prediction made by the fold
    # in which that flow was held out. So nothing below is scored on data it
    # trained on. Fold count is clamped so tiny datasets still run.
    n_splits = min(5, n_ai, n_normal)
    if n_splits < 2:
        print("Need at least 2 flows per class to cross-validate.", file=sys.stderr)
        return 1
    if n_splits < 5:
        print(f"(note: only {min(n_ai, n_normal)} flows in smallest class; "
              f"using {n_splits}-fold instead of 5-fold)\n")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    print(f"=== {n_splits}-fold cross-validated (OUT-OF-SAMPLE) confusion matrix ===")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("            pred_normal  pred_AI")
    print(f"true_normal   {cm[0, 0]:>9}  {cm[0, 1]:>7}")
    print(f"true_AI       {cm[1, 0]:>9}  {cm[1, 1]:>7}\n")

    print("=== classification report (out-of-sample) ===")
    print(classification_report(y, y_pred, target_names=["normal", "AI"], digits=3))

    # ---- Fit the final model on ALL data and save it ----
    clf.fit(X, y)

    print("=== feature importances (final model, trained on all flows) ===")
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order:
        bar = "#" * int(round(importances[i] * 40))
        print(f"  {FEATURE_COLS[i]:<20} {importances[i]:.3f}  {bar}")

    # Bundle the feature-column order with the model so detect.py aligns inputs
    # exactly and can re-assert the "no identifiers" contract at load time.
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
