"""
Train the Wireless IDS classifier.

Reads every labelled pcap in the training directory, turns them into
per-(device, window) feature rows, then trains a scikit-learn classifier to
label each window normal (0) or malicious (1).

Key choices explained for the write-up:
  * We hold out WHOLE scenarios for testing (a "group" split), not random
    rows. Random splitting would put windows from the same attack in both
    train and test and inflate the score. Holding out whole captures tests
    whether the model generalises to an attack it has never seen.
  * We train a Random Forest (default) and also a single Decision Tree so the
    poster can show the tree is interpretable and the forest is more robust.
  * We report accuracy, precision, recall, F1 and a confusion matrix, and save
    plots to results/.

Run:
  python src/train.py --data data/train --model Random Forest --out models/wids.joblib
"""
import argparse
import glob
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit

from features import FEATURE_COLUMNS, pcap_to_windows


def load_dataset(data_dir, window_sec, step_sec):
    """Build one big feature table from all pcaps, tagging each row with its
    scenario name (used as the group for the split)."""
    frames = []
    pcaps = sorted(glob.glob(os.path.join(data_dir, "*.pcap")))
    if not pcaps:
        raise SystemExit(f"No pcaps found in {data_dir}")
    for p in pcaps:
        base = p[:-5]
        label_csv = base + ".labels.csv"
        label_csv = label_csv if os.path.exists(label_csv) else None
        w = pcap_to_windows(p, label_csv, window_sec=window_sec, step_sec=step_sec)
        w["scenario"] = os.path.basename(base)
        frames.append(w)
        print(f"  loaded {os.path.basename(p):40s} windows={len(w):4d}  malicious={int(w.label.sum()):4d}")
    df = pd.concat(frames, ignore_index=True)
    return df


def make_model(name):
    if name.lower() in ("rf", "randomforest", "random_forest"):
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1)
    if name.lower() in ("dt", "tree", "decisiontree", "decision_tree"):
        return DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42)
    raise ValueError(f"unknown model {name}")


def plot_confusion(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "malicious"])
    ax.set_yticklabels(["normal", "malicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / total if total else 0
            ax.text(j, i, f"{cm[i, j]}\n{frac:.1%}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_importance(model, out_path):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([FEATURE_COLUMNS[i] for i in order], imp[order], color="#2b7bba")
    ax.set_title("Feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--model", default="random_forest", help="random_forest or decision_tree")
    ap.add_argument("--out", default="models/wids.joblib")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.data, args.window, args.step)
    print(f"\nTotal windows: {len(df)}   malicious: {int(df.label.sum())}   "
          f"normal: {int((df.label == 0).sum())}")
    print(f"Scenarios: {df.scenario.nunique()}")

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values
    groups = df["scenario"].values

    # hold out whole scenarios
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    print(f"Train windows: {len(Xtr)}   Test windows: {len(Xte)}")
    print(f"Held-out test scenarios: {sorted(set(df.scenario.values[test_idx]))}")

    model = make_model(args.model)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1],
                                                  zero_division=0, average=None)

    print("\n================ RESULTS (held-out scenarios) ================")
    print(f"Model: {args.model}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix [rows=actual, cols=predicted]:")
    print(f"                 pred_normal  pred_malicious")
    print(f"  actual_normal       {cm[0,0]:6d}        {cm[0,1]:6d}")
    print(f"  actual_malicious    {cm[1,0]:6d}        {cm[1,1]:6d}")
    print("\nPer-class:")
    print(f"  normal     precision={p[0]:.3f} recall={r[0]:.3f} f1={f1[0]:.3f}")
    print(f"  malicious  precision={p[1]:.3f} recall={r[1]:.3f} f1={f1[1]:.3f}")
    print("\n" + classification_report(yte, pred, target_names=["normal", "malicious"],
                                       zero_division=0))

    plot_confusion(cm, os.path.join(args.results, "confusion_matrix.png"),
                   f"{args.model}  (acc={acc:.3f})")
    plot_importance(model, os.path.join(args.results, "feature_importance.png"))

    # persist model + the feature column order + window settings
    joblib.dump({"model": model, "features": FEATURE_COLUMNS,
                 "window_sec": args.window, "step_sec": args.step,
                 "model_name": args.model}, args.out)
    print(f"\nSaved model -> {args.out}")
    print(f"Saved plots -> {args.results}/confusion_matrix.png, feature_importance.png")

    # also write a short metrics file for the poster
    with open(os.path.join(args.results, "metrics.txt"), "w") as fh:
        fh.write(f"model: {args.model}\n")
        fh.write(f"accuracy: {acc:.4f}\n")
        fh.write(f"malicious precision: {p[1]:.4f}\n")
        fh.write(f"malicious recall: {r[1]:.4f}\n")
        fh.write(f"malicious f1: {f1[1]:.4f}\n")
        fh.write(f"confusion_matrix [[TN,FP],[FN,TP]]: {cm.tolist()}\n")
        fh.write(f"test_scenarios: {sorted(set(df.scenario.values[test_idx]))}\n")


if __name__ == "__main__":
    main()
