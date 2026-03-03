"""
IoT-23 Prediction Engine — Exact same logic from the user's original code.
Loads all XGBoost models and runs ensemble prediction.
"""
import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb


def load_models(models_root: str):
    """
    Loads all models that have BOTH:
      - model.xgb
      - feature_list.txt
    Returns:
      (items, debug_lines)
    """
    items = []
    debug_lines = []

    abs_root = os.path.abspath(models_root)
    debug_lines.append(f"MODELS_ROOT (resolved): {abs_root}")

    if not os.path.exists(abs_root):
        debug_lines.append("MODELS_ROOT path does NOT exist.")
        return [], debug_lines

    ds_dirs = sorted(glob.glob(os.path.join(abs_root, "dataset*")))
    debug_lines.append(f"Found dataset folders: {len(ds_dirs)}")

    for ds_dir in ds_dirs:
        model_path = os.path.join(ds_dir, "model.xgb")
        feat_path = os.path.join(ds_dir, "feature_list.txt")

        if not os.path.exists(model_path):
            debug_lines.append(f"SKIP {os.path.basename(ds_dir)}: missing model.xgb")
            continue
        if not os.path.exists(feat_path):
            debug_lines.append(f"SKIP {os.path.basename(ds_dir)}: missing feature_list.txt")
            continue

        # load feature list
        with open(feat_path, "r") as f:
            feats = [ln.strip() for ln in f if ln.strip()]

        # load model
        bst = xgb.Booster()
        bst.load_model(model_path)

        items.append({"name": os.path.basename(ds_dir), "bst": bst, "features": feats})
        debug_lines.append(f"LOADED {os.path.basename(ds_dir)}: features={len(feats)}")

    return items, debug_lines


def one_hot_proto_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw columns:
      proto -> proto_tcp, proto_udp, proto_icmp
      service -> service_dns, service_http, service_-
    If proto/service columns are missing, it just returns df unchanged.
    """
    out = df.copy()

    if "proto" in out.columns:
        p = out["proto"].astype(str).str.lower()
        out["proto_tcp"] = (p == "tcp").astype(float)
        out["proto_udp"] = (p == "udp").astype(float)
        out["proto_icmp"] = (p == "icmp").astype(float)

    if "service" in out.columns:
        s = out["service"].astype(str).str.lower()
        out["service_dns"] = (s == "dns").astype(float)
        out["service_http"] = (s == "http").astype(float)
        out["service_-"] = (s == "-").astype(float)

    return out


def prepare_features(df_raw: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    - Drops label columns if present (y_binary, y_multiclass)
    - Adds one-hot proto/service columns if raw proto/service exist
    - Ensures all required feature columns exist (missing -> 0)
    - Coerces numeric; NaN -> 0
    - Returns X in the exact feature_list order
    """
    df = df_raw.copy()
    df = df.drop(columns=["y_binary", "y_multiclass"], errors="ignore")

    df = one_hot_proto_service(df)

    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X


def predict_ensemble(df_raw: pd.DataFrame, models: list, threshold: float = 0.5, combine_rule: str = "mean"):
    """
    Returns:
      final_prob (N,)
      final_pred (N,)
      per_model_probs (N, M)
    """
    if len(models) == 0:
        raise RuntimeError("No models found. Fix MODELS_ROOT path or ensure model.xgb + feature_list.txt exist.")

    per_probs = []
    for m in models:
        X = prepare_features(df_raw, m["features"])
        dmat = xgb.DMatrix(X, feature_names=m["features"])
        prob = m["bst"].predict(dmat)
        per_probs.append(prob)

    per_model_probs = np.vstack(per_probs).T  # (N, M)

    if combine_rule == "mean":
        final_prob = per_model_probs.mean(axis=1)
    else:  # "max"
        final_prob = per_model_probs.max(axis=1)

    final_pred = (final_prob >= threshold).astype(int)
    return final_prob, final_pred, per_model_probs
