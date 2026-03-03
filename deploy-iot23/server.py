"""
Lightweight Flask server that serves the existing dashboard
and provides a /api/predict endpoint for CSV upload + detection.
"""
import os, json, glob
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, "cap PPt", "IOT23_Models", "IOT23_Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Detection_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ── Load models at startup ──
def load_models():
    items = []
    if not os.path.exists(MODELS_ROOT):
        return items
    for ds_dir in sorted(glob.glob(os.path.join(MODELS_ROOT, "dataset*"))):
        model_path = os.path.join(ds_dir, "model.xgb")
        feat_path = os.path.join(ds_dir, "feature_list.txt")
        if not os.path.exists(model_path) or not os.path.exists(feat_path):
            continue
        with open(feat_path) as f:
            feats = [ln.strip() for ln in f if ln.strip()]
        bst = xgb.Booster()
        bst.load_model(model_path)
        items.append({"name": os.path.basename(ds_dir), "bst": bst, "features": feats})
    return items

MODELS = load_models()
print(f"Loaded {len(MODELS)} models")

# ── Prediction logic (exact same as user's code) ──
def one_hot_proto_service(df):
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

def prepare_features(df_raw, feature_list):
    df = df_raw.copy()
    df = df.drop(columns=["y_binary", "y_multiclass"], errors="ignore")
    df = one_hot_proto_service(df)
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

def predict_ensemble(df_raw, threshold=0.5, combine_rule="mean"):
    per_probs = []
    for m in MODELS:
        X = prepare_features(df_raw, m["features"])
        dmat = xgb.DMatrix(X, feature_names=m["features"])
        per_probs.append(m["bst"].predict(dmat))
    per_model_probs = np.vstack(per_probs).T
    final_prob = per_model_probs.mean(axis=1) if combine_rule == "mean" else per_model_probs.max(axis=1)
    final_pred = (final_prob >= threshold).astype(int)
    return final_prob, final_pred, per_model_probs

# ── Routes ──
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if len(MODELS) == 0:
        return jsonify({"error": "No models loaded"}), 500
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    threshold = float(request.form.get("threshold", 0.5))
    combine = request.form.get("combine", "mean")

    if file.filename.lower().endswith(".parquet"):
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(file)

    final_prob, final_pred, per_model_probs = predict_ensemble(df, threshold, combine)
    model_names = [m["name"] for m in MODELS]
    best_idx = per_model_probs.argmax(axis=1)

    results = []
    benign_rows = []
    malicious_rows = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        # convert numpy types to python types
        for k, v in row.items():
            if isinstance(v, (np.integer,)): row[k] = int(v)
            elif isinstance(v, (np.floating,)): row[k] = float(v)
            elif pd.isna(v): row[k] = None
        entry = {
            "index": i,
            "probability": round(float(final_prob[i]), 6),
            "verdict": "MALICIOUS" if final_pred[i] == 1 else "BENIGN",
            "top_model": model_names[best_idx[i]],
            "top_model_conf": round(float(per_model_probs[i].max()), 6),
        }
        results.append(entry)
        if final_pred[i] == 1:
            malicious_rows.append(row)
        else:
            benign_rows.append(row)

    # Auto-save
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = file.filename.rsplit(".", 1)[0]
    if malicious_rows:
        pd.DataFrame(malicious_rows).to_csv(os.path.join(OUTPUT_DIR, f"{base}_DROPPED_{ts}.csv"), index=False)
    if benign_rows:
        pd.DataFrame(benign_rows).to_csv(os.path.join(OUTPUT_DIR, f"{base}_CLEAN_{ts}.csv"), index=False)

    # Per-model stats for charts
    model_stats = []
    for j, name in enumerate(model_names):
        probs = per_model_probs[:, j]
        flagged = int((probs >= threshold).sum())
        model_stats.append({"name": name, "flagged": flagged, "avg_prob": round(float(probs.mean()), 4)})

    n_mal = int(final_pred.sum())
    n_ben = int(len(df) - n_mal)
    return jsonify({
        "total": len(df),
        "benign": n_ben,
        "malicious": n_mal,
        "threat_pct": round(n_mal / len(df) * 100, 2) if len(df) > 0 else 0,
        "results": results[:500],  # cap at 500 for browser performance
        "model_stats": model_stats,
        "prob_distribution": {
            "benign": [round(float(x), 4) for x in final_prob[final_pred == 0][:200]],
            "malicious": [round(float(x), 4) for x in final_prob[final_pred == 1][:200]]
        },
        "saved": True
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard: http://localhost:{port}")
    print(f"Models: {len(MODELS)} loaded from {MODELS_ROOT}")
    app.run(host="0.0.0.0", port=port, debug=False)
