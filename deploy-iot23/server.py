"""
Lightweight Flask server that serves the existing dashboard
and provides a /api/predict endpoint for CSV upload + detection.
"""
import os, sys, json, glob, traceback, time
import numpy as np
import pandas as pd
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, "cap PPt", "IOT23_Models", "IOT23_Models")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit for free tier

# ── Load models at startup with error handling ──
MODELS = []
LOAD_LOG = []

def load_models():
    global MODELS, LOAD_LOG
    LOAD_LOG.append(f"BASE_DIR: {BASE_DIR}")
    LOAD_LOG.append(f"MODELS_ROOT: {MODELS_ROOT}")
    LOAD_LOG.append(f"MODELS_ROOT exists: {os.path.exists(MODELS_ROOT)}")

    if not os.path.exists(MODELS_ROOT):
        # Try alternative paths
        alt_paths = [
            os.path.join(BASE_DIR, "IOT23_Models"),
            os.path.join(BASE_DIR, "cap PPt", "IOT23_Models"),
            os.path.join(BASE_DIR, "models"),
        ]
        for alt in alt_paths:
            LOAD_LOG.append(f"Trying alt: {alt} -> exists: {os.path.exists(alt)}")
            if os.path.exists(alt):
                return load_from_path(alt)
        LOAD_LOG.append("No model directory found!")
        # List what IS in BASE_DIR
        try:
            items = os.listdir(BASE_DIR)
            LOAD_LOG.append(f"Files in BASE_DIR: {items[:20]}")
        except:
            pass
        return

    load_from_path(MODELS_ROOT)

def load_from_path(root):
    global MODELS, LOAD_LOG
    try:
        import xgboost as xgb
        ds_dirs = sorted(glob.glob(os.path.join(root, "dataset*")))
        LOAD_LOG.append(f"Found {len(ds_dirs)} dataset folders in {root}")
        for ds_dir in ds_dirs:
            model_path = os.path.join(ds_dir, "model.xgb")
            feat_path = os.path.join(ds_dir, "feature_list.txt")
            if not os.path.exists(model_path) or not os.path.exists(feat_path):
                LOAD_LOG.append(f"SKIP {os.path.basename(ds_dir)}: missing files")
                continue
            try:
                with open(feat_path) as f:
                    feats = [ln.strip() for ln in f if ln.strip()]
                bst = xgb.Booster()
                bst.load_model(model_path)
                MODELS.append({"name": os.path.basename(ds_dir), "bst": bst, "features": feats})
                LOAD_LOG.append(f"OK {os.path.basename(ds_dir)}")
            except Exception as e:
                LOAD_LOG.append(f"ERR {os.path.basename(ds_dir)}: {e}")
    except Exception as e:
        LOAD_LOG.append(f"FATAL: {traceback.format_exc()}")

try:
    load_models()
except Exception as e:
    LOAD_LOG.append(f"load_models CRASH: {traceback.format_exc()}")

print(f"Loaded {len(MODELS)} models", flush=True)
for line in LOAD_LOG:
    print(f"  {line}", flush=True)

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
    import xgboost as xgb
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

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": len(MODELS), "log": LOAD_LOG})

@app.route("/<path:filename>")
def static_files(filename):
    if filename.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if len(MODELS) == 0:
            return jsonify({"error": f"No models loaded. Log: {LOAD_LOG[-5:]}"}), 500
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
        for i in range(min(len(df), 500)):
            results.append({
                "index": i,
                "probability": round(float(final_prob[i]), 6),
                "verdict": "MALICIOUS" if final_pred[i] == 1 else "BENIGN",
                "top_model": model_names[best_idx[i]],
                "top_model_conf": round(float(per_model_probs[i].max()), 6),
            })

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
            "results": results,
            "model_stats": model_stats,
            "prob_distribution": {
                "benign": [round(float(x), 4) for x in final_prob[final_pred == 0][:200]],
                "malicious": [round(float(x), 4) for x in final_prob[final_pred == 1][:200]]
            },
            "saved": False
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard: http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
