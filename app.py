"""Flask web app for NextWord AI.

Exposes RESTful endpoints for next-word prediction, sequence auto-completion,
live benchmark evaluations, dataset management, and model stats.
"""
from flask import Flask, render_template, request, jsonify
from predict import NextWordPredictor
import os, subprocess, threading

app = Flask(__name__)
predictor = NextWordPredictor()
training_lock = threading.Lock()
is_training = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/lab")
def lab():
    return render_template("lab.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    language = (data.get("language") or "english").lower()
    sequence_len = int(data.get("sequence_length") or 1)
    count = int(data.get("count") or 5)

    if not text:
        return jsonify({"suggestions": [], "sequence": "", "message": "Enter text to see predictions."}), 200

    if language not in {"english", "tamil"}:
        return jsonify({"error": "Language must be English or Tamil."}), 400

    try:
        suggestions = predictor.predict_single(text, language=language, count=count)
        sequence = ""
        if sequence_len > 1:
            sequence = predictor.predict_sequence(text, language=language, steps=sequence_len)

        return jsonify({
            "suggestions": suggestions,
            "sequence": sequence,
            "mode": predictor.mode(language)
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction error: {str(exc)}"}), 500

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return dataset, model, and vocabulary statistics."""
    stats = predictor.get_dataset_stats()
    return jsonify({
        "stats": stats,
        "is_training": is_training
    })

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Run top-1 and top-3 accuracy evaluation on corpus data."""
    results = {}
    for lang in ("english", "tamil"):
        path = os.path.join(BASE_DIR, "dataset", f"{lang}.txt")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        top1_hits = top3_hits = total = 0
        for line in lines:
            words = line.split()
            if len(words) < 3:
                continue
            prefix, target = " ".join(words[:-1]), words[-1].casefold()
            preds = predictor.predict_single(prefix, language=lang, count=3)
            words_pred = [p["word"].casefold() for p in preds]

            if words_pred and words_pred[0] == target:
                top1_hits += 1
            if target in words_pred:
                top3_hits += 1
            total += 1

        results[lang] = {
            "total_evaluations": total,
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "top1_accuracy": round((top1_hits / total) * 100, 1) if total else 0,
            "top3_accuracy": round((top3_hits / total) * 100, 1) if total else 0
        }
    return jsonify({"evaluation": results})

@app.route("/api/add-sentence", methods=["POST"])
def api_add_sentence():
    """Add a new training sentence to the selected dataset."""
    data = request.get_json(silent=True) or {}
    sentence = (data.get("sentence") or "").strip()
    language = (data.get("language") or "english").lower()

    if not sentence or len(sentence.split()) < 2:
        return jsonify({"error": "Please provide a valid sentence with at least 2 words."}), 400

    if language not in {"english", "tamil"}:
        return jsonify({"error": "Language must be English or Tamil."}), 400

    dataset_path = os.path.join(BASE_DIR, "dataset", f"{language}.txt")
    with open(dataset_path, "a", encoding="utf-8") as f:
        f.write(f"\n{sentence}")

    return jsonify({
        "success": True,
        "message": f"Added sentence to {language} dataset! Train the model to include it in neural predictions."
    })

def run_train_script():
    global is_training
    try:
        subprocess.run(["python", "train.py"], cwd=BASE_DIR, check=True)
        predictor.load_models()
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        is_training = False

@app.route("/api/train", methods=["POST"])
def api_train():
    """Trigger background model retraining."""
    global is_training
    if is_training:
        return jsonify({"message": "Training is already in progress...", "status": "busy"}), 200

    with training_lock:
        is_training = True
        thread = threading.Thread(target=run_train_script)
        thread.start()

    return jsonify({"message": "Started neural LSTM model training in background!", "status": "started"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
