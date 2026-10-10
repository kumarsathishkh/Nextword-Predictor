<<<<<<< HEAD
# NextWord AI — Tamil & English Next-Word Prediction

A student open-ended deep learning experiment demonstrating next-word prediction with NLP and LSTM. Includes a Flask web interface, separate English/Tamil sample corpora, a training script, and a simple evaluation script.

## Features
- English and Tamil input modes
- Up to three word suggestions
- LSTM model training with TensorFlow/Keras
- Flask web interface
- Sample-corpus fallback before models are trained
- Evaluation helper

## Project structure
```text
NextWord-AI/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── dataset/
├── model/
├── templates/
├── static/
└── results/
```

## Setup (Python 3.10–3.11 recommended for compatible TensorFlow releases)
1. Create and activate a virtual environment:
   - Windows: `python -m venv .venv`
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `python app.py`
4. Open `http://127.0.0.1:5000`.

The site can run with the sample-corpus fallback even before training. This fallback is rule-based corpus matching, **not** a neural-model prediction.

## Train the models
Run:
```bash
python train.py
```
This trains separate models from `dataset/english.txt` and `dataset/tamil.txt`, saving `.keras` model files and tokenizer metadata under `model/`. The included corpora are tiny demonstration data; predictions will be limited. For a credible experiment, use a larger, legally usable dataset and reserve unseen sentences for testing.

## Evaluate
Run:
```bash
python evaluate.py
```
This reports a top-3 hit rate over the included corpus. It is only a smoke test because the tiny corpus is not an independent held-out test set. Do not report it as generalization accuracy.

## Suggested open-ended experiment extensions
- Compare LSTM with a simple n-gram baseline.
- Measure top-1 and top-3 accuracy on a held-out dataset.
- Compare English and Tamil results.
- Plot training/validation loss and accuracy.
- Try a larger licensed corpus and document its source.
- Deploy through a production WSGI server after development.

## Important notes
- No pretrained model is bundled; train the models locally with `python train.py`.
- `model/` contains metadata placeholders only until training succeeds.
- The starter corpus is for demonstration and should not be described as a large dataset.
- Never publish private or copyrighted training text without permission.

## Author
Student Open-Ended Deep Learning Experiment
=======
# Nextword-Predictor
>>>>>>> 60225de524ac5625276358338dca7e42f5977476
