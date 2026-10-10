"""Next-word prediction engine.

In-memory high performance engine combining trained LSTM Neural Softmax probabilities with
prefix completion, multi-level n-gram backoff, and smart grammatical fallback heuristics.
"""
import os
import json
import re
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMON_GRAMMAR_FALLBACKS = {
    "english": {
        "of": [("india", 60.0), ("the", 25.0), ("data", 15.0)],
        "in": [("india", 50.0), ("the", 30.0), ("python", 20.0)],
        "the": [("prime", 40.0), ("model", 30.0), ("future", 30.0)],
        "is": [("a", 40.0), ("the", 30.0), ("very", 30.0)],
        "a": [("prime", 35.0), ("student", 35.0), ("crucial", 30.0)],
        "to": [("learn", 40.0), ("become", 30.0), ("build", 30.0)],
        "for": [("data", 45.0), ("machine", 35.0), ("development", 20.0)],
        "and": [("analytics", 40.0), ("technology", 30.0), ("machine", 30.0)],
    },
    "tamil": {
        "ஒரு": [("மாணவன்", 50.0), ("சிறந்த", 30.0), ("ஜனநாயக", 20.0)],
        "இந்தியாவின்": [("பிரதமர்", 60.0), ("தலைநகரம்", 40.0)],
        "மிகவும்": [("பிடிக்கும்", 50.0), ("நல்லது", 30.0), ("முக்கியமானது", 20.0)],
    }
}

class NextWordPredictor:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.max_lengths = {}
        self.corpus_lines = {}
        self.corpus_tokens = {}
        self.vocab_words = {}
        self.load_models()

    def load_models(self):
        """Reload models, tokenizers, and in-memory corpus caches."""
        self.models.clear()
        self.tokenizers.clear()
        self.max_lengths.clear()
        self.corpus_lines.clear()
        self.corpus_tokens.clear()
        self.vocab_words.clear()

        # Cache datasets in memory for instant searching
        for lang in ("english", "tamil"):
            path = os.path.join(BASE_DIR, "dataset", f"{lang}.txt")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    lines = [line.strip().lower() for line in f if line.strip()]
                self.corpus_lines[lang] = lines
                self.corpus_tokens[lang] = [
                    [re.sub(r'[^\w\s]', '', w) for w in line.split() if w]
                    for line in lines
                ]
                words_set = set()
                for tok_list in self.corpus_tokens[lang]:
                    words_set.update(tok_list)
                self.vocab_words[lang] = words_set
            else:
                self.corpus_lines[lang] = []
                self.corpus_tokens[lang] = []
                self.vocab_words[lang] = set()

        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            self.pad_sequences = pad_sequences

            for lang in ("english", "tamil"):
                model_path = os.path.join(BASE_DIR, "model", f"{lang}_model.keras")
                tok_path = os.path.join(BASE_DIR, "model", f"{lang}_tokenizer.json")
                length_path = os.path.join(BASE_DIR, "model", f"{lang}_maxlen.json")

                if os.path.exists(model_path) and os.path.exists(tok_path) and os.path.exists(length_path):
                    with open(tok_path, encoding="utf-8") as f:
                        self.tokenizers[lang] = tokenizer_from_json(f.read())
                    with open(length_path, encoding="utf-8") as f:
                        self.max_lengths[lang] = json.load(f)["max_sequence_len"]
                    self.models[lang] = load_model(model_path)
        except Exception as e:
            print(f"Model loading notice: {e}")
            self.pad_sequences = None

    def mode(self, language):
        return "Trained LSTM Neural Network" if language in self.models else "N-Gram Corpus Fallback Engine"

    def get_dataset_stats(self):
        """Dataset metrics for dashboard analytics."""
        stats = {}
        for lang in ("english", "tamil"):
            lines = self.corpus_lines.get(lang, [])
            all_words = [w for line in lines for w in line.split()]
            stats[lang] = {
                "total_sentences": len(lines),
                "total_words": len(all_words),
                "vocab_size": len(self.vocab_words.get(lang, set())),
                "has_trained_model": lang in self.models,
                "max_sequence_len": self.max_lengths.get(lang, 0)
            }
        return stats

    def _prefix_completion(self, text, language, count=5):
        """Fast in-memory partial word prefix completion."""
        raw_words = text.strip().split()
        if not raw_words or text.endswith(" "):
            return []

        last_token = re.sub(r'[^\w\s]', '', raw_words[-1].lower())
        if len(last_token) < 2:
            return []

        candidates = {}

        # Search tokenizer vocab
        if language in self.tokenizers:
            tok = self.tokenizers[language]
            for word, _ in tok.word_index.items():
                if word != "<oov>" and word.startswith(last_token) and word != last_token:
                    candidates[word] = candidates.get(word, 0) + 10

        # Search corpus vocab
        for word in self.vocab_words.get(language, set()):
            if word.startswith(last_token) and word != last_token:
                candidates[word] = candidates.get(word, 0) + 5

        if not candidates:
            return []

        total = sum(candidates.values())
        sorted_cand = sorted(candidates.items(), key=lambda x: -x[1])[:count]
        return [{"word": w, "confidence": max(round((c / total) * 100, 1), 25.0)} for w, c in sorted_cand]

    def _ngram_suggestions(self, text, language, count=5):
        """In-memory multi-level n-gram matching."""
        words = [re.sub(r'[^\w\s]', '', w) for w in text.lower().split() if w]
        if not words:
            return []

        candidates = {}
        tokens_list = self.corpus_tokens.get(language, [])

        for tokens in tokens_list:
            for n in (min(4, len(words)), min(3, len(words)), min(2, len(words)), 1):
                if n <= 0 or len(tokens) <= n:
                    continue
                wanted = words[-n:]
                for i in range(len(tokens) - n):
                    if tokens[i:i+n] == wanted:
                        candidate = tokens[i+n]
                        weight = 25 if n == 4 else 15 if n == 3 else 8 if n == 2 else 2
                        candidates[candidate] = candidates.get(candidate, 0) + weight

        if not candidates:
            last = words[-1]
            for tokens in tokens_list:
                for i in range(1, len(tokens)):
                    if tokens[i-1] == last:
                        candidates[tokens[i]] = candidates.get(tokens[i], 0) + 1

        if not candidates and words[-1] in COMMON_GRAMMAR_FALLBACKS.get(language, {}):
            for word, conf in COMMON_GRAMMAR_FALLBACKS[language][words[-1]]:
                candidates[word] = int(conf)

        if not candidates:
            return []

        total_weight = sum(candidates.values())
        sorted_cand = sorted(candidates.items(), key=lambda item: -item[1])[:count]

        results = []
        for word, weight in sorted_cand:
            conf = round((weight / total_weight) * 100, 1)
            results.append({"word": word, "confidence": max(conf, 10.0)})
        return results

    def predict_single(self, text, language="english", count=5):
        """Predict top next words with high precision."""
        text_clean = text.strip().lower()
        if not text_clean:
            return []

        prefix_matches = self._prefix_completion(text, language=language, count=count)
        ngram_results = self._ngram_suggestions(text_clean, language=language, count=count)

        neural_results = []
        if language in self.models and self.tokenizers.get(language):
            tokenizer = self.tokenizers[language]
            seq = tokenizer.texts_to_sequences([text_clean])[0]
            
            oov_id = tokenizer.word_index.get("<oov>", 1)
            filtered_seq = [idx for idx in seq if idx != oov_id]

            if filtered_seq:
                maxlen = self.max_lengths[language] - 1
                padded = self.pad_sequences([filtered_seq], maxlen=maxlen, padding="pre")
                probs = self.models[language].predict(padded, verbose=0)[0]

                reverse_vocab = {idx: word for word, idx in tokenizer.word_index.items() if word != "<oov>"}
                top_indices = np.argsort(probs)[::-1]

                for idx in top_indices:
                    if idx in reverse_vocab:
                        word = reverse_vocab[idx]
                        conf = round(float(probs[idx]) * 100, 1)
                        if conf > 0.1:
                            neural_results.append({"word": word, "confidence": conf})
                        if len(neural_results) >= count:
                            break

        combined = {}
        for item in prefix_matches:
            combined[item["word"]] = item["confidence"] + 60.0

        for item in ngram_results:
            w = item["word"]
            combined[w] = combined.get(w, 0) + item["confidence"] * 1.5

        for item in neural_results:
            w = item["word"]
            combined[w] = combined.get(w, 0) + item["confidence"]

        if not combined:
            return ngram_results

        sorted_combined = sorted(combined.items(), key=lambda x: -x[1])[:count]
        total = sum(c for _, c in sorted_combined)
        return [{"word": w, "confidence": min(99.0, max(round((c / total) * 100, 1), 5.0))} for w, c in sorted_combined]

    def predict_sequence(self, text, language="english", steps=3):
        """Recursively auto-complete multiple words ahead safely."""
        current_text = text.strip()
        generated_words = []
        seen_words = set()

        for _ in range(steps):
            preds = self.predict_single(current_text, language=language, count=3)
            if not preds:
                break
            
            # Select first word not causing infinite loop
            next_word = preds[0]["word"]
            for p in preds:
                if p["word"] not in seen_words:
                    next_word = p["word"]
                    break

            generated_words.append(next_word)
            seen_words.add(next_word)

            words = current_text.split()
            if words and not current_text.endswith(" ") and next_word.startswith(words[-1].lower()):
                words[-1] = next_word
                current_text = " ".join(words)
            else:
                current_text += f" {next_word}"

        return " ".join(generated_words)
