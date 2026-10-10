"""Next-word evaluation suite across datasets.
"""
import os
from predict import NextWordPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    predictor = NextWordPredictor()
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
            word_list = [p["word"].casefold() for p in preds]

            if word_list and word_list[0] == target:
                top1_hits += 1
            if target in word_list:
                top3_hits += 1
            total += 1

        print(f"[{lang.upper()}] Top-1 Accuracy: {top1_hits}/{total} ({top1_hits/total:.1%}) | Top-3 Hit Rate: {top3_hits}/{total} ({top3_hits/total:.1%})")

if __name__ == "__main__":
    main()
