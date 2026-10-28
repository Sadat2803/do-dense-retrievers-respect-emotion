"""Emotion features from a pre-trained emotion classifier (no training on our data).

Runs SamLowe/roberta-base-go_emotions (RoBERTa-base fine-tuned on the 28 GoEmotions labels)
over the queries and replies of every split and caches its 28 logits:
  go__{tag}_{split}__{side}.npy     (side q = query, d = reply)
These logits are later concatenated with the sentence embedding used by the recognizer.
Usage: python emo_features.py [ED|ESConv|all]
"""
import os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

CACHE = os.path.join(ROOT, "cache")
MODEL = "SamLowe/roberta-base-go_emotions"
torch.set_num_threads(os.cpu_count())


@torch.no_grad()
def logits(model, tok, texts, bs=64):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=64, padding=True, return_tensors="pt")
        out.append(model(**enc).logits.numpy())
        if i == 0:
            el = time.time() - t0
            print(f"  ~{bs/el:.0f} texts/s, {len(texts)} texts -> ~{len(texts)/(bs/el)/60:.1f} min", flush=True)
    return np.concatenate(out).astype(np.float32)


def run(tag):
    if tag == "ED":
        splits = {"train": dp.load_ed("train"), "valid": dp.load_ed("valid"), "test": dp.load_ed("test")}
    else:
        splits = dp.load_esconv()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    for split, recs in splits.items():
        for side, field in (("q", "query"), ("d", "target")):
            path = os.path.join(CACHE, f"go__{tag}_{split}__{side}.npy")
            if os.path.exists(path):
                continue
            print(tag, split, side, flush=True)
            np.save(path, logits(model, tok, [r[field] for r in recs]))


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    for tag in (("ED", "ESConv") if w == "all" else (w,)):
        run(tag)
    print("done", flush=True)
