"""Fine-tunes a transformer emotion recognizer (one for queries, one for replies) on CPU.

For each corpus (ED / ESConv) and side (q = query, d = reply): fine-tune DistilRoBERTa-base
on the training split, keep the epoch with the best validation accuracy, and save the
predicted class probabilities for the validation and test splits to cache/ as
  ft__{tag}__{side}__{split}.npy   (rows aligned with data_prep order, columns = sorted labels)
Usage: python finetune_emotion.py ED q [epochs] [max_train]
"""
import os, sys, time, json, math, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "results")
MODEL = os.environ.get("EMO_FT_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
torch.set_num_threads(os.cpu_count())
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def load(tag):
    if tag == "ED":
        return dp.load_ed("train"), dp.load_ed("valid"), dp.load_ed("test")
    es = dp.load_esconv()
    return es["train"], es["valid"], es["test"]


def batches(tok, texts, labels, bs, shuffle):
    idx = list(range(len(texts)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        enc = tok([texts[j] for j in b], truncation=True, max_length=64, padding=True, return_tensors="pt")
        yield enc, (torch.tensor([labels[j] for j in b]) if labels is not None else None)


@torch.no_grad()
def predict(model, tok, texts, bs=64):
    model.eval()
    out = []
    for enc, _ in batches(tok, texts, None, bs, False):
        out.append(torch.softmax(model(**enc).logits, -1).numpy())
    return np.concatenate(out)


def main(tag, side, epochs=3, max_train=None, speed_test=False):
    tr, va, te = load(tag)
    labels = sorted({r["emotion"] for r in tr + va + te})
    l2i = {l: i for i, l in enumerate(labels)}
    field = "query" if side == "q" else "target"
    Xtr, ytr = [r[field] for r in tr], [l2i[r["emotion"]] for r in tr]
    if max_train:
        Xtr, ytr = Xtr[:max_train], ytr[:max_train]
    Xva, yva = [r[field] for r in va], np.array([l2i[r["emotion"]] for r in va])
    Xte, yte = [r[field] for r in te], np.array([l2i[r["emotion"]] for r in te])
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))
    bs = 32
    # Freeze the embeddings and the lower layers; train only the top FREEZE_KEEP layers and the head (CPU budget).
    keep = int(os.environ.get("FREEZE_KEEP", "4"))
    base = getattr(model, model.base_model_prefix)
    for p in base.embeddings.parameters():
        p.requires_grad = False
    layers = base.encoder.layer
    for layer in layers[: max(0, len(layers) - keep)]:
        for p in layer.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable)/1e6:.1f}M of {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    steps = epochs * math.ceil(len(Xtr) / bs)
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best = (-1, None, None)
    log = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for k, (enc, y) in enumerate(batches(tok, Xtr, ytr, bs, True)):
            loss = model(**enc, labels=y).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            if speed_test and k == 19:
                el = time.time() - t0
                print(f"speed: {20 * bs / el:.1f} examples/s -> one epoch of {len(Xtr)} = {len(Xtr) / (20 * bs / el) / 60:.1f} min", flush=True)
                return
            if k % 100 == 0:
                print(f"[{tag}-{side}] epoch {ep+1} step {k} loss {loss.item():.3f} elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        Pva = predict(model, tok, Xva)
        acc = float((Pva.argmax(1) == yva).mean())
        log.append({"epoch": ep + 1, "valid_acc": acc, "minutes": (time.time() - t0) / 60})
        print(f"[{tag}-{side}] epoch {ep+1} valid acc {acc:.3f}", flush=True)
        if acc > best[0]:
            best = (acc, Pva, predict(model, tok, Xte))
    acc_va, Pva, Pte = best
    np.save(os.path.join(CACHE, f"ft__{tag}__{side}__valid.npy"), Pva.astype(np.float32))
    np.save(os.path.join(CACHE, f"ft__{tag}__{side}__test.npy"), Pte.astype(np.float32))
    test_acc = float((Pte.argmax(1) == yte).mean())
    print(f"[{tag}-{side}] best valid {acc_va:.3f} test {test_acc:.3f}", flush=True)
    p = os.path.join(OUT, "finetune_log.json")
    allog = json.load(open(p)) if os.path.exists(p) else {}
    allog[f"{tag}_{side}"] = {"model": MODEL, "epochs": log, "best_valid": acc_va, "test_acc": test_acc, "n_train": len(Xtr)}
    json.dump(allog, open(p, "w"), indent=1)


if __name__ == "__main__":
    tag, side = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if len(sys.argv) > 4 and sys.argv[4] == "speed":
        main(tag, side, epochs, speed_test=True)
    else:
        main(tag, side, epochs)
