"""Loads EmpatheticDialogues (ED) and ESConv into a common format.

Each record = one conversation:
  emotion : conversation emotion label
  query   : the help-seeker / speaker's first emotional message
  target  : the listener / supporter reply that follows it
"""
import os
import csv, json, random, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

DATA = os.path.join(ROOT, "data")

# 32 ED emotions grouped by valence (surprised, sentimental, nostalgic are ambiguous)
POSITIVE = {"excited", "proud", "grateful", "impressed", "hopeful", "confident", "anticipating",
            "joyful", "prepared", "content", "caring", "trusting", "faithful"}
NEGATIVE = {"afraid", "angry", "annoyed", "anxious", "apprehensive", "ashamed", "devastated",
            "disappointed", "disgusted", "embarrassed", "furious", "guilty", "jealous", "lonely",
            "sad", "terrified"}


def polarity(e):
    return 1 if e in POSITIVE else (-1 if e in NEGATIVE else 0)


def _clean(s):
    return re.sub(r"\s+", " ", s.replace("_comma_", ",")).strip()


def load_ed(split):
    """ED csv lines contain stray commas in some fields; take the first 8 comma-separated fields."""
    convs = {}
    with open(os.path.join(DATA, "ed", "empatheticdialogues", f"{split}.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 8:
                continue
            conv_id, idx, emotion, prompt, spk, utt = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            try:
                idx = int(idx)
            except ValueError:
                continue
            c = convs.setdefault(conv_id, {"emotion": emotion, "situation": _clean(prompt), "turns": {}})
            c["turns"][idx] = _clean(utt)
    out = []
    for cid, c in convs.items():
        if 1 in c["turns"] and 2 in c["turns"]:
            out.append({"id": cid, "emotion": c["emotion"], "situation": c["situation"],
                        "query": c["turns"][1], "target": c["turns"][2]})
    return out


def load_esconv(seed=13):
    """ESConv has no official split: 70/10/20 by conversation with a fixed seed.
    query = first seeker turn with at least 6 words; target = the next supporter turn."""
    data = json.load(open(os.path.join(DATA, "ESConv.json"), encoding="utf-8"))
    recs = []
    for i, d in enumerate(data):
        dlg = d["dialog"]
        for j, t in enumerate(dlg):
            if t["speaker"] == "seeker" and len(t["content"].split()) >= 6:
                nxt = next((u for u in dlg[j + 1:] if u["speaker"] == "supporter"), None)
                if nxt:
                    recs.append({"id": f"es{i}", "emotion": d["emotion_type"].strip().lower(),
                                 "situation": d.get("situation", ""), "query": _clean(t["content"]),
                                 "target": _clean(nxt["content"])})
                break
    rng = random.Random(seed)
    rng.shuffle(recs)
    n = len(recs)
    return {"train": recs[: int(.7 * n)], "valid": recs[int(.7 * n): int(.8 * n)], "test": recs[int(.8 * n):]}


if __name__ == "__main__":
    from collections import Counter
    for s in ("train", "valid", "test"):
        r = load_ed(s)
        print("ED", s, len(r), "emotions", len(set(x["emotion"] for x in r)))
    print(load_ed("test")[0])
    es = load_esconv()
    for s, r in es.items():
        print("ESConv", s, len(r), Counter(x["emotion"] for x in r).most_common(12))
    print(es["test"][0])
