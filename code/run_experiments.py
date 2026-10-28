"""Emotion-aware dense retrieval: benchmark, baselines and confidence-gated polarity-aware reranking.

Task: given a speaker's first emotional message (query), retrieve the listener's reply (gold)
from all replies of the same split. Every candidate inherits its conversation's emotion label.

Metrics (per query, averaged): R@1, R@10, MRR on the gold reply; EmoAcc@10 = share of the top-10
with the query's emotion; Harm@10 = share of the top-10 with opposite valence (ED only, queries
with non-ambiguous valence).

Methods (all hyper-parameters tuned on the validation split, never on test):
  base       semantic score only
  filter     keep candidates whose predicted emotion equals the query's predicted emotion
  comb       Emotional RAG 'combination': z(sem) + lam * cos(p_q, p_d)
  seq        Emotional RAG 'sequential': top-N by semantics, re-ranked by emotion similarity
  cgp (ours) z(sem) + c_q * (lam * cos(p_q, p_d) - mu * conflict(p_q, p_d)),
             c_q = classifier confidence (1 - normalised entropy), conflict = P(pos)P(neg)+P(neg)P(pos)
  oracle     comb with the true emotion labels (upper bound)
Selection rule on validation: maximise EmoAcc@10 subject to MRR >= 0.99 * MRR(base).
"""
import json, os, sys, time, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

OUT = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
RNG = np.random.default_rng(0)

DENSE = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "all-mpnet-base-v2": ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "multi-qa-MiniLM-L6": ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "", ""),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small-v2": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "gte-small": ("thenlper/gte-small", "", ""),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-base-v2": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "gte-base": ("thenlper/gte-base", "", ""),
}
EMO_ENCODER = "all-mpnet-base-v2"     # fixed encoder behind the emotion classifier
# Recognizer variant: "lr" = logistic regression on mpnet embeddings (original),
# "go" = the same plus 28 GoEmotions logits as features, "ft" = fine-tuned transformer probabilities.
EMO_VARIANT = os.environ.get("EMO_VARIANT", "lr")
SUFFIX = "" if EMO_VARIANT == "lr" else f"_{EMO_VARIANT}"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_FIRST_STAGE = "gte-base"
CE_TOPK = 100
PROBE_N = 8000

_models = {}


def embed(name, texts, tag, kind):
    path = os.path.join(CACHE, f"{name}__{tag}__{kind}.npy")
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    if name not in _models:
        _models[name] = SentenceTransformer(DENSE[name][0], device="cpu")
    prefix = DENSE[name][1] if kind == "q" else DENSE[name][2]
    E = _models[name].encode([prefix + t for t in texts], batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False).astype(np.float32)
    np.save(path, E)
    return E


def ce_scores(tag, split, qs, ds):
    """Cross-encoder re-ranking of the CE_TOPK best candidates of the first-stage dense retriever.
    Candidates outside the top-K keep their first-stage order, just below the lowest re-ranked score."""
    path = os.path.join(CACHE, f"ce__{tag}_{split}.npz")
    Q = embed(CE_FIRST_STAGE, qs, f"{tag}_{split}", "q"); D = embed(CE_FIRST_STAGE, ds, f"{tag}_{split}", "d")
    S1 = Q @ D.T
    top = np.argsort(-S1, axis=1)[:, :CE_TOPK]
    if os.path.exists(path):
        ce = np.load(path)["ce"]
    else:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(CE_MODEL, device="cpu", max_length=128)
        pairs = [(qs[i], ds[j]) for i in range(len(qs)) for j in top[i]]
        t0 = time.time()
        ce = model.predict(pairs, batch_size=128, show_progress_bar=False).reshape(len(qs), CE_TOPK).astype(np.float32)
        print(f"  cross-encoder {tag}/{split}: {len(pairs)} pairs in {(time.time()-t0)/60:.1f} min", flush=True)
        np.savez(path, ce=ce, top=top)
    S = np.empty_like(S1)
    rows = np.arange(len(qs))[:, None]
    floor = ce.min(1, keepdims=True) - 1.0
    S[:] = floor + 0.01 * (S1 - S1.max(1, keepdims=True))          # rest: below the re-ranked block, first-stage order
    S[rows, top] = ce
    return S


def zrows(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------ metrics
def evaluate(S, emo_q, emo_d, pol_q, pol_d, k=10):
    n = S.shape[0]
    gold = np.arange(n)
    order = np.argsort(-S, axis=1)
    ranks = np.argmax(order == gold[:, None], axis=1) + 1
    top = order[:, :k]
    emo_acc = (emo_d[top] == emo_q[:, None]).mean(1)
    valid = pol_q != 0
    harm = np.where(valid, ((pol_d[top] * pol_q[:, None]) < 0).mean(1), np.nan)
    return {"R@1": (ranks == 1), "R@10": (ranks <= 10), "MRR": 1.0 / ranks, "EmoAcc@10": emo_acc, "Harm@10": harm}


def summarise(m):
    return {k: float(np.nanmean(v)) for k, v in m.items()}


# ------------------------------------------------------------------ emotion module
def emotion_module(train, valid, test, tag, labels):
    """Two classifiers on the fixed emotion encoder: one for queries, one for replies."""
    lab2i = {l: i for i, l in enumerate(labels)}
    out = {}
    if EMO_VARIANT == "ft":
        for side in ("q", "d"):
            for split in ("valid", "test"):
                out[(side, split)] = np.load(os.path.join(CACHE, f"ft__{tag}__{side}__{split}.npy"))
            yte = np.array([lab2i[r["emotion"]] for r in test])
            out[(side, "acc_test")] = float((out[(side, "test")].argmax(1) == yte).mean())
            out[(side, "C")] = None
        return out

    def feats(field, kind, split, recs):
        X = embed(EMO_ENCODER, [r[field] for r in recs], f"{tag}_{split}", kind)
        if EMO_VARIANT == "go":
            G = np.load(os.path.join(CACHE, f"go__{tag}_{split}__{kind}.npy"))
            X = np.hstack([X, G / 4.0])     # logits rescaled to a range comparable with the unit-norm embedding
        return X

    for side, field, kind in (("q", "query", "q"), ("d", "target", "d")):
        Xtr = feats(field, kind, "train", train)
        ytr = np.array([lab2i[r["emotion"]] for r in train])
        Xva = feats(field, kind, "valid", valid)
        yva = np.array([lab2i[r["emotion"]] for r in valid])
        best = None
        for C in (0.25, 1.0, 4.0):
            clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
            acc = (clf.predict(Xva) == yva).mean()
            if best is None or acc > best[0]:
                best = (acc, C, clf)
        clf = best[2]
        for split, recs in (("valid", valid), ("test", test)):
            X = feats(field, kind, split, recs)
            P = np.zeros((len(recs), len(labels)), dtype=np.float32)
            P[:, clf.classes_] = clf.predict_proba(X)
            out[(side, split)] = P
        yte = np.array([lab2i[r["emotion"]] for r in test])
        out[(side, "acc_test")] = float((out[(side, "test")].argmax(1) == yte).mean())
        out[(side, "C")] = best[1]
    return out


def conf_from_probs(P):
    H = -(P * np.log(P + 1e-12)).sum(1) / math.log(P.shape[1])
    return 1.0 - H


# ------------------------------------------------------------------ methods
def m_comb(Sz, Pq, Pd, lam):
    Pqn = Pq / (np.linalg.norm(Pq, axis=1, keepdims=True) + 1e-9)
    Pdn = Pd / (np.linalg.norm(Pd, axis=1, keepdims=True) + 1e-9)
    return Sz + lam * (Pqn @ Pdn.T)


def m_cgp(Sz, Pq, Pd, cq, lam, mu, pos_idx, neg_idx):
    Pqn = Pq / (np.linalg.norm(Pq, axis=1, keepdims=True) + 1e-9)
    Pdn = Pd / (np.linalg.norm(Pd, axis=1, keepdims=True) + 1e-9)
    emo = Pqn @ Pdn.T
    if len(pos_idx) and len(neg_idx):
        qp, qn = Pq[:, pos_idx].sum(1), Pq[:, neg_idx].sum(1)
        dp_, dn = Pd[:, pos_idx].sum(1), Pd[:, neg_idx].sum(1)
        conflict = np.outer(qp, dn) + np.outer(qn, dp_)
    else:
        conflict = 0.0
    return Sz + cq[:, None] * (lam * emo - mu * conflict)


def m_seq(Sz, Pq, Pd, N):
    Pqn = Pq / (np.linalg.norm(Pq, axis=1, keepdims=True) + 1e-9)
    Pdn = Pd / (np.linalg.norm(Pd, axis=1, keepdims=True) + 1e-9)
    emo = Pqn @ Pdn.T
    out = Sz.copy()                      # candidates outside the top-N keep their semantic order
    top = np.argsort(-Sz, axis=1)[:, :N]
    rows = np.arange(Sz.shape[0])[:, None]
    # inside the top-N, order by emotion similarity, semantic score as a small tie-breaker
    out[rows, top] = 1000 + emo[rows, top] * 100 + Sz[rows, top] * 1e-3
    return out


def m_mult(S, Pq, Pd):
    """Emotional RAG 'C-M': product of the (min-max scaled) semantic similarity and the emotion similarity."""
    Pqn = Pq / (np.linalg.norm(Pq, axis=1, keepdims=True) + 1e-9)
    Pdn = Pd / (np.linalg.norm(Pd, axis=1, keepdims=True) + 1e-9)
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return ((S - lo) / (hi - lo + 1e-9)) * (Pqn @ Pdn.T)


def m_seq_emo_first(Sz, Pq, Pd, N):
    """Emotional RAG 'S-E': top-N by emotion similarity, then re-ranked by semantic score."""
    Pqn = Pq / (np.linalg.norm(Pq, axis=1, keepdims=True) + 1e-9)
    Pdn = Pd / (np.linalg.norm(Pd, axis=1, keepdims=True) + 1e-9)
    emo = Pqn @ Pdn.T + Sz * 1e-6          # semantic score breaks ties among equal emotion scores
    out = Sz.copy()                        # candidates outside the top-N keep their semantic order
    top = np.argsort(-emo, axis=1)[:, :N]
    rows = np.arange(Sz.shape[0])[:, None]
    out[rows, top] = 1000 + Sz[rows, top]
    return out


def m_filter(Sz, Pq, Pd):
    same = Pq.argmax(1)[:, None] == Pd.argmax(1)[None, :]
    return np.where(same, Sz + 1000, Sz)


def select(metrics_fn, grid, base_mrr):
    """argmax EmoAcc@10 s.t. MRR >= 0.99 base MRR (falls back to best MRR if none qualifies)."""
    best, best_ok = None, None
    for params in grid:
        s = summarise(metrics_fn(*params))
        if s["MRR"] >= 0.99 * base_mrr and (best_ok is None or s["EmoAcc@10"] > best_ok[1]["EmoAcc@10"]):
            best_ok = (params, s)
        if best is None or s["MRR"] > best[1]["MRR"]:
            best = (params, s)
    return (best_ok or best)[0]


def paired_test(a, b, iters=10000):
    """Two-sided paired randomisation test on per-query values."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    obs = abs(d.mean())
    signs = RNG.choice([-1, 1], size=(iters, len(d)))
    return float(((np.abs((signs * d).mean(1))) >= obs).mean())


# ------------------------------------------------------------------ main loop
def run_dataset(tag, train, valid, test, use_polarity):
    labels = sorted({r["emotion"] for r in train + valid + test})
    lab2i = {l: i for i, l in enumerate(labels)}
    pos_idx = [lab2i[l] for l in labels if dp.polarity(l) > 0] if use_polarity else []
    neg_idx = [lab2i[l] for l in labels if dp.polarity(l) < 0] if use_polarity else []
    emo = emotion_module(train, valid, test, tag, labels)
    print(f"[{tag}] emotion classifier accuracy (test): queries {emo[('q','acc_test')]:.3f}, replies {emo[('d','acc_test')]:.3f}", flush=True)

    def arrays(recs):
        e = np.array([lab2i[r["emotion"]] for r in recs])
        p = np.array([dp.polarity(r["emotion"]) if use_polarity else 0 for r in recs])
        return e, p

    res = {"labels": labels, "emotion_clf": {"query_acc": emo[("q", "acc_test")], "reply_acc": emo[("d", "acc_test")]},
           "retrievers": {}}
    per_query = {}
    retrievers = ["BM25", "TF-IDF"] + list(DENSE) + ["CE-rerank"]
    if os.environ.get("RETRIEVERS"):
        retrievers = os.environ["RETRIEVERS"].split(",")
    rng_probe = np.random.default_rng(1)
    probe_idx = rng_probe.choice(len(train), size=min(PROBE_N, len(train)), replace=False)

    for name in retrievers:
        t0 = time.time()
        S = {}
        for split, recs in (("valid", valid), ("test", test)):
            qs, ds = [r["query"] for r in recs], [r["target"] for r in recs]
            if name == "BM25":
                bm = BM25Okapi([dp_tok(x) for x in ds])
                S[split] = np.array([bm.get_scores(dp_tok(q)) for q in qs], dtype=np.float32)
            elif name == "TF-IDF":
                v = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit([r["target"] for r in train])
                S[split] = (v.transform(qs) @ v.transform(ds).T).toarray().astype(np.float32)
            elif name == "CE-rerank":
                S[split] = ce_scores(tag, split, qs, ds)
            else:
                Q = embed(name, qs, f"{tag}_{split}", "q"); D = embed(name, ds, f"{tag}_{split}", "d")
                S[split] = Q @ D.T
        # linear probe: how much emotion information do the query embeddings carry?
        if name in DENSE:
            Xp = embed(name, [train[i]["query"] for i in probe_idx], f"{tag}_probe", "q")
            yp = np.array([lab2i[train[i]["emotion"]] for i in probe_idx])
            Xt = embed(name, [r["query"] for r in test], f"{tag}_test", "q")
            probe = LogisticRegression(C=1.0, max_iter=2000).fit(Xp, yp)
            probe_acc = float((probe.predict(Xt) == np.array([lab2i[r["emotion"]] for r in test])).mean())
        else:
            probe_acc = None

        ev, pv = arrays(valid); et, pt = arrays(test)
        Szv, Szt = zrows(S["valid"]), zrows(S["test"])
        Pqv, Pdv, Pqt, Pdt = emo[("q", "valid")], emo[("d", "valid")], emo[("q", "test")], emo[("d", "test")]
        cqv, cqt = conf_from_probs(Pqv), conf_from_probs(Pqt)
        onehot = lambda e: np.eye(len(labels), dtype=np.float32)[e]

        base_v = summarise(evaluate(Szv, ev, ev, pv, pv))
        LAMS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
        lam_grid = [(x,) for x in LAMS]
        lam = select(lambda l: evaluate(m_comb(Szv, Pqv, Pdv, l), ev, ev, pv, pv), lam_grid, base_v["MRR"])[0]
        N = select(lambda n: evaluate(m_seq(Szv, Pqv, Pdv, n), ev, ev, pv, pv), [(n,) for n in (2, 3, 5, 10, 20, 50, 100)], base_v["MRR"])[0]
        N_e = select(lambda n: evaluate(m_seq_emo_first(Szv, Pqv, Pdv, n), ev, ev, pv, pv),
                     [(n,) for n in (50, 100, 200, 500, 1000)], base_v["MRR"])[0]
        cgp_grid = [(l, m) for l in LAMS for m in ((0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0) if use_polarity else (0.0,))]
        lam_c, mu_c = select(lambda l, m: evaluate(m_cgp(Szv, Pqv, Pdv, cqv, l, m, pos_idx, neg_idx), ev, ev, pv, pv), cgp_grid, base_v["MRR"])
        lam_o = select(lambda l: evaluate(m_comb(Szv, onehot(ev), onehot(ev), l), ev, ev, pv, pv), lam_grid, base_v["MRR"])[0]

        methods = {
            "base": Szt,
            "filter": m_filter(Szt, Pqt, Pdt),
            "comb": m_comb(Szt, Pqt, Pdt, lam),
            "mult": m_mult(S["test"], Pqt, Pdt),
            "seq": m_seq(Szt, Pqt, Pdt, N),
            "seq_emo": m_seq_emo_first(Szt, Pqt, Pdt, N_e),
            "cgp": m_cgp(Szt, Pqt, Pdt, cqt, lam_c, mu_c, pos_idx, neg_idx),
            "oracle": m_comb(Szt, onehot(et), onehot(et), lam_o),
        }
        r = {"params": {"comb_lambda": lam, "seq_N": N, "seq_emo_N": N_e, "cgp_lambda": lam_c, "cgp_mu": mu_c, "oracle_lambda": lam_o},
             "probe_acc": probe_acc, "methods": {}}
        pq = {}
        for mname, M in methods.items():
            m = evaluate(M, et, et, pt, pt)
            pq[mname] = m
            r["methods"][mname] = summarise(m)
        # significance of cgp against base and against comb
        r["p_values"] = {f"cgp_vs_{b}_{k}": paired_test(pq["cgp"][k], pq[b][k])
                         for b in ("base", "comb") for k in ("MRR", "EmoAcc@10") + (("Harm@10",) if use_polarity else ())}
        res["retrievers"][name] = r
        per_query[name] = {m: {k: v.tolist() for k, v in d.items()} for m, d in pq.items()}
        b, c = r["methods"]["base"], r["methods"]["cgp"]
        print(f"[{tag}] {name:20s} {time.time()-t0:6.0f}s  probe={probe_acc}  base MRR {b['MRR']:.3f} Emo {b['EmoAcc@10']:.3f} Harm {b['Harm@10']:.3f} | "
              f"comb Emo {r['methods']['comb']['EmoAcc@10']:.3f} | cgp MRR {c['MRR']:.3f} Emo {c['EmoAcc@10']:.3f} Harm {c['Harm@10']:.3f} | oracle Emo {r['methods']['oracle']['EmoAcc@10']:.3f}", flush=True)
        json.dump(res, open(os.path.join(OUT, f"{tag}{SUFFIX}_results.json"), "w"), indent=1)
    json.dump(per_query, open(os.path.join(OUT, f"{tag}{SUFFIX}_per_query.json"), "w"))
    return res


def dp_tok(s):
    import re
    return re.findall(r"[a-z0-9']+", s.lower())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ed", "all"):
        run_dataset("ED", dp.load_ed("train"), dp.load_ed("valid"), dp.load_ed("test"), use_polarity=True)
    if which in ("esconv", "all"):
        es = dp.load_esconv()
        run_dataset("ESConv", es["train"], es["valid"], es["test"], use_polarity=False)
    print("done", flush=True)
