"""Additional analyses on cached embeddings (run after run_experiments.py).

1. Trade-off curves: MRR vs EmoAcc@10 / Harm@10 as the emotion weight grows (comb vs cgp).
2. Robustness to emotion-classifier quality: classifiers trained on 10/25/50/100 % of the
   training conversations; CGP parameters re-tuned on validation each time.
3. Per-emotion gains of CGP over the base retriever.
4. Probe accuracy vs emotional consistency of retrieval, across encoders.
Outputs JSON in results/ and PDF figures in paper/images/.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as rx
import data_prep as dp
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

OUT = rx.OUT
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
MAIN = "gte-small"   # detailed analyses: strong on both corpora and distinct from the emotion encoder (mpnet)


def setup(tag):
    if tag == "ED":
        tr, va, te = dp.load_ed("train"), dp.load_ed("valid"), dp.load_ed("test")
        use_pol = True
    else:
        es = dp.load_esconv(); tr, va, te = es["train"], es["valid"], es["test"]
        use_pol = False
    labels = sorted({r["emotion"] for r in tr + va + te})
    lab2i = {l: i for i, l in enumerate(labels)}
    pos = [lab2i[l] for l in labels if dp.polarity(l) > 0] if use_pol else []
    neg = [lab2i[l] for l in labels if dp.polarity(l) < 0] if use_pol else []
    arr = lambda recs: (np.array([lab2i[r["emotion"]] for r in recs]),
                        np.array([dp.polarity(r["emotion"]) if use_pol else 0 for r in recs]))
    return tr, va, te, labels, lab2i, pos, neg, arr, use_pol


def sem(name, tag, split, recs):
    Q = rx.embed(name, [r["query"] for r in recs], f"{tag}_{split}", "q")
    D = rx.embed(name, [r["target"] for r in recs], f"{tag}_{split}", "d")
    return rx.zrows(Q @ D.T)


def classifier_probs(tag, tr, va, te, labels, lab2i, frac, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tr), size=max(len(labels) * 3, int(frac * len(tr))), replace=False)
    out = {}
    for side, field in (("q", "query"), ("d", "target")):
        Xtr = rx.embed(rx.EMO_ENCODER, [r[field] for r in tr], f"{tag}_train", side)[idx]
        ytr = np.array([lab2i[tr[i]["emotion"]] for i in idx])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
        for split, recs in (("valid", va), ("test", te)):
            X = rx.embed(rx.EMO_ENCODER, [r[field] for r in recs], f"{tag}_{split}", side)
            P = np.zeros((len(recs), len(labels)), dtype=np.float32)
            P[:, clf.classes_] = clf.predict_proba(X)
            out[(side, split)] = P
    yte = np.array([lab2i[r["emotion"]] for r in te])
    out["acc_q"] = float((out[("q", "test")].argmax(1) == yte).mean())
    return out


def main(tag, main_retriever):
    tr, va, te, labels, lab2i, pos, neg, arr, use_pol = setup(tag)
    ev, pv = arr(va); et, pt = arr(te)
    Szv, Szt = sem(main_retriever, tag, "valid", va), sem(main_retriever, tag, "test", te)
    res = {"retriever": main_retriever}

    # 1. trade-off curves on test (for the figure), full classifier
    P = classifier_probs(tag, tr, va, te, labels, lab2i, 1.0)
    cq = rx.conf_from_probs(P[("q", "test")])
    _, mu_best = json.load(open(os.path.join(OUT, f"{tag}_results.json")))["retrievers"][main_retriever]["params"]["cgp_lambda"], \
        json.load(open(os.path.join(OUT, f"{tag}_results.json")))["retrievers"][main_retriever]["params"]["cgp_mu"]
    curves = {"comb": [], "cgp": []}
    for lam in (0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        s = rx.summarise(rx.evaluate(rx.m_comb(Szt, P[("q", "test")], P[("d", "test")], lam), et, et, pt, pt))
        curves["comb"].append({"lam": lam, **s})
        s = rx.summarise(rx.evaluate(rx.m_cgp(Szt, P[("q", "test")], P[("d", "test")], cq, lam, mu_best, pos, neg), et, et, pt, pt))
        curves["cgp"].append({"lam": lam, **s})
    res["curves"] = curves

    # 2. robustness to classifier quality (re-tuned on validation each time)
    rob = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        Pf = classifier_probs(tag, tr, va, te, labels, lab2i, frac)
        cqv, cqt = rx.conf_from_probs(Pf[("q", "valid")]), rx.conf_from_probs(Pf[("q", "test")])
        base_v = rx.summarise(rx.evaluate(Szv, ev, ev, pv, pv))
        LAMS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
        grid_c = [(x,) for x in LAMS]
        lam = rx.select(lambda l: rx.evaluate(rx.m_comb(Szv, Pf[("q", "valid")], Pf[("d", "valid")], l), ev, ev, pv, pv), grid_c, base_v["MRR"])[0]
        grid = [(l, m) for l in LAMS for m in ((0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0) if use_pol else (0.0,))]
        l2, m2 = rx.select(lambda l, m: rx.evaluate(rx.m_cgp(Szv, Pf[("q", "valid")], Pf[("d", "valid")], cqv, l, m, pos, neg), ev, ev, pv, pv), grid, base_v["MRR"])
        row = {"frac": frac, "clf_acc_q": Pf["acc_q"],
               "base": rx.summarise(rx.evaluate(Szt, et, et, pt, pt)),
               "comb": rx.summarise(rx.evaluate(rx.m_comb(Szt, Pf[("q", "test")], Pf[("d", "test")], lam), et, et, pt, pt)),
               "cgp": rx.summarise(rx.evaluate(rx.m_cgp(Szt, Pf[("q", "test")], Pf[("d", "test")], cqt, l2, m2, pos, neg), et, et, pt, pt))}
        rob.append(row)
        print(tag, "frac", frac, "acc", round(Pf["acc_q"], 3), "comb Emo", round(row["comb"]["EmoAcc@10"], 3),
              "cgp Emo", round(row["cgp"]["EmoAcc@10"], 3), "cgp MRR", round(row["cgp"]["MRR"], 3), flush=True)
    res["robustness"] = rob

    # 3. per-emotion gains (from the stored per-query results)
    pq = json.load(open(os.path.join(OUT, f"{tag}_per_query.json")))[main_retriever]
    per = {}
    for i, l in enumerate(labels):
        m = et == i
        if m.sum() < 5:
            continue
        b = np.array(pq["base"]["EmoAcc@10"])[m].mean(); c = np.array(pq["cgp"]["EmoAcc@10"])[m].mean()
        bm = np.array(pq["base"]["MRR"])[m].mean(); cm = np.array(pq["cgp"]["MRR"])[m].mean()
        per[l] = {"n": int(m.sum()), "base_emo": float(b), "cgp_emo": float(c), "base_mrr": float(bm), "cgp_mrr": float(cm)}
    res["per_emotion"] = per
    json.dump(res, open(os.path.join(OUT, f"{tag}_analysis.json"), "w"), indent=1)
    return res


def simulated_accuracy(tag, main_retriever, accs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), seeds=3):
    """What would a better emotion recognizer buy? Simulated recognizers that output the true label
    with probability `acc` (otherwise a random other label), for queries and candidates alike.
    C-A is re-tuned on validation for every accuracy level; results averaged over seeds."""
    tr, va, te, labels, lab2i, pos, neg, arr, use_pol = setup(tag)
    ev, pv = arr(va); et, pt = arr(te)
    Szv, Szt = sem(main_retriever, tag, "valid", va), sem(main_retriever, tag, "test", te)
    L = len(labels)
    base_v = rx.summarise(rx.evaluate(Szv, ev, ev, pv, pv))
    out = []
    for acc in accs:
        rows = []
        for seed in range(seeds):
            rng = np.random.default_rng(100 + seed)
            def noisy(e):
                keep = rng.random(len(e)) < acc
                other = (e + rng.integers(1, L, size=len(e))) % L
                return np.eye(L, dtype=np.float32)[np.where(keep, e, other)]
            Qv, Dv, Qt, Dt = noisy(ev), noisy(ev), noisy(et), noisy(et)
            grid = [(x,) for x in (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32)]
            lam = rx.select(lambda l: rx.evaluate(rx.m_comb(Szv, Qv, Dv, l), ev, ev, pv, pv), grid, base_v["MRR"])[0]
            rows.append(rx.summarise(rx.evaluate(rx.m_comb(Szt, Qt, Dt, lam), et, et, pt, pt)))
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
        out.append({"acc": acc, **mean})
        print(tag, "simulated acc", acc, {k: round(v, 3) for k, v in mean.items()}, flush=True)
    a = json.load(open(os.path.join(OUT, f"{tag}_analysis.json")))
    a["simulated_accuracy"] = out
    json.dump(a, open(os.path.join(OUT, f"{tag}_analysis.json"), "w"), indent=1)


def figures():
    ed = json.load(open(os.path.join(OUT, "ED_analysis.json")))
    edres = json.load(open(os.path.join(OUT, "ED_results.json")))
    plt.rcParams.update({"font.size": 8, "figure.dpi": 200})

    # trade-off
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for name, mk in (("comb", "o--"), ("cgp", "s-")):
        c = ed["curves"][name]
        ax[0].plot([x["EmoAcc@10"] for x in c], [x["MRR"] for x in c], mk, ms=3, label={"comb": "Combination (C-A)", "cgp": "CGP"}[name])
        ax[1].plot([x["Harm@10"] for x in c], [x["MRR"] for x in c], mk, ms=3, label={"comb": "Combination (C-A)", "cgp": "CGP"}[name])
    ax[0].set_xlabel("EmoAcc@10 (higher is better)"); ax[0].set_ylabel("MRR")
    ax[1].set_xlabel("Harm@10 (lower is better)"); ax[1].set_ylabel("MRR")
    for a in ax: a.grid(alpha=.3); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tradeoff_ed.pdf")); plt.close(fig)

    # probe vs retrieval
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    offsets = {"all-MiniLM-L6-v2": (-38, 4), "bge-small-en-v1.5": (4, 5), "e5-small-v2": (4, -7), "all-mpnet-base-v2": (-52, -3)}
    for name, r in edres["retrievers"].items():
        if r["probe_acc"] is None: continue
        ax.scatter(r["probe_acc"], r["methods"]["base"]["EmoAcc@10"], s=12)
        ax.annotate(name.replace("-v1.5", "").replace("-v2", ""), (r["probe_acc"], r["methods"]["base"]["EmoAcc@10"]), fontsize=6,
                    xytext=offsets.get(name, (3, 2)), textcoords="offset points")
    ax.set_xlabel("Linear-probe emotion accuracy"); ax.set_ylabel("EmoAcc@10 of retrieval")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(FIG, "probe_vs_retrieval_ed.pdf")); plt.close(fig)

    # robustness
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    rob = ed["robustness"]
    x = [r["clf_acc_q"] for r in rob]
    ax.plot(x, [r["base"]["EmoAcc@10"] for r in rob], "k:", label="Base")
    ax.plot(x, [r["comb"]["EmoAcc@10"] for r in rob], "o--", ms=3, label="Combination (C-A)")
    ax.plot(x, [r["cgp"]["EmoAcc@10"] for r in rob], "s-", ms=3, label="CGP")
    ax.set_xlabel("Emotion classifier accuracy (queries)"); ax.set_ylabel("EmoAcc@10")
    ax.grid(alpha=.3); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness_ed.pdf")); plt.close(fig)

    # simulated recognizer accuracy (both corpora)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for tag, mk in (("ED", "s-"), ("ESConv", "o--")):
        p = os.path.join(OUT, f"{tag}_analysis.json")
        sim = json.load(open(p)).get("simulated_accuracy")
        if not sim: continue
        x = [r["acc"] for r in sim]
        ax[0].plot(x, [r["EmoAcc@10"] for r in sim], mk, ms=3, label=tag)
        ax[1].plot(x, [r["MRR"] for r in sim], mk, ms=3, label=tag)
    ax[0].set_xlabel("Simulated recognizer accuracy"); ax[0].set_ylabel("EmoAcc@10")
    ax[1].set_xlabel("Simulated recognizer accuracy"); ax[1].set_ylabel("MRR")
    for a in ax: a.grid(alpha=.3); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "simulated_accuracy.pdf")); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ED"):
        main("ED", sys.argv[2] if len(sys.argv) > 2 else MAIN)
    if which in ("all", "ESConv"):
        main("ESConv", sys.argv[2] if len(sys.argv) > 2 else MAIN)
    if which in ("all", "sim"):
        simulated_accuracy("ED", MAIN)
        simulated_accuracy("ESConv", MAIN)
    if which in ("all", "fig", "sim"):
        figures()
