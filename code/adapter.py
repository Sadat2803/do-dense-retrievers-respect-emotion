"""Emotion-aware embedding adapters on top of frozen retrievers (CPU).

For a frozen dense retriever, learn residual linear maps for queries and replies,
    q' = norm(q + A_q q),   d' = norm(d + A_d d)      (A initialised at zero = identity start),
from training (query, gold reply, emotion) triples, with the loss
    L = CE(softmax(q' d'^T / tau), target) + gamma * E_softmax[opposite valence]
target = (1-beta) on the gold reply + beta spread over in-batch replies with the query's emotion.
  * Topical adapter (TA):        beta = 0, gamma = 0  (task adaptation only, control)
  * Emotion-aware adapter (EA):  beta > 0 and/or gamma > 0, chosen on validation
Training uses all training pairs (19,531 on ED, 909 on ESConv); epochs are chosen
on validation MRR (TA) or by the paper's rule (EA: max EmoAcc@10 with MRR >= 0.99 MRR(TA)).
Outputs results/{tag}_adapter_results.json and per-query values for significance tests.
Usage: python adapter.py [ED|ESConv|all] [retriever,...]
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as rx
import data_prep as dp

torch.set_num_threads(os.cpu_count())
OUT = rx.OUT
DENSE = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6", "bge-small-en-v1.5", "e5-small-v2", "gte-small",
         "bge-base-en-v1.5", "e5-base-v2", "gte-base"]
TAU = 0.05


def setup(tag):
    if tag == "ED":
        tr, va, te = dp.load_ed("train"), dp.load_ed("valid"), dp.load_ed("test")
        use_pol = True
    else:
        es = dp.load_esconv(); tr, va, te = es["train"], es["valid"], es["test"]
        use_pol = False
    labels = sorted({r["emotion"] for r in tr + va + te})
    return tr, va, te, labels, use_pol


class Adapter(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Aq = torch.nn.Linear(dim, dim, bias=False)
        self.Ad = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.zeros_(self.Aq.weight); torch.nn.init.zeros_(self.Ad.weight)

    def q(self, x):
        return torch.nn.functional.normalize(x + self.Aq(x), dim=-1)

    def d(self, x):
        return torch.nn.functional.normalize(x + self.Ad(x), dim=-1)


def train_adapter(Qtr, Dtr, etr, vtr, beta, gamma, epochs, seed=0, bs=256, lr=3e-4, wd=1e-3, eval_fn=None):
    """Returns a list of (epoch, state_dict copy) snapshots so the epoch can be chosen on validation."""
    torch.manual_seed(seed)
    model = Adapter(Qtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Q, D = torch.tensor(Qtr), torch.tensor(Dtr)
    e, v = torch.tensor(etr), torch.tensor(vtr)
    n = len(Q)
    snaps = []
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            q, d = model.q(Q[b]), model.d(D[b])
            logits = q @ d.T / TAU
            same = (e[b][:, None] == e[b][None, :]).float()
            eye = torch.eye(len(b))
            others = same * (1 - eye)
            cnt = others.sum(1, keepdim=True)
            target = (1 - beta) * eye + beta * torch.where(cnt > 0, others / cnt.clamp(min=1), eye)
            loss = -(target * torch.log_softmax(logits, 1)).sum(1).mean()
            if gamma > 0:
                opp = ((v[b][:, None] * v[b][None, :]) < 0).float()
                loss = loss + gamma * (torch.softmax(logits, 1) * opp).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        snaps.append((ep, {k: t.clone() for k, t in model.state_dict().items()}))
    return model, snaps


def apply(model, state, Q, D):
    model.load_state_dict(state)
    with torch.no_grad():
        return (model.q(torch.tensor(Q)) @ model.d(torch.tensor(D)).T).numpy()


def run(tag, retrievers):
    tr, va, te, labels, use_pol = setup(tag)
    l2i = {l: i for i, l in enumerate(labels)}
    pos = [l2i[l] for l in labels if dp.polarity(l) > 0] if use_pol else []
    neg = [l2i[l] for l in labels if dp.polarity(l) < 0] if use_pol else []
    arr = lambda recs: (np.array([l2i[r["emotion"]] for r in recs]), np.array([dp.polarity(r["emotion"]) if use_pol else 0 for r in recs]))
    ev, pv = arr(va); et, pt = arr(te)
    trs = tr                                   # all training pairs
    etr, vtr = arr(trs)
    emo = rx.emotion_module(tr, va, te, tag, labels)
    Pqv, Pdv, Pqt, Pdt = emo[("q", "valid")], emo[("d", "valid")], emo[("q", "test")], emo[("d", "test")]
    LAMS = [(x,) for x in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)]
    path = os.path.join(OUT, f"{tag}_adapter_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    pq_path = os.path.join(OUT, f"{tag}_adapter_per_query.json")
    per_query = json.load(open(pq_path)) if os.path.exists(pq_path) else {}
    EPOCHS = 12 if tag == "ED" else 40
    for name in retrievers:
        t0 = time.time()
        Qtr = rx.embed(name, [r["query"] for r in trs], f"{tag}_train", "q")
        Dtr = rx.embed(name, [r["target"] for r in trs], f"{tag}_train", "d")
        Qv, Dv = rx.embed(name, [r["query"] for r in va], f"{tag}_valid", "q"), rx.embed(name, [r["target"] for r in va], f"{tag}_valid", "d")
        Qt, Dt = rx.embed(name, [r["query"] for r in te], f"{tag}_test", "q"), rx.embed(name, [r["target"] for r in te], f"{tag}_test", "d")
        out = {}

        # ---- topical adapter: epoch chosen on validation MRR
        model, snaps = train_adapter(Qtr, Dtr, etr, vtr, 0.0, 0.0, EPOCHS)
        best = max(snaps, key=lambda s: rx.summarise(rx.evaluate(apply(model, s[1], Qv, Dv), ev, ev, pv, pv))["MRR"])
        S_ta_v, S_ta_t = apply(model, best[1], Qv, Dv), apply(model, best[1], Qt, Dt)
        ta_mrr_v = rx.summarise(rx.evaluate(S_ta_v, ev, ev, pv, pv))["MRR"]
        out["TA"] = {"epoch": best[0]}

        # ---- emotion-aware adapter: (beta, gamma, epoch) by the paper's rule against the topical adapter
        cands = []
        for beta in (0.1, 0.25, 0.5):
            for gamma in ((0.0, 1.0, 4.0) if use_pol else (0.0,)):
                m2, sn2 = train_adapter(Qtr, Dtr, etr, vtr, beta, gamma, EPOCHS)
                for ep, st in sn2:
                    s = rx.summarise(rx.evaluate(apply(m2, st, Qv, Dv), ev, ev, pv, pv))
                    cands.append((s, beta, gamma, ep, st))
        ok = [c for c in cands if c[0]["MRR"] >= 0.99 * ta_mrr_v]
        pick = max(ok, key=lambda c: c[0]["EmoAcc@10"]) if ok else max(cands, key=lambda c: c[0]["MRR"])
        _, beta, gamma, ep, st = pick
        S_ea_v, S_ea_t = apply(model, st, Qv, Dv), apply(model, st, Qt, Dt)
        out["EA"] = {"beta": beta, "gamma": gamma, "epoch": ep, "constraint_met": bool(ok)}

        # ---- metrics on test: base, C-A, TA, TA + C-A, EA, EA + C-A, EA + CGP
        methods, pqm = {}, {}
        def add(label, Sv, St, fuse):
            nonlocal methods
            Szv, Szt = rx.zrows(Sv), rx.zrows(St)
            if fuse is None:
                M = Szt
            else:
                bv = rx.summarise(rx.evaluate(Szv, ev, ev, pv, pv))["MRR"]
                if fuse == "ca":
                    lam = rx.select(lambda l: rx.evaluate(rx.m_comb(Szv, Pqv, Pdv, l), ev, ev, pv, pv), LAMS, bv)[0]
                    M = rx.m_comb(Szt, Pqt, Pdt, lam); out[label] = {"lambda": lam}
                else:
                    cqv, cqt = rx.conf_from_probs(Pqv), rx.conf_from_probs(Pqt)
                    grid = [(l, m) for (l,) in LAMS for m in ((0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0) if use_pol else (0.0,))]
                    l2, m2_ = rx.select(lambda l, m: rx.evaluate(rx.m_cgp(Szv, Pqv, Pdv, cqv, l, m, pos, neg), ev, ev, pv, pv), grid, bv)
                    M = rx.m_cgp(Szt, Pqt, Pdt, cqt, l2, m2_, pos, neg); out[label] = {"lambda": l2, "mu": m2_}
            m = rx.evaluate(M, et, et, pt, pt)
            methods[label] = rx.summarise(m); pqm[label] = m

        Q0v, D0v = Qv @ Dv.T, None
        add("Base", Qv @ Dv.T, Qt @ Dt.T, None)
        add("Base+C-A", Qv @ Dv.T, Qt @ Dt.T, "ca")
        add("TA", S_ta_v, S_ta_t, None)
        add("TA+C-A", S_ta_v, S_ta_t, "ca")
        add("EA", S_ea_v, S_ea_t, None)
        add("EA+C-A", S_ea_v, S_ea_t, "ca")
        add("EA+CGP", S_ea_v, S_ea_t, "cgp")
        keys = ("MRR", "EmoAcc@10") + (("Harm@10",) if use_pol else ())
        pvals = {}
        for a, b in (("EA", "TA"), ("EA", "Base+C-A"), ("EA+C-A", "TA+C-A"), ("EA+C-A", "Base+C-A"), ("TA", "Base")):
            for k in keys:
                pvals[f"{a}_vs_{b}_{k}"] = rx.paired_test(pqm[a][k], pqm[b][k])
        res[name] = {"params": out, "methods": methods, "p_values": pvals}
        per_query[name] = {m: {k: v.tolist() for k, v in d.items()} for m, d in pqm.items()}
        json.dump(res, open(path, "w"), indent=1); json.dump(per_query, open(pq_path, "w"))
        f = lambda m: f"MRR {methods[m]['MRR']:.3f} Emo {methods[m]['EmoAcc@10']:.3f} Harm {methods[m]['Harm@10']:.3f}"
        print(f"[{tag}] {name:20s} {time.time()-t0:5.0f}s | Base {f('Base')} | Base+C-A {f('Base+C-A')} | TA {f('TA')} | TA+C-A {f('TA+C-A')} | EA {f('EA')} (b={beta},g={gamma},ep={ep}) | EA+C-A {f('EA+C-A')}", flush=True)


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    rets = sys.argv[2].split(",") if len(sys.argv) > 2 else DENSE
    for tag in (("ED", "ESConv") if w == "all" else (w,)):
        run(tag, rets)
    print("done", flush=True)
