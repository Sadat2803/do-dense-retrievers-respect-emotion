"""Quick exploration (validation split only): how far can the emotion objective move the adapter?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapter as ad, run_experiments as rx, data_prep as dp

name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"
tr, va, te, labels, use_pol = ad.setup("ED")
l2i = {l: i for i, l in enumerate(labels)}
arr = lambda recs: (np.array([l2i[r["emotion"]] for r in recs]), np.array([dp.polarity(r["emotion"]) for r in recs]))
ev, pv = arr(va)
idx = np.random.default_rng(1).choice(len(tr), size=rx.PROBE_N, replace=False)
trs = [tr[i] for i in idx]; etr, vtr = arr(trs)
Qtr = rx.embed(name, [r["query"] for r in trs], "ED_probe", "q"); Dtr = rx.embed(name, [r["target"] for r in trs], "ED_probe", "d")
Qv = rx.embed(name, [r["query"] for r in va], "ED_valid", "q"); Dv = rx.embed(name, [r["target"] for r in va], "ED_valid", "d")
for beta, gamma in ((0.0, 0.0), (0.5, 4.0), (0.75, 8.0), (0.9, 16.0), (0.95, 16.0)):
    m, sn = ad.train_adapter(Qtr, Dtr, etr, vtr, beta, gamma, 30)
    best = None
    for ep, st in sn:
        if ep % 5: continue
        s = rx.summarise(rx.evaluate(ad.apply(m, st, Qv, Dv), ev, ev, pv, pv))
        print(f"beta {beta} gamma {gamma} ep {ep:2d}: MRR {s['MRR']:.3f} Emo {s['EmoAcc@10']:.3f} Harm {s['Harm@10']:.3f}", flush=True)
