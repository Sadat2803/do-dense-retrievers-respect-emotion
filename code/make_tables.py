"""Generates the LaTeX tables of the paper from the result JSON files (no hand-copied numbers)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

R = os.path.join(ROOT, "results")
T = os.path.join(ROOT, "tables")
os.makedirs(T, exist_ok=True)
ED = json.load(open(os.path.join(R, "ED_results.json")))
ES = json.load(open(os.path.join(R, "ESConv_results.json")))
DENSE = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6", "bge-small-en-v1.5", "e5-small-v2", "gte-small",
         "bge-base-en-v1.5", "e5-base-v2", "gte-base"]
ALL = ["BM25", "TF-IDF"] + DENSE + ["CE-rerank"]
NICE = {"BM25": "BM25", "TF-IDF": "TF-IDF", "all-MiniLM-L6-v2": "MiniLM-L6", "all-mpnet-base-v2": "MPNet-base",
        "multi-qa-MiniLM-L6": "MultiQA-MiniLM", "bge-small-en-v1.5": "BGE-small", "e5-small-v2": "E5-small", "gte-small": "GTE-small",
        "bge-base-en-v1.5": "BGE-base", "e5-base-v2": "E5-base", "gte-base": "GTE-base", "CE-rerank": "GTE-base + CE"}
SEP_AFTER = {"TF-IDF", "gte-base"}     # horizontal rules after lexical and after dense retrievers
METH = [("base", "Base"), ("filter", "Filter"), ("comb", "C-A"), ("mult", "C-M"), ("seq", "S-S"), ("seq_emo", "S-E"),
        ("cgp", "CGP"), ("oracle", "Oracle")]
f3 = lambda x: f"{x:.3f}"


def write(name, s):
    open(os.path.join(T, name), "w", encoding="utf-8").write(s)


# ---------------------------------------------------------------- data statistics
ed_tr, ed_va, ed_te = dp.load_ed("train"), dp.load_ed("valid"), dp.load_ed("test")
es = dp.load_esconv()
words = lambda recs, f: np.mean([len(r[f].split()) for r in recs])
rows = []
for name, tr, va, te in (("ED", ed_tr, ed_va, ed_te), ("ESConv", es["train"], es["valid"], es["test"])):
    labs = {r["emotion"] for r in tr + va + te}
    rows.append(f"{name} & {len(tr):,} & {len(va):,} & {len(te):,} & {len(labs)} & {words(te,'query'):.1f} & {words(te,'target'):.1f} \\\\")
write("data.tex", r"""\begin{table}[t]
\centering
\caption{Corpora used in the experiments (ED: EmpatheticDialogues). Counts are conversations (one query and one gold reply each); the candidate pool of a split is the set of all its replies. Lengths are mean numbers of words in the test split.}
\label{tab:data}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrr}
\hline
Corpus & Train & Valid & Test & Emotions & Query len. & Reply len. \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- base retrievers
rows = []
for n in ALL:
    e, s = ED["retrievers"][n], ES["retrievers"][n]
    b, c = e["methods"]["base"], s["methods"]["base"]
    probe = lambda r: "--" if r["probe_acc"] is None else f3(r["probe_acc"])
    rows.append(f"{NICE[n]} & {f3(b['R@10'])} & {f3(b['MRR'])} & {f3(b['EmoAcc@10'])} & {f3(b['Harm@10'])} & {probe(e)} & "
                f"{f3(c['R@10'])} & {f3(c['MRR'])} & {f3(c['EmoAcc@10'])} & {probe(s)} \\\\")
    if n in SEP_AFTER:
        rows.append(r"\hline")
from collections import Counter
maj = lambda te: Counter(r["emotion"] for r in te).most_common(1)[0][1] / len(te)
write("base.tex", r"""\begin{table*}[t]
\centering
\caption{Off-the-shelf retrievers: topical relevance (R@10, MRR), emotional consistency (EmoAcc@10), harm (Harm@10, share of top-10 replies with opposite valence; ED only) and linear-probe emotion accuracy of the query embeddings. Majority-class accuracy of the probe is """ + f"{maj(ed_te):.3f}" + r""" (ED) and """ + f"{maj(es['test']):.3f}" + r""" (ESConv).}
\label{tab:base}
\begin{tabular}{lccccc|cccc}
\hline
 & \multicolumn{5}{c|}{EmpatheticDialogues} & \multicolumn{4}{c}{ESConv} \\
Retriever & R@10 & MRR & EmoAcc@10 & Harm@10 $\downarrow$ & Probe & R@10 & MRR & EmoAcc@10 & Probe \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- methods averaged over dense retrievers
def mean_over(res, m, k):
    return np.mean([res["retrievers"][n]["methods"][m][k] for n in DENSE])

rows = []
for m, label in METH:
    if m == "oracle":
        rows.append(r"\hline")
    rows.append(f"{label} & {f3(mean_over(ED,m,'R@1'))} & {f3(mean_over(ED,m,'MRR'))} & {f3(mean_over(ED,m,'EmoAcc@10'))} & {f3(mean_over(ED,m,'Harm@10'))} & "
                f"{f3(mean_over(ES,m,'R@1'))} & {f3(mean_over(ES,m,'MRR'))} & {f3(mean_over(ES,m,'EmoAcc@10'))} \\\\")
write("methods_mean.tex", r"""\begin{table}[t]
\centering
\caption{Emotion-aware strategies, averaged over the nine dense retrievers (test split). Hyper-parameters are tuned on validation to maximise EmoAcc@10 with at most 1\% loss of MRR; C-M has no parameter. Oracle uses the true emotion labels.}
\label{tab:methods}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc|ccc}
\hline
 & \multicolumn{4}{c|}{EmpatheticDialogues} & \multicolumn{3}{c}{ESConv} \\
Method & R@1 & MRR & Emo & Harm$\downarrow$ & R@1 & MRR & Emo \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- per-retriever detail on ED (C-A vs CGP, significance)
def star(p):
    return "$^{\\ast}$" if p < 0.05 else ""

rows = []
for n in ALL:
    r = ED["retrievers"][n]; M = r["methods"]; P = r["p_values"]
    rows.append(f"{NICE[n]} & {f3(M['base']['MRR'])} & {f3(M['base']['EmoAcc@10'])} & {f3(M['base']['Harm@10'])} & "
                f"{f3(M['comb']['MRR'])} & {f3(M['comb']['EmoAcc@10'])} & {f3(M['comb']['Harm@10'])} & "
                f"{f3(M['cgp']['MRR'])}{star(P['cgp_vs_comb_MRR'])} & {f3(M['cgp']['EmoAcc@10'])}{star(P['cgp_vs_comb_EmoAcc@10'])} & {f3(M['cgp']['Harm@10'])}{star(P['cgp_vs_comb_Harm@10'])} \\\\")
    if n in SEP_AFTER:
        rows.append(r"\hline")
write("ed_detail.tex", r"""\begin{table*}[t]
\centering
\caption{EmpatheticDialogues, per retriever: base retriever, additive combination (C-A) and CGP. $^{\ast}$: CGP differs from C-A at $p<0.05$ (two-sided paired randomisation test, 10,000 permutations); all emotion-aware rows differ from Base at $p<0.001$ on EmoAcc@10 and Harm@10.}
\label{tab:eddetail}
\begin{tabular}{lccc|ccc|ccc}
\hline
 & \multicolumn{3}{c|}{Base} & \multicolumn{3}{c|}{C-A} & \multicolumn{3}{c}{CGP} \\
Retriever & MRR & Emo & Harm$\downarrow$ & MRR & Emo & Harm$\downarrow$ & MRR & Emo & Harm$\downarrow$ \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- simulated accuracy
EDa = json.load(open(os.path.join(R, "ED_analysis.json"))); ESa = json.load(open(os.path.join(R, "ESConv_analysis.json")))
rows = []
for a, b in zip(EDa["simulated_accuracy"], ESa["simulated_accuracy"]):
    rows.append(f"{a['acc']:.1f} & {f3(a['MRR'])} & {f3(a['EmoAcc@10'])} & {f3(a['Harm@10'])} & {f3(b['MRR'])} & {f3(b['EmoAcc@10'])} \\\\")
write("simacc.tex", r"""\begin{table}[t]
\centering
\caption{C-A with simulated emotion recognizers of given accuracy on queries and candidates (GTE-small, mean of three seeds, $\lambda$ re-tuned on validation). Base GTE-small: MRR 0.244 / 0.307, EmoAcc@10 0.174 / 0.333 (ED / ESConv).}
\label{tab:simacc}
\begin{tabular}{lccc|cc}
\hline
 & \multicolumn{3}{c|}{EmpatheticDialogues} & \multicolumn{2}{c}{ESConv} \\
Accuracy & MRR & Emo & Harm$\downarrow$ & MRR & Emo \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- recognizer variants (ablation)
VARS = [("", "Logistic regression on MPNet embeddings"), ("_go", r"\quad + GoEmotions features"), ("_ft", "Partially fine-tuned MiniLM-L12")]
ABL = ["gte-small", "gte-base"]
if all(os.path.exists(os.path.join(R, f"ED{v}_results.json")) for v, _ in VARS):
    rows = []
    for v, label in VARS:
        r = json.load(open(os.path.join(R, f"ED{v}_results.json")))
        acc = r["emotion_clf"]
        cells = [f"{acc['query_acc']:.3f}", f"{acc['reply_acc']:.3f}"]
        for n in ABL:
            M = r["retrievers"][n]["methods"]
            cells += [f3(M["comb"]["MRR"]), f3(M["comb"]["EmoAcc@10"]), f3(M["comb"]["Harm@10"])]
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    write("recognizers.tex", r"""\begin{table*}[t]
\centering
\caption{Effect of the emotion recognizer on EmpatheticDialogues: recognition accuracy on test queries and replies, and C-A results for GTE-small and GTE-base (base retrievers: MRR """ + f3(ED["retrievers"]["gte-small"]["methods"]["base"]["MRR"]) + " / " + f3(ED["retrievers"]["gte-base"]["methods"]["base"]["MRR"]) + r""", EmoAcc@10 """ + f3(ED["retrievers"]["gte-small"]["methods"]["base"]["EmoAcc@10"]) + " / " + f3(ED["retrievers"]["gte-base"]["methods"]["base"]["EmoAcc@10"]) + r"""). All recognizers were trained on a CPU.}
\label{tab:recognizers}
\begin{tabular}{lcc|ccc|ccc}
\hline
 & \multicolumn{2}{c|}{Recognition accuracy} & \multicolumn{3}{c|}{GTE-small + C-A} & \multicolumn{3}{c}{GTE-base + C-A} \\
Recognizer & Query & Reply & MRR & Emo & Harm$\downarrow$ & MRR & Emo & Harm$\downarrow$ \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- adapters
AED, AES = os.path.join(R, "ED_adapter_results.json"), os.path.join(R, "ESConv_adapter_results.json")
if os.path.exists(AED) and os.path.exists(AES):
    A_ED, A_ES = json.load(open(AED)), json.load(open(AES))
    dense_ed = [n for n in DENSE if n in A_ED]; dense_es = [n for n in DENSE if n in A_ES]
    AM = [("Base", "Base"), ("Base+C-A", "Base + C-A"), ("TA", "TA"), ("TA+C-A", "TA + C-A"),
          ("EA", "EA"), ("EA+C-A", "EA + C-A"), ("EA+CGP", "EA + CGP")]
    am = lambda A, ns, m, k: np.mean([A[n]["methods"][m][k] for n in ns])
    rows = []
    for m, label in AM:
        rows.append(f"{label} & {f3(am(A_ED,dense_ed,m,'R@1'))} & {f3(am(A_ED,dense_ed,m,'MRR'))} & {f3(am(A_ED,dense_ed,m,'EmoAcc@10'))} & {f3(am(A_ED,dense_ed,m,'Harm@10'))} & "
                    f"{f3(am(A_ES,dense_es,m,'R@1'))} & {f3(am(A_ES,dense_es,m,'MRR'))} & {f3(am(A_ES,dense_es,m,'EmoAcc@10'))} \\\\")
        if m in ("Base+C-A", "TA+C-A"):
            rows.append(r"\hline")
    write("adapters_mean.tex", r"""\begin{table}[t]
\centering
\caption{Adapting the frozen retrievers, averaged over the nine dense encoders (test split). TA: topical adapter trained without emotion; EA: emotion-aware adapter. C-A and CGP are tuned on validation as in Table~\ref{tab:methods}, with the respective retriever as reference.}
\label{tab:adapters}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccc|ccc}
\hline
 & \multicolumn{4}{c|}{EmpatheticDialogues} & \multicolumn{3}{c}{ESConv} \\
Method & R@1 & MRR & Emo & Harm$\downarrow$ & R@1 & MRR & Emo \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
""")
    rows = []
    for n in dense_ed:
        M, P = A_ED[n]["methods"], A_ED[n]["p_values"]
        st = lambda k: star(P[f"EA+C-A_vs_Base+C-A_{k}"])
        rows.append(f"{NICE[n]} & {f3(M['Base+C-A']['MRR'])} & {f3(M['Base+C-A']['EmoAcc@10'])} & {f3(M['Base+C-A']['Harm@10'])} & "
                    f"{f3(M['TA+C-A']['MRR'])} & {f3(M['TA+C-A']['EmoAcc@10'])} & {f3(M['TA+C-A']['Harm@10'])} & "
                    f"{f3(M['EA+C-A']['MRR'])}{st('MRR')} & {f3(M['EA+C-A']['EmoAcc@10'])}{st('EmoAcc@10')} & {f3(M['EA+C-A']['Harm@10'])}{st('Harm@10')} \\\\")
    write("adapters_detail.tex", r"""\begin{table*}[t]
\centering
\caption{EmpatheticDialogues, per retriever: C-A on the base retriever, on the topical adapter (TA) and on the emotion-aware adapter (EA). $^{\ast}$: EA + C-A differs from Base + C-A at $p<0.05$ (two-sided paired randomisation test, 10,000 permutations).}
\label{tab:adaptersdetail}
\begin{tabular}{lccc|ccc|ccc}
\hline
 & \multicolumn{3}{c|}{Base + C-A} & \multicolumn{3}{c|}{TA + C-A} & \multicolumn{3}{c}{EA + C-A} \\
Retriever & MRR & Emo & Harm$\downarrow$ & MRR & Emo & Harm$\downarrow$ & MRR & Emo & Harm$\downarrow$ \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
""")

print("tables written to", T)
