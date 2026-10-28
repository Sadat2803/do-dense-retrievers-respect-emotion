# Do Dense Retrievers Respect Emotion?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23730675.svg)](https://doi.org/10.5281/zenodo.23730675)

Code and results for the paper **"Do Dense Retrievers Respect Emotion? Measuring and Improving Emotional
Consistency in Retrieval"** by Islam Sadat and Samir Kechid (LRIA, USTHB, Algiers).

The study measures whether retrievers return replies that fit the emotion of a message, compares
training-free emotion-aware strategies, and proposes an emotion-aware adapter trained on frozen embeddings.
Experiments use EmpatheticDialogues and ESConv, twelve retrievers (BM25, TF-IDF, nine dense encoders and a
cross-encoder re-ranker), and run entirely on a CPU.

## Contents

| Path | Content |
|---|---|
| `code/download_data.py` | downloads both corpora from their official sources and checks them against the copies used in the paper |
| `code/data_prep.py` | loaders and the valence grouping of the 32 EmpatheticDialogues emotions |
| `code/run_experiments.py` | benchmark, emotion recognizer, training-free strategies (Filter, C-A, C-M, S-S, S-E, CGP, oracle), significance tests |
| `code/adapter.py` | topical (TA) and emotion-aware (EA) adapters |
| `code/analysis.py` | robustness, per-emotion results, trade-off curves, simulated recognizers, figures |
| `code/emo_features.py`, `code/finetune_emotion.py` | the two alternative recognizers (GoEmotions features, partially fine-tuned MiniLM) |
| `code/make_tables.py` | LaTeX tables of the paper from the result files |
| `results/*_results.json` | all metrics, selected hyper-parameters and p-values reported in the paper |
| `results/*_per_query.json` | per-query R@1, R@10, MRR, EmoAcc@10 and Harm@10 for every method, for further tests |
| `results/*_analysis.json`, `results/finetune_log.json` | analyses and recognizer training logs |

Result file names: `ED` = EmpatheticDialogues; `_adapter` = adapters; `_go` / `_ft` = alternative recognizers.

## Reproducing

Python 3.12:

```
pip install -r requirements.txt
cd code
python download_data.py
python run_experiments.py all          # benchmark and training-free strategies -> results/*_results.json
python analysis.py all                 # analyses and figures -> figures/
python adapter.py all                  # adapters -> results/*_adapter_results.json
python emo_features.py all             # optional: GoEmotions recognizer, then EMO_VARIANT=go python run_experiments.py all
python finetune_emotion.py ED q 2      # optional: fine-tuned recognizer (also ED d, ESConv q, ESConv d), then EMO_VARIANT=ft
python make_tables.py                  # LaTeX tables -> tables/
```

Set `RETRIEVERS=name1,name2` to run a subset of retrievers. Sentence embeddings are cached in `cache/`.
All hyper-parameters are selected on the validation split with one rule: maximise EmoAcc@10 subject to
MRR >= 0.99 x MRR of the reference retriever.

## Data

The corpora are not redistributed here; `download_data.py` fetches them from their authors:
EmpatheticDialogues (Rashkin et al., ACL 2019, Facebook Research, CC BY-NC 4.0) and ESConv
(Liu et al., ACL-IJCNLP 2021, thu-coai). Please respect their licences and cite them.
ESConv is split 70/10/20 by conversation with seed 13.

## Licence

Code: MIT (see `LICENSE`). Result files: CC BY 4.0.

## Citation

Archived version: DOI [10.5281/zenodo.23730675](https://doi.org/10.5281/zenodo.23730675). See `CITATION.cff`. The paper is under review; the reference will be updated on publication.
