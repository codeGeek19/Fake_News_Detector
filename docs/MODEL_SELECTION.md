# Model selection: what was checked, what was chosen, what is NOT verified

"Verified" below means I read the model card or paper abstract. I could not download or run any model while building this,
so no accuracy claim here has been reproduced by me.

## Hindi + Marathi classifier: DevBERT
| Candidate | Evidence | Decision |
|---|---|---|
| **DevBERT** `l3cube-pune/hindi-marathi-dev-bert` | Model card: one Devanagari BERT for **both Hindi and Marathi**, 238M params, CC-BY-4.0, built by fine-tuning MuRIL on Hindi+Marathi monolingual text. The paper (arXiv 2211.11418) reports DevBERT/HindBERT models improving over MuRIL, IndicBERT and XLM-R on Hindi and Marathi **text classification** (the authors' own benchmarks, not fake news). | **Chosen** as the encoder. One model serves both languages. |
| IndicBERT v2 (AI4Bharat) | You named it. I did not verify its Hub id in this session. | Works with the same scripts: `train.py --model <id>`; compare it against DevBERT on the same split. |
| MuRIL, XLM-R | Candidates from earlier research; the DevBERT paper claims DevBERT beats both on classification. | Not used. |

**Important:** DevBERT / IndicBERT are pretrained *language models*, not fake-news classifiers. The classification head must be fine-tuned
on labelled data. The dataset used is Zenodo 11408513 (CC-BY-4.0; Hindi and Marathi news articles labelled fake/real, article sources not described on the record page).
`ml/training/build_indic_dataset.py` + `train.py` do this. Nothing has been trained yet, so there are no Hindi/Marathi accuracy numbers.
Expect in-domain test scores to look high and real-world accuracy to be lower (random split, unknown source mix, no dates).

## English
| Role | Model | Notes |
|---|---|---|
| Fake/real classifier | `jy46604790/Fake-News-Bert-Detect` (RoBERTa, Apache-2.0) | Only English detector whose label meaning I could verify (LABEL_0 = fake). Training data not documented. **I found no independent benchmark proving it is the most accurate**, so I did not claim that. |
| Claim-vs-evidence judge | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MIT, 435M) | Trained on MNLI, **FEVER-NLI** (fact verification), ANLI, LingNLI, WANLI; self-reported ANLI-all accuracy 0.702, MNLI-matched 0.912. The strongest verified fit for judging a claim against retrieved evidence. |
| Rejected | `newsmediabias/FakeNews-Classifier-NELA-GT` | No model card, so its labels cannot be verified. |
| Rejected | `hamzab/roberta-fake-news-classification` | Its expected input format was not verified. |

To pick the most accurate English classifier **on your own data**, benchmark candidates with
`ml/evaluation/compare_pretrained.py` on LIAR (short claims) and ISOT (articles). That is the only honest way to rank them.

## Shared components
| Role | Model |
|---|---|
| Hindi/Marathi claim-vs-evidence judge | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT; Hindi is in its training data, Marathi is not confirmed) |
| Semantic retrieval embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M) |
| Translate-then-classify fallback | `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0, research model; hin_Deva/mar_Deva -> eng_Latn); lighter backups `Helsinki-NLP/opus-mt-mr-en` (verified to exist, small) and `Helsinki-NLP/opus-mt-hi-en` (not verified) |

## Why the classifier never decides the result
Every classifier here learned writing patterns from old data. The final verdict comes from retrieved fact-checks and NLI-judged news, and is
Uncertain without evidence. The classifier estimate is shown separately, abstains on short text and below a probability threshold.
