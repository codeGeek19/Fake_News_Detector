# Multilingual Fake News Verification (English · हिंदी · मराठी)

A web app that helps you **check a news claim** instead of just labelling it. Paste a headline, claim or article, pick a language, and you get **two separate answers**:

| | What it is | Where it comes from |
|---|---|---|
| **Result** 🟢 Likely Real · 🔴 Likely Fake · 🟡 Uncertain | The evidence-based assessment | Fact-checks and current news found online, judged against your claim |
| **Model's take** Likely Real / Likely Fake + confidence | What a transformer classifier thinks from the writing alone | A pretrained or fine-tuned classifier |

The **Result can be Uncertain while the Model's take is still shown**. The model always gives its take when a classifier can run, and adds caveats (*short text*, *low confidence*, *read a machine translation*) when it may be unreliable. The classifier **never decides the final Result**: with no evidence, the Result stays Uncertain.

It is an **AI-assisted check, not a truth detector.** Always open the source links and verify for yourself.

---

## What you see after clicking "Analyze News"
- **Result** (evidence-based) and **Model's take** side by side, with confidence and caveat chips
- **Claims**: the main factual claims found in your text, each with its own verdict
- **Why?**: a plain-language explanation in your language, plus a "How this works" box
- **Important words**: Integrated Gradients highlights (🔴 pushes toward fake, 🟢 toward real). *These show what influenced the model, not what is true.*
- **Token analysis**: interactive tokens, ids, positions
- **Evidence and Related sources**: cards with publisher, date, headline, snippet, support/contradict label and a link to the original
- **Technical details**: model names and versions, training-data cutoff, tokens, timings, when the evidence was checked

---

## Status: what works and what is unverified
| Part | State |
|---|---|
| Backend, frontend, retrieval, explanations, SQLite history | **77 backend + 12 frontend tests pass**; the frontend builds |
| ML scripts (dataset builders, training, calibration, benchmarking) | 19 tests pass (1 needs PyTorch and is skipped without it) |
| Downloading the real models, live news and fact-check APIs, Docker, Integrated Gradients on real weights | **Never run.** The build environment had no network, GPU or Docker. Expect small first-run fixes. |
| **Hindi / Marathi native classifier (DevBERT)** | **Code is complete, but the weights don't exist until you train once** (see below). Until then Hindi/Marathi use the translation fallback. |
| English accuracy | **Cannot be certified.** No independent benchmark exists for the pretrained detector. A script lets you measure it yourself. |

---

## Models (sources and rejected candidates in `docs/MODEL_SELECTION.md`)
| Role | Model |
|---|---|
| English classifier | `jy46604790/Fake-News-Bert-Detect` (RoBERTa, Apache-2.0; LABEL_0 = fake) |
| English claim-vs-evidence judge | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MIT; trained on FEVER, MNLI, ANLI) |
| Hindi + Marathi classifier | **DevBERT** `l3cube-pune/hindi-marathi-dev-bert` (CC-BY-4.0), fine-tuned by you on a labelled Zenodo dataset (record 11408513) |
| Hindi / Marathi judge | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT) |
| Retrieval embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) |
| Translation fallback (Hindi/Marathi → English) | `facebook/nllb-200-distilled-600M` (CC-BY-NC, research model); lighter backups `Helsinki-NLP/opus-mt-mr-en` / `opus-mt-hi-en` |

DevBERT and IndicBERT are pretrained *language models*, not fake-news classifiers. They need the one-time fine-tuning step below.

---

## Run it (Windows)
You need **Python 3.10+** and **Node.js 18+** (https://nodejs.org).

```bat
copy .env.example .env
```
Optional but strongly recommended: put a free **Google Fact Check Tools API key** in `.env` as `GOOGLE_FACTCHECK_API_KEY=...`. Without it most results will be Uncertain.

**Terminal 1: backend**
```bat
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```
**Terminal 2: frontend**
```bat
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173**. The first start downloads several GB of models (once), so the first request is slow. Docker alternative: `docker compose up --build` (UI on http://localhost:3000).

Check **http://localhost:8000/api/health** at any time: it shows which models and providers are active and, under `load_errors`, **why anything failed to load**.

---

## "Model's take: Not available": what to do
This appears for Hindi/Marathi when no native model is installed **and** the translation fallback could not run. The line tells you why:

| Message | Fix |
|---|---|
| *no model is set up for this language yet* | Train DevBERT (next section) or leave `TRANSLATE_MODEL` set |
| *the translation model could not be loaded* | Open `/api/health` and read `load_errors` (usually out of memory or a failed download). Free RAM, or set `TRANSLATE_MODEL=` (empty) so the small `opus-mt` backups are used; failed loads retry automatically after 2 minutes |
| *the English model needed for translation could not be loaded* | Check `MODEL_EN_PATH` and your internet connection |
| *the text could not be translated* | Try a shorter or cleaner input |

Heavy defaults need RAM: the English judge (435M parameters), the multilingual judge, the English classifier, the embedding model and the translator together use many GB. If your machine struggles, set `NLI_MODEL_EN=` (empty) to use the smaller multilingual judge for English, and `PRELOAD_MODELS=false`.

---

## Train the Hindi + Marathi classifier (once, plain Python)
```bat
python -m venv ml\.venv
ml\.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ml\requirements.txt

python ml\training\build_indic_dataset.py --inspect
python ml\training\build_indic_dataset.py
python ml\training\train.py --model l3cube-pune/hindi-marathi-dev-bert --lang hi_mr ^
   --train_csv data\dev_train.csv --val_csv data\dev_val.csv --test_csv data\dev_test.csv --out_dir out\dev --max_len 256
python ml\evaluation\calibrate.py --model_dir out\dev\final --csv data\dev_val.csv
xcopy /E /I out\dev\final models\dev
```
Restart the backend; `/api/health` should now show `models/dev` for Hindi and Marathi.
- `--inspect` shows what the dataset looks like (label counts and samples). I could not open the files while writing the script, so the layout is auto-detected. **If labels are numeric (0/1) it refuses to guess** which means fake: re-run with `--fake_values 1` (or whichever is right).
- **Speed:** a base-size model trains slowly on CPU (hours). Use a GPU machine if you can, or add `--max_train_samples 4000 --max_len 128` for a first run.
- **IndicBERT instead:** pass its Hugging Face id to `--model` (I did not verify the id) and compare on the same files.
- **Read the scores sceptically.** The split is random and the dataset's sources are not described, so test scores will look better than real-world accuracy.

---

## Check English accuracy yourself
```bat
python ml\training\build_english_eval.py
python ml\evaluation\compare_pretrained.py --model jy46604790/Fake-News-Bert-Detect:LABEL_0 --csv data\liar_test.csv --csv data\isot_test.csv --limit 1000
```
Add more `--model ID:FAKE_LABEL` arguments to rank other classifiers. **Judge by LIAR (short claims)**; ISOT is easy and says little about real news. If another model clearly wins, set `MODEL_EN_PATH` and `MODEL_EN_FAKE_LABEL` in `.env`.
To evaluate the whole system (classifier + evidence + rules) on claims you have verdicts for, use `ml/evaluation/eval_pipeline.py`.

---

## How it works
1. **Language**: detected from script and common words (Hindi and Marathi share Devanagari; ties follow your toggle). Hinglish is detected, not transliterated.
2. **Claims**: extracted by rules (attributions removed, opinions flagged, questions skipped).
3. **Model's take**: native classifier per language; if none, the text is machine-translated and the English classifier reads the translation (shown in the UI). Token table and Integrated Gradients highlights.
4. **Retrieve** per claim: Google Fact Check API, GDELT, Google News RSS, and NewsAPI (English only, optional).
5. **Re-rank** by word overlap, embedding similarity, recency and publisher type (a weak signal, never "always true"); remove syndicated duplicates.
6. **Judge** each article against the claim with an NLI model (supports / contradicts / related).
7. **Decide** with transparent rules: a published fact-check verdict decides; otherwise at least 2 independent publishers with NLI confidence ≥ 0.8; else Uncertain. Flags: `unverified`, `developing_story`, `satire_flagged`, `opinion`.
8. **Explain** with templates built only from model output and retrieved evidence, in English, Hindi or Marathi. An optional LLM paraphrase (`ENABLE_LLM_SUMMARY`) is grounded and discarded if it adds numbers or links not in its input; it can never change the verdict.
9. **Log** each analysis to SQLite (timestamp, model versions, sources with dates). Your text is **not** stored unless `STORE_TEXT=true`.

---

## Key settings (`.env`; full list in `.env.example`)
| Setting | Meaning |
|---|---|
| `MODEL_EN_PATH`, `MODEL_EN_FAKE_LABEL` | English classifier (Hub id or folder) and which output label means fake |
| `MODEL_HI_PATH`, `MODEL_MR_PATH` | Native Hindi/Marathi model folder (default `../models/dev`); a missing folder is fine, the fallback is used |
| `NLI_MODEL_EN`, `NLI_MODEL` | Claim-vs-evidence judges (English / Hindi+Marathi) |
| `TRANSLATE_MODEL`, `TRANSLATE_FALLBACK_HI/MR` | Translation fallback chain; empty disables |
| `MIN_CLASSIFIER_WORDS`, `CLASSIFIER_THRESHOLD` | When the model's take is flagged *short text* / *low confidence* |
| `GOOGLE_FACTCHECK_API_KEY`, `NEWSAPI_KEY` | Evidence providers (keys stay on the server) |
| `ENABLE_LLM_SUMMARY`, `ANTHROPIC_API_KEY` | Optional grounded summary |

## API
`POST /api/analyze` `{"text": "...", "language": "en|hi|mr|auto"}` · `GET /api/health` · `/api/models` · `/api/sources` · `/api/history` · interactive docs at `/docs`.
Errors: 422 invalid input · 413 too long · 429 rate limit · 503 English model cannot load · 504 timeout. A missing Hindi/Marathi model is **not** an error; the response says why through `classifier_status`.

## Project layout
```
backend/   FastAPI app: preprocessing, services (classifier, translation, verdict, explanation), retrieval, explainability, tests
frontend/  React + TypeScript + Tailwind UI (English / Hindi / Marathi) and tests
ml/        dataset builders, training, calibration, benchmarking, pipeline evaluation, tests
docs/      MODEL_SELECTION.md
models/    put your trained model folders here (models/dev)
```
Run the tests: `cd backend && pytest` · `cd frontend && npm test` · `pytest ml/tests` (from the project root).

## Limitations (keep these in your report)
- No classifier here can know about events after its training data; that is what the evidence layer is for.
- The English detector's training data is undocumented. The Hindi/Marathi dataset's sources are undescribed and its split is random, so there is no temporal evaluation.
- Headline-only NLI is weak, and weaker in Hindi; Marathi support in the multilingual judge is unconfirmed.
- Claim extraction is rule-based. Relevance is word overlap plus embeddings, not a trained ranker. "Original reporting" is not detected.
- Translation errors affect the Hindi/Marathi fallback, and the NLLB model is non-commercial and research-licensed.
- Satire detection relies on fact-checker labels; GDELT dates mean "first seen".
- Not implemented: dataset versioning, scheduled retraining, PostgreSQL/Redis (SQLite and an in-memory cache are used).
