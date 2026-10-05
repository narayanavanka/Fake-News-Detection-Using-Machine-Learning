# Fake News Detector (WELFake + calibrated LinearSVC)

```
fake-news-detector/
├── backend/
│   ├── app.py          FastAPI app (lifespan auto-train, /predict/manual, /predict/url, /health)
│   ├── train_model.py  cleaning, TF-IDF + CalibratedClassifierCV(LinearSVC), auto_train_if_missing()
│   ├── scraper.py      newspaper3k with BeautifulSoup fallback
│   ├── verify.py       live checks: source, feeds, news coverage, fact-checks, combined verdict
│   ├── anthropic_verify.py  optional paid web verification (Anthropic web search)
│   ├── gemini_verify.py     free web verification (Gemini + Google Search grounding)
│   └── feeds.py        trusted-outlet RSS headlines, reloaded every 5 minutes
├── frontend/           index.html, style.css, app.js (served by FastAPI at /)
├── .env.example        copy to .env and add your free API keys
├── data/               put WELFake_Dataset.csv here
├── models/             trained artifacts are written here
└── requirements.txt
```

## Setup (Python 3.9-3.12)
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Get the dataset from https://www.kaggle.com/datasets/nitaisatapathy/welfake-dataset and save it as
   `data/WELFake_Dataset.csv`.
   Alternatively, set up Kaggle credentials and run with `WELFAKE_AUTO_DOWNLOAD=1` to download it automatically.

## Train
- **Manual:** `python -m backend.train_model` (prints accuracy, F1 and a classification report; saves to `models/`).
- **Automatic:** just start the server. If `models/` has no artifacts, training starts in a background thread.
  The API stays up and reports `training` on `/health`; the UI shows a banner until it's ready.
  If the dataset is missing the server still starts and explains what to do (no crash).

## Run
```
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. API docs are at http://localhost:8000/docs.

```
curl -X POST localhost:8000/predict/manual -H "Content-Type: application/json" \
  -d '{"title": "...", "text": "..."}'
curl -X POST localhost:8000/predict/url -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

## Notes
- **Label direction:** `FAKE_LABEL = 1` in `backend/train_model.py` for the WELFake CSV used here.
- Training and the API share the same `clean_text()` function, so preprocessing cannot drift.
- The vectorizer is saved to `models/tfidf_vectorizer.joblib` and loaded back at inference, so the same TF-IDF preprocessing is used for both training and prediction.
- Training now writes a small label sanity check into `models/metadata.json`, and the API re-runs that check at startup before serving predictions.
- Duplicate articles are dropped before splitting, so test accuracy isn't inflated by leakage.
- Tuning: adjust `C` in `LinearSVC` and `max_features`/`min_df` in `TfidfVectorizer` (`train_model.py`).
- Sites that are paywalled or render text with JavaScript can't be scraped; paste the text instead.
- The model reflects the writing style of its training data. Treat results as a signal, not a fact-check.

## Live verification
The model only judges *writing style*. The app now keeps that output separate from the evidence verdict:
the writing-style card always shows the model's own label and probabilities, while the top verdict comes
from live verification (`backend/verify.py`).

| Check | What it does | Key needed? |
|---|---|---|
| Web verification | Checks the article's main claims against current reporting via LLM web search (Gemini free tier by default; Anthropic if you add its key) and returns supporting or contradicting links | Gemini: free / Anthropic: paid |
| Source | Is the site a well-known news organisation? (short list in `verify.py`) | No |
| Trusted feeds | Compares the headline with the latest posts in live RSS feeds of about 15 outlets (BBC, Guardian, NPR, Al Jazeera, The Hindu and more), reloaded every 5 minutes in the background | No |
| Fact-checks | Searches published fact-checks with the Google Fact Check Tools API | Yes, free |
| Live coverage | Looks for other outlets reporting the same headline in the last 7 days: GDELT (updates about every 15 min) and, if you add a key, GNews as a backup | No (GNews optional) |

**Scoring:** web verification carries the strongest weight because it checks live evidence directly. The writing-style model is only a weak tie-breaker. With a `GEMINI_API_KEY` (free) web verification runs on Gemini with Google Search grounding; if `ANTHROPIC_API_KEY` is also set, Anthropic takes priority and Gemini stays as fallback.

### Get your free keys (about 5 minutes)
1. Copy `.env.example` to `.env` (Windows: `copy .env.example .env`, Mac/Linux: `cp .env.example .env`).
2. **Gemini key (free, recommended):** open https://aistudio.google.com, sign in with a Google account and
   click *Get API key*. Paste it after `GEMINI_API_KEY=` in `.env`. This enables web verification on the
   free tier using Google Search grounding — no card needed.
3. **Anthropic key (optional, paid):** create an API key at https://console.anthropic.com and paste it after
   `ANTHROPIC_API_KEY=`. If both keys are set, Anthropic is used and Gemini becomes the fallback. Without it,
   only Gemini is used.
4. **Fact-check key (recommended):** open https://console.cloud.google.com, create or pick a project, go to
   *APIs & Services > Library*, search "Fact Check Tools API" and click **Enable**. Then *Credentials > Create
   credentials > API key*. Paste it after `GOOGLE_FACTCHECK_API_KEY=` in `.env`.
5. **GNews key (optional):** sign up at https://gnews.io (no card) and paste the key after `GNEWS_API_KEY=`.
6. Run `pip install -r requirements.txt` again and restart the server.

The app works without any keys; the cards for missing services just show "Not set up".

## Recent evaluation set
To build a recent CSV that better reflects current articles and debunked claims, run:

```
python -m backend.build_recent_dataset
```

By default this writes `data/recent_test_set.csv` with:
- recent real articles scraped from trusted newsroom feeds
- recent false or misleading claims pulled from the Google Fact Check API

You can change the size or output path:

```
python -m backend.build_recent_dataset --real-limit 30 --fake-limit 30 --output data/recent_test_set.csv
```

### Good to know
- **Trusted feeds** is the freshest check: a story is matched minutes after an outlet publishes it. Each feed only holds its
  latest 20-60 stories (hours to about a day), so older stories won't match, and "no match" adds no penalty.
  Matching is keyword-based (at least 60% of the headline's main words). The article's own outlet is skipped.
- The feed list is in `backend/feeds.py`. A dead or moved feed is skipped automatically; the terminal shows
  `RSS refresh: X/Y feeds loaded` every 5 minutes. Add or remove feeds there.
- If a trusted feed has the story but the news search doesn't yet, the missing coverage is not held against it.
- Free news APIs are not truly real-time: GDELT is about 15 minutes behind, GNews's free plan is 12 hours behind.
- GDELT allows about one request per 5 seconds, so a check can take several seconds. Lookups are cached for 10 minutes.
- Fact-check matches are keyword-based and can be loose. Open the links to confirm they are about the same claim.
- "No coverage found" can simply mean a story is brand new or local. The trusted-outlets list is short and does not
  guarantee accuracy. Treat the verdict as a starting point and read the evidence.
