"""FastAPI backend: serves predictions and the frontend."""
import json
import logging
import mimetypes
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.feeds import start_background_refresh
from backend.scraper import ScraperError, scrape_article
from backend.verify import verify
from backend.train_model import (FAKE_LABEL, META_PATH, MODEL_PATH, VECTORIZER_PATH,
                                 auto_train_if_missing, clean_text, fake_index,
                                 label_sanity_check, models_exist)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fakenews.api")

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # API keys live in .env (see .env.example)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

mimetypes.add_type("text/css", ".css")  # some Windows registries map these wrongly
mimetypes.add_type("application/javascript", ".js")


class NoCacheStaticFiles(StaticFiles):
    """Always revalidate, so browsers never show stale CSS/JS after an update."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class ModelState:
    status = "loading"  # loading | training | ready | unavailable
    message = "Loading model..."
    vectorizer = model = None
    fake_idx = 0
    metrics: dict = {}


state = ModelState()


def _load_artifacts():
    state.vectorizer = joblib.load(VECTORIZER_PATH)
    state.model = joblib.load(MODEL_PATH)
    state.metrics = json.loads(META_PATH.read_text())
    meta_fake_label = state.metrics.get("fake_label")
    if meta_fake_label is not None and meta_fake_label != FAKE_LABEL:
        log.warning("Metadata fake_label=%s disagrees with code fake_label=%s", meta_fake_label, FAKE_LABEL)
        state.metrics["fake_label_warning"] = (
            f"metadata fake_label {meta_fake_label} disagrees with code fake_label {FAKE_LABEL}"
        )
    state.fake_idx = fake_index(state.model, FAKE_LABEL)
    sanity = label_sanity_check(state.model, state.vectorizer, FAKE_LABEL)
    state.metrics["sanity_check"] = sanity
    if not sanity["passed"]:
        raise ValueError(f"Model label sanity check failed: {sanity}")
    state.status, state.message = "ready", "Model ready."
    log.info("Model loaded (accuracy %s)", state.metrics.get("accuracy"))


def _bootstrap():
    """Load artifacts, auto-training first when they are missing. Never raises."""
    try:
        if not models_exist():
            state.status, state.message = "training", "Training the model on WELFake. This takes a few minutes."
            if not auto_train_if_missing():
                state.status = "unavailable"
                state.message = "Dataset not found. Put WELFake_Dataset.csv in the data/ folder and restart."
                return
        _load_artifacts()
    except Exception as exc:
        log.exception("Model setup failed")
        state.status, state.message = "unavailable", f"Model setup failed: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Serving frontend from %s", FRONTEND_DIR)
    start_background_refresh()
    threading.Thread(target=_bootstrap, daemon=True).start()  # keeps the API up during training
    yield


app = FastAPI(title="Fake News Detector", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ManualRequest(BaseModel):
    title: str = Field("", max_length=500)
    text: str = Field("", max_length=100_000)


class URLRequest(BaseModel):
    url: str = Field(..., min_length=4, max_length=2048)


class Probabilities(BaseModel):
    real: float
    fake: float


class PredictionResponse(BaseModel):
    label: Literal["Real", "Fake"]
    confidence: float
    probabilities: Probabilities
    verification: Optional[Dict[str, Any]] = None


class ScrapedArticle(BaseModel):
    url: str
    domain: Optional[str]
    title: str
    text_snippet: str
    word_count: int
    authors: List[str]
    publish_date: Optional[str]
    method: str


class URLPredictionResponse(PredictionResponse):
    article: ScrapedArticle


@app.exception_handler(ScraperError)
async def scraper_error_handler(_: Request, exc: ScraperError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


def classify(title: str, text: str, min_words: int = 10) -> dict:
    if state.status != "ready":
        raise HTTPException(503, state.message)
    cleaned = clean_text(f"{title} {text}")
    if len(cleaned.split()) < min_words:
        raise HTTPException(422, f"Add more text: at least {min_words} words are needed for a reliable result.")
    proba = state.model.predict_proba(state.vectorizer.transform([cleaned]))[0]
    fake = float(proba[state.fake_idx])
    real = float(proba[1 - state.fake_idx])
    return {
        "label": "Fake" if fake >= real else "Real",
        "confidence": round(max(fake, real) * 100, 2),
        "probabilities": {"real": round(real * 100, 2), "fake": round(fake * 100, 2)},
    }


@app.get("/health")
def health():
    return {"status": state.status, "message": state.message, "metrics": state.metrics}


@app.post("/predict/manual", response_model=PredictionResponse)
def predict_manual(req: ManualRequest):
    result = classify(req.title, req.text)
    result["verification"] = verify(req.title, req.text, None, result)
    return result


@app.post("/predict/url", response_model=URLPredictionResponse)
def predict_url(req: URLRequest):
    if state.status != "ready":  # fail fast before hitting the network
        raise HTTPException(503, state.message)
    art = scrape_article(req.url)
    result = classify(art["title"], art["text"], min_words=50)
    result["verification"] = verify(art["title"], art["text"], art["url"], result)
    snippet = art["text"][:700].rsplit(" ", 1)[0] + ("..." if len(art["text"]) > 700 else "")
    result["article"] = {**art, "text_snippet": snippet, "word_count": len(art["text"].split())}
    return result


if FRONTEND_DIR.is_dir():  # mounted last so API routes win
    app.mount("/", NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
