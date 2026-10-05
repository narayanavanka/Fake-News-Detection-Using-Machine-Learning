import argparse
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

log = logging.getLogger("fakenews.train")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_NAME = "WELFake_Dataset.csv"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "svm_model.joblib"
META_PATH = MODELS_DIR / "metadata.json"

# In the WELFake CSV used in this project, label 1 maps to fake articles and the
# other class maps to real articles. Keep this fixed unless you deliberately swap
# to a differently labelled dataset and retrain everything.
FAKE_LABEL = 1

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_JUNK = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")

FAKE_PROBE = (
    "BREAKING secret documents prove elites are hiding a miracle cure from the public "
    "while the media stays silent and anonymous insiders expose the cover up."
)
REAL_PROBE = (
    "Reuters reported that the city council approved its annual budget after a public "
    "debate and officials said the full document would be published next week."
)


def clean_text(text) -> str:
    """Lowercase; strip HTML, URLs, punctuation and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = _HTML.sub(" ", text.lower())
    text = _URL.sub(" ", text)
    return _SPACE.sub(" ", _JUNK.sub(" ", text)).strip()


def find_dataset():
    for c in (os.getenv("WELFAKE_CSV"), DATA_DIR / DATASET_NAME, BASE_DIR / DATASET_NAME):
        if c and Path(c).is_file():
            return Path(c)
    if os.getenv("WELFAKE_AUTO_DOWNLOAD") == "1":
        try:
            import kagglehub

            folder = Path(kagglehub.dataset_download("nitaisatapathy/welfake-dataset"))
            src = next(folder.rglob("*.csv"))
            DATA_DIR.mkdir(exist_ok=True)
            shutil.copy(src, DATA_DIR / DATASET_NAME)
            return DATA_DIR / DATASET_NAME
        except Exception as exc:
            log.error("Automatic dataset download failed: %s", exc)
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["content"] = (df["title"] + " " + df["text"]).map(clean_text)
    df = df[df["content"].str.split().str.len() >= 5]
    return df.drop_duplicates(subset="content")[["content", "label"]]


def load_modern_real(csv_path=None, oversample: int = 1) -> pd.DataFrame:
    """Fresh real articles scraped from the project's trusted RSS feeds.

    WELFake's real class is mostly 2016-era Reuters/AP wire copy, so modern
    articles (sports recaps, official statements, non-US outlets) get misread
    as fake. Mixing these in widens what the model accepts as 'real'.

    oversample: repeat the modern articles this many times so their vocabulary
    carries real weight against WELFake's 60k+ wire articles (WELFake's fake
    class covers many political/defense topics that the modern real class
    shares, so a 1:60 mix leaves the bias intact).
    """
    path = Path(csv_path) if csv_path else DATA_DIR / "real_news_modern.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["content", "label"])
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = (df["title"] + " " + df["text"]).map(clean_text)
    df = df[df["content"].str.split().str.len() >= 40]
    df = df.drop_duplicates(subset="content")[["content"]]
    df["label"] = 1 - FAKE_LABEL  # the real class
    if oversample > 1:
        df = pd.concat([df] * oversample, ignore_index=True)
    log.info("Merged %d modern real articles (x%d oversample) from %s",
             len(df), oversample, path.name)
    return df


def models_exist() -> bool:
    return all(p.is_file() for p in (VECTORIZER_PATH, MODEL_PATH, META_PATH))


def fake_index(model, fake_label: int = FAKE_LABEL) -> int:
    classes = list(model.classes_)
    if fake_label not in classes:
        raise ValueError(f"fake_label={fake_label} not present in model classes {classes}")
    return classes.index(fake_label)


def label_sanity_check(model, vectorizer, fake_label: int = FAKE_LABEL) -> dict:
    classes = [int(c) for c in model.classes_]
    fake_idx = fake_index(model, fake_label)
    real_labels = [c for c in classes if c != fake_label]
    if len(real_labels) != 1:
        raise ValueError(f"Expected exactly one non-fake class, got {classes}")
    real_label = real_labels[0]
    real_idx = classes.index(real_label)

    fake_text = clean_text(FAKE_PROBE)
    real_text = clean_text(REAL_PROBE)
    fake_proba = model.predict_proba(vectorizer.transform([fake_text]))[0]
    real_proba = model.predict_proba(vectorizer.transform([real_text]))[0]
    fake_pred = int(model.predict(vectorizer.transform([fake_text]))[0])
    real_pred = int(model.predict(vectorizer.transform([real_text]))[0])

    return {
        "passed": fake_pred == fake_label and real_pred == real_label,
        "fake_probe_prediction": fake_pred,
        "real_probe_prediction": real_pred,
        "fake_probe_fake_probability": round(float(fake_proba[fake_idx]), 4),
        "real_probe_real_probability": round(float(real_proba[real_idx]), 4),
        "classes": classes,
    }


def train(csv_path=None) -> dict:
    csv_path = Path(csv_path) if csv_path else find_dataset()
    if csv_path is None or not csv_path.is_file():
        raise FileNotFoundError(f"{DATASET_NAME} not found. Put it in {DATA_DIR}.")
    start = time.time()
    df = load_dataset(csv_path)
    log.info("Loaded %d unique articles from %s", len(df), csv_path)
    modern = load_modern_real(oversample=15)
    if not modern.empty:
        df = pd.concat([df, modern], ignore_index=True)
        log.info("Combined training set: %d articles (incl. oversampled modern real)", len(df))

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["content"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000, stop_words="english",
        sublinear_tf=True, min_df=3, max_df=0.9,
    )
    Xtr = vectorizer.fit_transform(X_tr)
    Xte = vectorizer.transform(X_te)

    model = CalibratedClassifierCV(
        estimator=LinearSVC(C=0.5, class_weight="balanced"), method="sigmoid", cv=5
    )
    model.fit(Xtr, y_tr)
    sanity = label_sanity_check(model, vectorizer, FAKE_LABEL)
    if not sanity["passed"]:
        raise RuntimeError(f"Label sanity check failed: {sanity}")

    pred = model.predict(Xte)
    names = ["Fake" if c == FAKE_LABEL else "Real" for c in model.classes_]
    metrics = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "f1_weighted": round(f1_score(y_te, pred, average="weighted"), 4),
        "train_size": len(y_tr), "test_size": len(y_te),
    }
    print(f"\nAccuracy: {metrics['accuracy']:.4f}   F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(classification_report(y_te, pred, target_names=names, digits=4))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    metrics.update(fake_label=FAKE_LABEL, trained_seconds=round(time.time() - start), sanity_check=sanity)
    META_PATH.write_text(json.dumps(metrics, indent=2))  # written last = "training complete"
    return metrics


def auto_train_if_missing() -> bool:
    """Train when artifacts are missing. Returns False if no dataset is available."""
    if models_exist():
        return True
    log.warning("Model artifacts missing - starting automatic training.")
    csv = find_dataset()
    if csv is None:
        log.error("Cannot auto-train: %s not found in %s.", DATASET_NAME, DATA_DIR)
        return False
    train(csv)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to WELFake_Dataset.csv")
    train(ap.parse_args().data)
