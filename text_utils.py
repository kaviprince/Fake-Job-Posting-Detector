"""
Shared text-cleaning utilities for the Fake Job Posting Detector.
Used by preprocessing.py, model_train.py, explain.py, and app.py so that
the SAME cleaning logic is applied at training time and at prediction time.
"""
import re
import nltk
from nltk.corpus import stopwords

_STOPWORDS = None


def ensure_nltk_data():
    """Download the 'stopwords' corpus only if it isn't already present."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_data()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/numbers, remove stopwords and short tokens."""
    stop_words = get_stopwords()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)            # keep letters only
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)
