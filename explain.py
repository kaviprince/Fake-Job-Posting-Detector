"""
Step 4: Given a trained model + vectorizer and a raw piece of job-posting
text, produce a prediction plus a simple explanation (top contributing
keywords). Used directly by app.py.
"""
import numpy as np
from text_utils import clean_text


def explain_prediction(raw_text: str, model, vectorizer, top_n: int = 10) -> dict:
    """
    Returns:
        {
            "prediction": "Fake" | "Real",
            "risk_percentage": float,      # probability of being fake, 0-100
            "model_confidence": float,     # confidence in the predicted class, 0-100
            "keywords": [str, ...]         # top words driving the prediction
        }
    """
    cleaned = clean_text(raw_text)
    X = vectorizer.transform([cleaned])

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    fake_idx = classes.index(1) if 1 in classes else int(np.argmax(proba))

    fake_prob = float(proba[fake_idx])
    prediction_label = "Fake" if fake_prob >= 0.5 else "Real"
    model_confidence = round(float(max(proba)) * 100, 2)
    risk_percentage = round(fake_prob * 100, 2)

    feature_names = np.array(vectorizer.get_feature_names_out())
    row = X.toarray()[0]
    nonzero_idx = np.nonzero(row)[0]

    keywords = []
    if len(nonzero_idx) > 0 and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_[nonzero_idx]
        tfidf_vals = row[nonzero_idx]
        scores = importances * tfidf_vals
        order = np.argsort(scores)[::-1][:top_n]
        keywords = [str(feature_names[nonzero_idx][i]) for i in order if scores[i] > 0]

    if not keywords:
        # Fallback: just show the highest TF-IDF terms present in the text
        top_idx = np.argsort(row)[::-1][:top_n]
        keywords = [str(feature_names[i]) for i in top_idx if row[i] > 0]

    return {
        "prediction": prediction_label,
        "risk_percentage": risk_percentage,
        "model_confidence": model_confidence,
        "keywords": keywords,
    }
