# Fake Job Posting Detector

Detects fraudulent job postings from pasted text, a job-post URL, or an
uploaded image, using TF-IDF + a Random Forest classifier.

## Project structure

```
text_utils.py         # shared text-cleaning function (used everywhere)
preprocessing.py       # step 1: clean the raw dataset
feature_extraction.py  # step 2: TF-IDF vectorizer helpers
model_train.py          # step 3: train + save the model
explain.py               # step 4: prediction + keyword explanation
app.py                    # Streamlit web app
data/                      # put fake_job_postings.csv here
models/                     # trained model + vectorizer get saved here
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the dataset ("Fake Job Postings", Kaggle) and place it at:
   ```
   data/fake_job_postings.csv
   ```
   https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

3. (Optional, for image OCR) Install Tesseract:
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki

## Train the model

```
python preprocessing.py
python model_train.py
```

This creates `data/cleaned_data.csv` and `models/fake_job_model.pkl` +
`models/tfidf_vectorizer.pkl`, and prints accuracy/precision/recall.

## Run the app

```
streamlit run app.py
```

Then paste a job description, a job posting URL, or upload a screenshot,
and click **Predict**.

## Notes

- The dataset is heavily imbalanced (far more real postings than fake
  ones), so the model uses `class_weight="balanced"`. If you want to
  compare models, try `LogisticRegression` or `XGBoost` in
  `model_train.py`.
- URL scraping is best-effort generic HTML text extraction — some sites
  (e.g. LinkedIn) block automated requests, in which case paste the
  description text directly instead.
