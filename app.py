import os
import shutil
import platform
import pickle
from io import BytesIO
from datetime import datetime

import streamlit as st
import requests
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup

from explain import explain_prediction

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="wide"
)

# =============================
# TESSERACT SETUP (cross-platform)
# =============================
# On Windows, the tesseract.exe path often isn't on PATH by default, so we
# point pytesseract at the common install location if we can find it.
# On macOS/Linux, `tesseract` is normally already on PATH once installed
# (e.g. via `brew install tesseract` or `apt install tesseract-ocr`).
_OCR_AVAILABLE = True
if platform.system() == "Windows":
    _default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_win_path):
        pytesseract.pytesseract.tesseract_cmd = _default_win_path
    elif shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
    else:
        _OCR_AVAILABLE = False
else:
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
    else:
        _OCR_AVAILABLE = False

# =============================
# LOAD MODEL & VECTORIZER
# =============================
model = None

MODEL_PATH = os.path.join("main", "models", "fake_job_model.pkl")
VECTORIZER_PATH = os.path.join("main", "models", "tfidf_vectorizer.pkl")

vectorizer = None
_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    _load_error = (
        f"Model files not found at '{MODEL_PATH}' / '{VECTORIZER_PATH}'.\n\n"
        "Run the training pipeline first:\n"
        "1. `python preprocessing.py`\n"
        "2. `python model_train.py`\n\n"
        "This will create the `models/` folder with the trained model and vectorizer."
    )
except Exception as e:
    _load_error = f"Failed to load model files: {e}"


# =============================
# URL -> TEXT (best-effort scraper)
# =============================
def is_url(text: str) -> bool:
    return text.strip().lower().startswith(("http://", "https://"))


def extract_text_from_job_url(url: str) -> str:
    """Best-effort extraction of visible text from a job posting page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        return text[:20000]  # cap length to keep things fast
    except Exception:
        return ""


# =============================
# OCR FUNCTION (IMAGE URL) — kept for potential future use
# =============================
def extract_text_from_image_url(url: str) -> str:
    if not _OCR_AVAILABLE:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            return ""
        image = Image.open(BytesIO(response.content))
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


# =============================
# GENERATE REPORT FUNCTION
# =============================
def generate_report(input_type, input_text, result, job_title="Not Provided"):
    date_now = datetime.now().strftime("%d-%m-%Y")

    report = f"""
FAKE JOB POSTING DETECTOR – AUTOMATIC ANALYSIS REPORT

Job Posting Analysis Report

Job Title: {job_title}
Platform Source: {input_type}
Date of Analysis: {date_now}

Prediction Result

Job Authenticity Status:
{"Fake Job" if result["prediction"] == "Fake" else "Real Job"}

Risk Percentage:
{result["risk_percentage"]}% probability of being Fake

Model Confidence:
The model is {result["model_confidence"]}% confident in this prediction.

Summary

This job posting was analyzed using a machine learning-based Fake Job Posting Detector.
The system examined the job description text obtained from user input and identified
patterns commonly associated with fraudulent job advertisements.

Key Indicators Identified
"""

    for word in result["keywords"]:
        report += f"- {word}\n"

    report += """
Model Details

Text Processing:
Stopword removal, punctuation removal, lowercase conversion

Feature Extraction:
TF-IDF Vectorization

Model Used:
Random Forest Classifier

Prediction Type:
Probability-based classification

Risk Interpretation

0-10% -> Very Low Risk (Highly Likely Real Job)
11-30% -> Low Risk (Likely Real Job)
31-60% -> Medium Risk (Caution Advised)
61-100% -> High Risk (Likely Fake Job)

Conclusion

Based on the analysis, users are advised to verify company details, avoid sharing
personal information, and proceed cautiously if the risk percentage is high.
"""
    return report


# =============================
# HEADER
# =============================
st.title("🕵️ Fake Job Posting Detector")
st.markdown(
    """
Detect **fake job postings** using Machine Learning.

**You can provide:**
- 📝 Job description text
- 🌐 Job post URL (LinkedIn, Indeed, etc.)
- 🖼 Upload a job image (poster / screenshot)
"""
)

if _load_error:
    st.error(_load_error)

if not _OCR_AVAILABLE:
    st.info(
        "ℹ️ Tesseract OCR was not found on this system, so image upload won't work. "
        "Install it to enable OCR: `brew install tesseract` (macOS), "
        "`sudo apt install tesseract-ocr` (Linux), or download it for Windows."
    )

st.divider()

# =============================
# LAYOUT
# =============================
left_col, right_col = st.columns(2)

# =============================
# INPUT SECTION
# =============================
with left_col:
    st.subheader("📥 Input")

    job_title = st.text_input(
        "Job Title (optional, will appear in report)",
        placeholder="Enter job title here"
    )

    user_input = st.text_area(
        "Text or Job Post URL",
        height=200,
        placeholder="Paste job description OR job post URL here"
    )

    uploaded_image = st.file_uploader(
        "Optional: Upload job image",
        type=["png", "jpg", "jpeg"]
    )

    predict_btn = st.button("🔍 Predict", disabled=(model is None or vectorizer is None))

# =============================
# RESULT SECTION
# =============================
with right_col:
    st.subheader("📊 Result")

    if predict_btn:
        final_text = ""
        input_type = "Text"

        # Priority 1: Uploaded Image
        if uploaded_image is not None:
            if not _OCR_AVAILABLE:
                st.error("❌ OCR is unavailable — Tesseract isn't installed on this system.")
            else:
                try:
                    final_text = pytesseract.image_to_string(Image.open(uploaded_image))
                    input_type = "Uploaded Image"
                    with st.expander("📄 Extracted Text (Uploaded Image)"):
                        st.write(final_text)
                except Exception:
                    st.error("❌ Unable to read uploaded image")

        # Priority 2: URL
        elif user_input.strip() != "" and is_url(user_input.strip()):
            with st.spinner("Fetching job posting from URL..."):
                final_text = extract_text_from_job_url(user_input.strip())
            input_type = "Job Post URL"
            if final_text:
                with st.expander("📄 Extracted Text (from URL)"):
                    st.write(final_text[:3000] + ("..." if len(final_text) > 3000 else ""))
            else:
                st.error(
                    "❌ Couldn't fetch or parse that URL. Some sites block automated "
                    "requests — try pasting the job description text instead."
                )

        # Priority 3: Plain text
        elif user_input.strip() != "":
            final_text = user_input
            input_type = "Pasted Text"

        else:
            st.warning("⚠️ Please enter text, a job URL, or upload an image")

        # MODEL PREDICTION
        if final_text.strip() != "":
            result = explain_prediction(final_text, model, vectorizer)

            prediction = result["prediction"]
            confidence = result["model_confidence"]
            risk = result["risk_percentage"]

            if prediction == "Fake":
                st.error("🚨 Prediction: FAKE JOB")
            else:
                st.success("✅ Prediction: REAL JOB")

            st.metric("Model Confidence", f"{confidence}%")
            st.metric("Risk Percentage", f"{risk}%")

            st.subheader("🔑 Key Influencing Words")
            if result["keywords"]:
                for word in result["keywords"]:
                    st.write(f"• {word}")
            else:
                st.write("No strong keyword signals found for this text.")

            report_text = generate_report(
                input_type=input_type,
                input_text=final_text,
                result=result,
                job_title=job_title if job_title.strip() != "" else "Not Provided"
            )

            st.subheader("📝 Automatic Job Analysis Report")
            st.text(report_text)

            st.download_button(
                label="Download Analysis Report",
                data=report_text,
                file_name="Fake_Job_Analysis_Report.txt",
                mime="text/plain"
            )
