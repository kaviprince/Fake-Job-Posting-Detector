"""
Step 1: Load the raw job postings CSV and produce a cleaned CSV ready for
feature extraction / training.

Usage:
    python preprocessing.py
    python preprocessing.py --input data/fake_job_postings.csv --output data/cleaned_data.csv
"""
import os
import argparse
import pandas as pd

from text_utils import clean_text

DEFAULT_INPUT = os.path.join("main", "data", "fake_job_postings.csv")
DEFAULT_OUTPUT = os.path.join("main", "data", "cleaned_data.csv")


def load_and_clean(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Dataset not found at '{input_path}'.\n"
            "Download the 'Fake Job Postings' dataset from Kaggle "
            "(https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) "
            "and place fake_job_postings.csv in a 'data' folder next to this script, "
            "or pass --input with the correct path."
        )

    df = pd.read_csv(input_path)

    if "description" not in df.columns:
        raise ValueError("Expected a 'description' column in the dataset.")

    df["description"] = df["description"].fillna("")

    # Combine a few useful text columns if present, for a richer signal
    combined = df["description"]
    for col in ["title", "requirements", "company_profile", "benefits"]:
        if col in df.columns:
            combined = combined + " " + df[col].fillna("")
    df["raw_text"] = combined

    print("Cleaning text (this can take a minute on the full dataset)...")
    df["clean_text"] = df["raw_text"].apply(clean_text)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to '{output_path}' ({len(df)} rows).")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the raw fake job postings dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to raw CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write cleaned CSV")
    args = parser.parse_args()
    load_and_clean(args.input, args.output)
