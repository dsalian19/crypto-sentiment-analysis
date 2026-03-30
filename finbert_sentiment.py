"""Apply FinBERT sentiment analysis to cleaned tweets.

Loads tweets_cleaned.csv, applies the ProsusAI/finbert model using
HuggingFace transformers pipeline, adds label and confidence columns,
and saves the results.
"""

import pandas as pd
from transformers import pipeline
from tqdm import tqdm


def load_data(filepath: str) -> pd.DataFrame:
    """Load the cleaned tweets CSV."""
    return pd.read_csv(filepath)


def create_finbert_pipeline():
    """Create a FinBERT sentiment analysis pipeline."""
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1  # Use CPU
    )


def process_batch(classifier, texts: list) -> tuple[list, list]:
    """Process a batch of texts and return labels and confidence scores."""
    results = classifier(texts, truncation=True, max_length=512)
    labels = [r["label"] for r in results]
    confidences = [r["score"] for r in results]
    return labels, confidences


def apply_finbert(df: pd.DataFrame, classifier, batch_size: int = 32) -> pd.DataFrame:
    """Apply FinBERT to the text_cleaned column in batches.

    Args:
        df: DataFrame with text_cleaned column
        classifier: HuggingFace sentiment analysis pipeline
        batch_size: Number of texts to process at once

    Returns:
        DataFrame with finbert_label and finbert_confidence columns
    """
    texts = df["text_cleaned"].astype(str).tolist()
    all_labels = []
    all_confidences = []

    # Process in batches with progress bar
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Processing batches"):
        batch = texts[i:i + batch_size]
        labels, confidences = process_batch(classifier, batch)
        all_labels.extend(labels)
        all_confidences.extend(confidences)

    df["finbert_label"] = all_labels
    df["finbert_confidence"] = all_confidences
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save the dataframe with FinBERT results to CSV."""
    df.to_csv(filepath, index=False, encoding="utf-8")


def main():
    print("Loading data...")
    df = load_data("data/tweets_cleaned.csv")

    print("Loading FinBERT model...")
    classifier = create_finbert_pipeline()

    print("Applying FinBERT sentiment analysis...")
    df = apply_finbert(df, classifier, batch_size=32)

    print("Saving results...")
    save_data(df, "data/tweets_finbert.csv")

    print(f"\nShape: {df.shape}")
    print("\nFirst few rows:")
    print(df[["text_cleaned", "finbert_label", "finbert_confidence"]].head())


if __name__ == "__main__":
    main()
