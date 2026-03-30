"""Clean and normalize tweet text."""

import re

import pandas as pd


def clean_tweet_text(text: str) -> str:
    """
    Clean tweet text by removing URLs, mentions, cashtags, punctuation,
    and normalizing whitespace.

    Args:
        text: Raw tweet text.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http[s]?://\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"\$\w+", "", text)

    text = re.sub(r"#(\w+)", r"\1", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip().lower()

    return text


def clean_and_save():
    """Load standardized tweets, clean text, and save."""
    input_path = "data/tweets_standardized.csv"
    output_path = "data/tweets_cleaned.csv"

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df)} tweets from {input_path}")

    if "text" not in df.columns:
        print("Error: 'text' column not found in dataset")
        return

    print("Cleaning tweet text...")
    df["text_cleaned"] = df["text"].apply(clean_tweet_text)

    empty_after_cleaning = df["text_cleaned"].eq("").sum()
    if empty_after_cleaning > 0:
        print(f"Warning: {empty_after_cleaning} tweets are empty after cleaning")

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} cleaned tweets to {output_path}")
    print("\nSample before/after:")
    for i in range(min(3, len(df))):
        print(f"  Original: {df['text'].iloc[i][:60]}...")
        print(f"  Cleaned:  {df['text_cleaned'].iloc[i][:60]}...")
        print()


if __name__ == "__main__":
    clean_and_save()
