"""Apply VADER sentiment analysis to cleaned tweets."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiment():
    """Load cleaned tweets, apply VADER sentiment, and save results."""
    input_path = "data/tweets_cleaned.csv"
    output_path = "data/tweets_vader.csv"

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df)} tweets from {input_path}")

    if "text_cleaned" not in df.columns:
        print("Error: 'text_cleaned' column not found in dataset")
        return

    analyzer = SentimentIntensityAnalyzer()

    print("Applying VADER sentiment analysis...")
    df["compound"] = df["text_cleaned"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    def categorize_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment"] = df["compound"].apply(categorize_sentiment)

    df.to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")
    print(f"\nSentiment Distribution:")
    print(df["sentiment"].value_counts())
    print(f"\nPercentages:")
    print(df["sentiment"].value_counts(normalize=True).mul(100).round(2))


if __name__ == "__main__":
    analyze_sentiment()
