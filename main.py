"""Run the full cryptocurrency sentiment analysis pipeline."""

import sys

from collect_tweets import collect_tweets
from preprocess import preprocess_data
from clean_text import clean_and_save
from vader_sentiment import analyze_sentiment
from tag_coins import tag_coins
from export_for_labeling import export_for_labeling
from merge_labels import merge_labels


def run_pipeline():
    """Execute the full pipeline in order."""
    steps = [
        ("Step 1: Collecting tweets from Twitter API...", collect_tweets),
        ("Step 2: Preprocessing tweet data...", preprocess_data),
        ("Step 3: Cleaning tweet text...", clean_and_save),
        ("Step 4: Running VADER sentiment analysis...", analyze_sentiment),
        ("Step 5: Tagging tweets by cryptocurrency...", tag_coins),
        ("Step 6: Exporting sample for manual labeling...", export_for_labeling),
        ("Step 7: Merging manual labels back into dataset...", merge_labels),
    ]

    for step_name, step_func in steps:
        print(f"\n{'=' * 60}")
        print(step_name)
        print('=' * 60)
        try:
            step_func()
        except Exception as e:
            print(f"\n❌ Pipeline failed during: {step_name}")
            print(f"Error: {e}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("✅ Pipeline complete!")
    print("Output saved to data/tweets_with_labels.csv")
    print('=' * 60)


if __name__ == "__main__":
    run_pipeline()
