"""Run the full cryptocurrency sentiment analysis pipeline."""

import sys

from clean_text import clean_and_save
from vader_sentiment import analyze_sentiment
from tag_coins import tag_coins
from export_for_labeling import export_for_labeling
from merge_labels import merge_labels


def run_pipeline():
    """Execute the full pipeline in order."""
    steps = [
        ("Step 1: Cleaning tweet text...", clean_and_save),
        ("Step 2: Running VADER sentiment analysis...", analyze_sentiment),
        ("Step 3: Tagging tweets by cryptocurrency...", tag_coins),
        ("Step 4: Exporting sample for manual labeling...", export_for_labeling),
        ("Step 5: Merging manual labels back into dataset...", merge_labels),
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
