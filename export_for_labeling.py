"""Export tweets for manual labeling."""

import pandas as pd


def export_for_labeling():
    """Export a sample of tweets for manual labeling."""
    input_path = "data/tweets_tagged.csv"
    output_path = "data/tweets_to_label.csv"

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

    sample_size = min(1000, len(df))

    if sample_size < len(df):
        print(f"Sampling {sample_size} tweets (random_state=42)...")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        print(f"Using all {sample_size} tweets (less than 500)")
        df_sample = df.copy()

    # Use original index as ID to enable proper merging later
    df_export = df_sample[["text"]].copy()
    df_export.insert(0, "id", df_sample.index)

    df_export["label"] = ""

    df_export.to_csv(output_path, index=False)

    print(f"Exported {len(df_export)} tweets to {output_path}")
    print(f"Columns: {list(df_export.columns)}")


if __name__ == "__main__":
    export_for_labeling()
