"""Merge manual labels back into the tagged tweets dataset."""

import pandas as pd


def merge_labels():
    """Load tagged tweets and manual labels, merge on id column."""
    tagged_path = "data/tweets_tagged.csv"
    labeled_path = "data/tweets_labeled.csv"
    output_path = "data/tweets_with_labels.csv"

    try:
        df_tagged = pd.read_csv(tagged_path)
    except FileNotFoundError:
        print(f"Error: File not found at {tagged_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    try:
        df_labeled = pd.read_csv(labeled_path)
    except FileNotFoundError:
        print(f"Error: File not found at {labeled_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df_tagged)} tweets from {tagged_path}")
    print(f"Loaded {len(df_labeled)} labeled tweets from {labeled_path}")

    if "id" not in df_tagged.columns:
        print("Error: 'id' column not found in tagged data")
        return

    if "id" not in df_labeled.columns:
        print("Error: 'id' column not found in labeled data")
        return

    if "label" not in df_labeled.columns:
        print("Error: 'label' column not found in labeled data")
        return

    df_labeled["id"] = df_labeled["id"].astype(str)
    df_tagged["id"] = df_tagged["id"].astype(str)

    df_merged = df_tagged.merge(
        df_labeled[["id", "label"]],
        on="id",
        how="left"
    )

    labeled_count = df_merged["label"].notna().sum()
    print(f"Merged {labeled_count} labels into dataset")

    df_merged.to_csv(output_path, index=False)

    print(f"Saved {len(df_merged)} tweets to {output_path}")
    print(f"Columns: {list(df_merged.columns)}")


if __name__ == "__main__":
    merge_labels()
