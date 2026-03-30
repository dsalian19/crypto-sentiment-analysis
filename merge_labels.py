"""Merge manual labels back into the tagged tweets dataset."""

import pandas as pd


def merge_labels():
    """Load tagged tweets and manual labels, merge on id (which is original index)."""
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

    # Check required columns
    if "label" not in df_labeled.columns:
        print("Error: 'label' column not found in labeled data")
        return

    if "id" not in df_labeled.columns:
        print("Error: 'id' column not found in labeled data")
        return

    # Use index as the merge key
    df_tagged = df_tagged.reset_index().rename(columns={"index": "id"})

    # Ensure id columns are same type
    df_labeled["id"] = df_labeled["id"].astype(int)
    df_tagged["id"] = df_tagged["id"].astype(int)

    df_merged = df_tagged.merge(
        df_labeled[["id", "label"]],
        on="id",
        how="left"
    )

    # Add VADER label mapped to same scale (-1, 0, 1)
    vader_map = {"negative": -1, "neutral": 0, "positive": 1}
    df_merged["vader_label"] = df_merged["sentiment"].map(vader_map)

    labeled_count = df_merged["label"].notna().sum()
    print(f"\nMerged {labeled_count} manual labels into dataset")

    # Calculate agreement between VADER and manual labels
    labeled_only = df_merged.dropna(subset=["label"])
    if len(labeled_only) > 0:
        agreement = (labeled_only["label"] == labeled_only["vader_label"]).sum()
        print(f"VADER agreement with manual labels: {agreement}/{len(labeled_only)} ({agreement/len(labeled_only)*100:.1f}%)")

        # Breakdown by label
        print("\nLabel distribution in manual labels:")
        print(labeled_only["label"].value_counts().sort_index())

    df_merged.to_csv(output_path, index=False)

    print(f"\nSaved {len(df_merged)} tweets to {output_path}")
    print(f"Columns: {list(df_merged.columns)}")


if __name__ == "__main__":
    merge_labels()
