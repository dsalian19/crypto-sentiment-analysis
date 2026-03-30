"""Tag tweets as Bitcoin-related or not."""

import pandas as pd


def tag_coins():
    """Load tweets and tag as BTC or other based on text content."""
    input_path = "data/tweets_vader.csv"
    output_path = "data/tweets_tagged.csv"

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df)} tweets from {input_path}")

    def categorize_coin(row):
        text = str(row.get("text_cleaned", "")).lower()
        original_text = str(row.get("text", "")).lower()

        combined_text = text + " " + original_text

        btc_keywords = ["bitcoin", "btc"]

        has_btc = any(kw in combined_text for kw in btc_keywords)

        if has_btc:
            return "BTC"
        else:
            return "other"

    print("Tagging tweets as BTC or other...")
    df["coin"] = df.apply(categorize_coin, axis=1)

    df.to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")
    print(f"\nCoin Distribution:")
    print(df["coin"].value_counts())


if __name__ == "__main__":
    tag_coins()
