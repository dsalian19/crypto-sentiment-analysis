import re

import pandas as pd


def clean_tweet_text(text: str) -> str:
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


SPAM_KEYWORDS = [
    "giveaway", "airdrop", "win", "click here", "join now",
    "claim", "free", "promo", "discount", "follow us",
    "dm us", "limited offer", "sign up", "register now",
    "100x", "guaranteed", "profit"
]


def is_spam(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SPAM_KEYWORDS)


def filter_spam(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df["text"].apply(is_spam)]
    after = len(df)
    removed = before - after
    print(f"Spam filter: removed {removed} tweets ({removed/before*100:.1f}%)")
    print(f"Remaining tweets: {after}")
    return df


def clean_and_save():
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

    print("\nFiltering spam and bot tweets...")
    df = filter_spam(df)

    print("\nCleaning tweet text...")
    df["text_cleaned"] = df["text"].apply(clean_tweet_text)

    empty_after_cleaning = df["text_cleaned"].eq("").sum()
    if empty_after_cleaning > 0:
        print(f"Warning: {empty_after_cleaning} tweets are empty after cleaning")

    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} cleaned tweets to {output_path}")


if __name__ == "__main__":
    clean_and_save()