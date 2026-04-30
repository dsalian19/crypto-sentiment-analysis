import pandas as pd


def preprocess_data(path):
    input_path = path
    output_path = "data/tweets_standardized.csv"

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Original shape: {df.shape}")

    column_mapping = {
        "like_count": "likes",
        "retweet_count": "retweets",
        "reply_count": "replies"
    }
    df = df.rename(columns=column_mapping)

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    if "text" in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=["text"])
        dropped = initial_rows - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing text")

    df.to_csv(output_path, index=False)

    print(f"Cleaned shape: {df.shape}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    preprocess_data()
