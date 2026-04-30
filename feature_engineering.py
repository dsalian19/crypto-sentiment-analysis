import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def create_engagement(df: pd.DataFrame) -> pd.DataFrame:
    if 'likes' in df.columns and 'retweets' in df.columns and 'replies' in df.columns:
        df['engagement'] = df['likes'] + df['retweets'] + df['replies'] + 1
    else:
        # Default engagement for tweets without social metrics (like from CSV dataset)
        df['engagement'] = 1.0
        print("⚠ No engagement metrics found (likes/retweets/replies). Using default engagement=1.0")
    return df


def create_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    finbert_score_map = {"positive": 1, "negative": -1, "neutral": 0}
    if "finbert_label" in df.columns:
        df["finbert_score"] = df["finbert_label"].map(finbert_score_map) * df['engagement']
    if "vader_compound" in df.columns:
        df["vader_weighted"] = df["vader_compound"] * df['engagement']
    return df


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['date'] = df['Date'].dt.date
    else:
        raise ValueError("No date column found. Expected 'created_at' or 'Date'")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    text_col = 'text' if 'text' in df.columns else 'Content' if 'Content' in df.columns else None
    if text_col is None:
        raise ValueError("No text column found. Expected 'text' or 'Content'")
    
    daily = df.groupby(['date']).agg(
        avg_vader_sentiment=('vader_weighted', 'mean'),
        avg_finbert_sentiment=('finbert_score', 'mean'),
        tweet_volume=(text_col, 'count')
    ).reset_index()
    return daily


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def main():
    # Load data
    df = load_data('data/tweets_finbert_vader.csv')
    print(f"✓ Loaded {len(df)} tweets")
    print(f"  Columns: {df.columns.tolist()}")

    # Create features
    df = create_engagement(df)
    df = create_weighted_score(df)
    df = extract_date(df)
    print("✓ Features created")

    # Aggregate by date and coin
    daily_df = aggregate_daily(df)
    print(f"✓ Aggregated to daily sentiment")

    # Save results
    save_data(daily_df, 'data/daily_sentiment.csv')
    print(f"✓ Saved to data/daily_sentiment.csv")

    # Print summary
    print(f"\nSummary:")
    print(f"  Shape: {daily_df.shape}")
    print(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print("\nFirst few rows:")
    print(daily_df.head())


if __name__ == '__main__':
    main()
