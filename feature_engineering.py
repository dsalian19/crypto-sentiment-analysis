"""Feature engineering script for crypto sentiment analysis.

Loads tweets_tagged.csv, creates engagement and weighted features,
aggregates by date and coin, and saves daily sentiment data.
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the tagged tweets CSV."""
    return pd.read_csv(filepath)


def create_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Create engagement column: likes + retweets + replies + 1."""
    df['engagement'] = df['likes'] + df['retweets'] + df['replies'] + 1
    return df


def create_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create weighted_score column: compound * engagement."""
    df['weighted_score'] = df['compound'] * df['engagement']
    return df


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse created_at to extract a date column."""
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Group by date and coin, aggregate sentiment metrics."""
    daily = df.groupby(['date', 'coin']).agg(
        avg_compound=('compound', 'mean'),
        avg_weighted_sentiment=('weighted_score', 'mean'),
        tweet_volume=('text', 'count')
    ).reset_index()
    return daily


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save the aggregated dataframe to CSV."""
    df.to_csv(filepath, index=False)


def main():
    # Load data
    df = load_data('data/tweets_tagged.csv')

    # Create features
    df = create_engagement(df)
    df = create_weighted_score(df)
    df = extract_date(df)

    # Aggregate by date and coin
    daily_df = aggregate_daily(df)

    # Save results
    save_data(daily_df, 'data/daily_sentiment.csv')

    # Print summary
    print(f"Shape: {daily_df.shape}")
    print("\nFirst few rows:")
    print(daily_df.head())


if __name__ == '__main__':
    main()
