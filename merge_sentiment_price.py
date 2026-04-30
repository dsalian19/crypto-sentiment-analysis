import pandas as pd


def load_sentiment_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def load_price_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, skiprows=[1])  # Skip second row (ticker labels)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.rename(columns={'Date': 'date'})
    return df


def merge_datasets(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(sentiment_df, price_df, on='date', how='inner')
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def main():
    # Load data
    sentiment_df = load_sentiment_data('data/daily_sentiment.csv')
    price_df = load_price_data('data/price_data.csv')

    # Merge datasets
    merged_df = merge_datasets(sentiment_df, price_df)

    # Clean data (drop rows with missing values)
    merged_df = clean_data(merged_df)

    # Save results
    save_data(merged_df, 'data/merged_final.csv')

    # Print summary
    print(f"Shape: {merged_df.shape}")
    print(f"\nFirst few rows:")
    print(merged_df.head())


if __name__ == '__main__':
    main()
