"""Download BTC-USD price data from Yahoo Finance.

Downloads daily price data for the date range found in daily_sentiment.csv,
calculates daily returns, and saves to CSV.
"""

import yfinance as yf
import pandas as pd


def get_date_range(filepath: str) -> tuple[str, str]:
    """Read daily_sentiment.csv and return min/max dates."""
    df = pd.read_csv(filepath)
    min_date = df['date'].min()
    max_date = df['date'].max()
    return str(min_date), str(max_date)


def download_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download price data from Yahoo Finance.

    Note: yfinance end date is exclusive, so we add 1 day to include the last date.
    """
    # Add one day to end_date to make it inclusive
    end_plus_one = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=start_date, end=end_plus_one, progress=False)
    df = df.reset_index()
    return df


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily_return column as percentage change in closing price."""
    df['daily_return'] = df['Close'].pct_change() * 100
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save the price dataframe to CSV."""
    df.to_csv(filepath, index=False)


def main():
    # Get date range from sentiment data
    start_date, end_date = get_date_range('data/daily_sentiment.csv')
    print(f"Downloading BTC-USD data from {start_date} to {end_date}")

    # Download price data
    price_df = download_price_data('BTC-USD', start_date, end_date)

    # Add daily return column
    price_df = add_daily_return(price_df)

    # Save results
    save_data(price_df, 'data/price_data.csv')

    # Print summary
    print(f"\nShape: {price_df.shape}")
    print(f"Date range: {price_df['Date'].min()} to {price_df['Date'].max()}")
    print("\nFirst few rows:")
    print(price_df.head())


if __name__ == '__main__':
    main()
