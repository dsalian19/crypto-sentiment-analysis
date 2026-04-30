import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def filter_btc(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["coin"] == "BTC"].copy()


def compute_correlation(x: pd.Series, y: pd.Series, name: str) -> tuple:
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return None, None

    corr, p_value = pearsonr(x_clean, y_clean)
    return corr, p_value


def main():
    # Configuration
    data_path = "data/merged_final.csv"
    output_dir = "outputs"
    plot_path = os.path.join(output_dir, "sentiment_vs_return.png")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(data_path)

    print(f"Filtering to BTC data...")
    df_btc = filter_btc(df)
    print(f"BTC records: {len(df_btc)}")

    if len(df_btc) < 3:
        print("Error: Not enough BTC data points for correlation analysis")
        return

    # Compute correlations
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    corr_sentiment, p_val_sentiment = compute_correlation(
        df_btc["avg_compound"], df_btc["daily_return"], "Sentiment"
    )
    if corr_sentiment is not None:
        print(f"\n1. Average Compound Sentiment vs Daily Return")
        print(f"   Pearson r: {corr_sentiment:.4f}")
        print(f"   p-value:   {p_val_sentiment:.4f}")
        print(f"   Significance: {'Significant' if p_val_sentiment < 0.05 else 'Not significant'} (alpha=0.05)")
    corr_volume, p_val_volume = compute_correlation(
        df_btc["tweet_volume"], df_btc["daily_return"], "Tweet Volume"
    )
    if corr_volume is not None:
        print(f"\n2. Tweet Volume vs Daily Return")
        print(f"   Pearson r: {corr_volume:.4f}")
        print(f"   p-value:   {p_val_volume:.4f}")
        print(f"   Significance: {'Significant' if p_val_volume < 0.05 else 'Not significant'} (α=0.05)")

    corr_weighted, p_val_weighted = compute_correlation(
        df_btc["avg_weighted_sentiment"], df_btc["daily_return"], "Weighted Sentiment"
    )
    if corr_weighted is not None:
        print(f"\n3. Weighted Sentiment vs Daily Return")
        print(f"   Pearson r: {corr_weighted:.4f}")
        print(f"   p-value:   {p_val_weighted:.4f}")
        print(f"   Significance: {'Significant' if p_val_weighted < 0.05 else 'Not significant'} (α=0.05)")

    if corr_sentiment is not None:
        strength = "strong" if abs(corr_sentiment) > 0.5 else "moderate" if abs(corr_sentiment) > 0.3 else "weak"
        direction = "positive" if corr_sentiment > 0 else "negative"
        print(f"\nSentiment-Return correlation: {strength} {direction} relationship")
        print(f"  → Higher sentiment {'predicts higher' if corr_sentiment > 0 else 'does not predict'} returns")

    if corr_volume is not None:
        strength = "strong" if abs(corr_volume) > 0.5 else "moderate" if abs(corr_volume) > 0.3 else "weak"
        direction = "positive" if corr_volume > 0 else "negative"
        print(f"\nVolume-Return correlation: {strength} {direction} relationship")
        print(f"  → More tweets {'predict higher' if corr_volume > 0 else 'do not predict'} returns")

if __name__ == "__main__":
    main()
