"""Load and inspect the collected tweets dataset."""

import pandas as pd


def inspect_data(filepath: str = "data/tweets_raw.csv") -> None:
    """
    Load the tweets CSV file and print inspection information.

    Args:
        filepath: Path to the tweets CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("=" * 60)
    print("TWEETS DATASET INSPECTION")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nFirst 5 Rows (text truncated):")
    for i in range(min(5, len(df))):
        text = str(df.iloc[i]['text'])[:80].encode('ascii', 'ignore').decode('ascii')
        print(f"  Row {i+1}: {text}...")

    print(f"\nData Types:")
    print(df.dtypes)

    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct.round(2)
    })
    print(missing_df)

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    inspect_data()
