import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def load_data():
    # Load data with manual labels
    df_labels = pd.read_csv("data/tweets_with_labels.csv")

    # Load FinBERT predictions
    df_finbert = pd.read_csv("data/tweets_predictions.csv")

    # Merge on text column
    df = df_labels.merge(
        df_finbert[["text", "finbert_label"]],
        on="text",
        how="left"
    )

    # Filter to rows with manual labels
    df = df[df["label"].notna()].copy()

    return df


def prepare_labels(df):
    label_map_reverse = {-1.0: "negative", 0.0: "neutral", 1.0: "positive"}
    df["manual_label"] = df["label"].map(label_map_reverse)

    df["vader_pred"] = df["sentiment"]

    df["finbert_pred"] = df["finbert_label"]

    return df


def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print(f"{'='*60}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Print with headers
    print(f"{'':15} {'pred: neg':>12} {'pred: neu':>12} {'pred: pos':>12}")
    for i, row_label in enumerate(labels):
        row_str = f"{'true: ' + row_label:15}"
        for j, val in enumerate(cm[i]):
            row_str += f"{val:>12}"
        print(row_str)

    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} tweets with manual labels\n")

    df = prepare_labels(df)

    # Evaluate VADER
    evaluate_model(df["manual_label"], df["vader_pred"], "VADER")

    # Evaluate FinBERT
    evaluate_model(df["manual_label"], df["finbert_pred"], "FinBERT")

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")

    vader_acc = (df["manual_label"] == df["vader_pred"]).mean()
    finbert_acc = (df["manual_label"] == df["finbert_pred"]).mean()

    print(f"VADER Accuracy:    {vader_acc:.4f} ({vader_acc*100:.2f}%)")
    print(f"FinBERT Accuracy:  {finbert_acc:.4f} ({finbert_acc*100:.2f}%)")
    print(f"\nImprovement:      +{finbert_acc - vader_acc:.4f} ({(finbert_acc - vader_acc)*100:.2f} percentage points)")


if __name__ == "__main__":
    main()
