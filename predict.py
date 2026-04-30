import pandas as pd
from transformers import pipeline
from tqdm import tqdm


def load_model(model_path: str):
    print(f"Loading fine-tuned model from {model_path}...")
    try:
        pipe = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path
        )
        print("Model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def run_predictions(pipe, texts: list, batch_size: int = 32) -> tuple:
    labels = []
    confidences = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Running predictions"):
        batch = texts[i:i + batch_size]
        try:
            results = pipe(batch, truncation=True, max_length=512)
            for r in results:
                labels.append(r["label"])
                confidences.append(r["score"])
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {e}")
            # Fill failed batch with unknowns so row count stays consistent
            for _ in batch:
                labels.append("unknown")
                confidences.append(0.0)

    return labels, confidences


def predict_and_save():
    input_path = "data/tweets_cleaned.csv"
    output_path = "data/tweets_predictions.csv"
    model_path = "models/finbert_finetuned"

    # Load cleaned tweets
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df)} tweets from {input_path}")

    if "text_cleaned" not in df.columns:
        print("Error: 'text_cleaned' column not found in dataset")
        return

    # Drop rows with empty cleaned text
    before = len(df)
    df = df[df["text_cleaned"].notna() & df["text_cleaned"].ne("")]
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with empty cleaned text")

    # Load model
    pipe = load_model(model_path)

    # Run predictions
    print(f"\nRunning predictions on {len(df)} tweets in batches of 32...")
    texts = df["text_cleaned"].tolist()
    labels, confidences = run_predictions(pipe, texts)

    # Add predictions to dataframe
    df["finbert_label"] = labels
    df["finbert_confidence"] = confidences

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

    # Print distribution
    print("\nPrediction label distribution:")
    print(df["finbert_label"].value_counts())

    # Print average confidence per label
    print("\nAverage confidence per label:")
    print(df.groupby("finbert_label")["finbert_confidence"].mean().round(4))


if __name__ == "__main__":
    ()