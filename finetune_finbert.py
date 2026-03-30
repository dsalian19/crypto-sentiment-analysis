"""Fine-tune ProsusAI/finbert on manually labeled crypto tweets.

Loads tweets_with_labels.csv, filters to labeled data, fine-tunes FinBERT
with 80/20 train/validation split, and saves the model.
"""

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset


def load_data(filepath: str) -> pd.DataFrame:
    """Load labeled tweets and filter to rows with manual labels."""
    df = pd.read_csv(filepath)
    # Filter to rows with manual labels
    df = df[df["label"].notna()].copy()
    # Map labels from -1, 0, 1 to 0, 1, 2 for model compatibility
    # -1 (negative) -> 0, 0 (neutral) -> 1, 1 (positive) -> 2
    label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
    df["label"] = df["label"].map(label_map).astype(int)
    return df


def prepare_datasets(df: pd.DataFrame, tokenizer, test_size: float = 0.2):
    """Split data and create HuggingFace Datasets."""
    # Split into train and validation
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    # Create HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df[["text_cleaned", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text_cleaned", "label"]])

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text_cleaned"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset = train_dataset.remove_columns(["text_cleaned"])
    val_dataset = val_dataset.remove_columns(["text_cleaned"])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).argmax(dim=-1)
    labels = torch.tensor(labels)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}


def main():
    # Configuration
    model_name = "ProsusAI/finbert"
    output_dir = "models/finbert_finetuned"
    data_path = "data/tweets_with_labels.csv"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} labeled tweets")
    print(f"Label distribution: {dict(df['label'].value_counts().sort_index())}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )

    print("\nPreparing datasets...")
    train_dataset, val_dataset = prepare_datasets(df, tokenizer, test_size=0.2)
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    val_accuracy = eval_results["eval_accuracy"]
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")

    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
