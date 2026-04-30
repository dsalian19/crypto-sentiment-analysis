import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/tweets_tagged.csv")

df["date"] = pd.to_datetime(df["created_at"]).dt.date
df["sentiment"].value_counts().plot(kind="pie", title="Sentiment Distribution")
plt.savefig("outputs/sentiment_dist.png")