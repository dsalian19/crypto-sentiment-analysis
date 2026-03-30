import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/tweets_tagged.csv")

# Tweet volume over time
df["date"] = pd.to_datetime(df["created_at"]).dt.date
# df.groupby("date")["text"].count().plot(kind="bar", title="Tweet Volume by Day")
# plt.savefig("outputs/tweet_volume.png")

# Sentiment distribution
df["sentiment"].value_counts().plot(kind="pie", title="Sentiment Distribution")
plt.savefig("outputs/sentiment_dist.png")

# # Average compound score over time
# df.groupby("date")["compound"].mean().plot(title="Average Sentiment Over Time")
# plt.savefig("outputs/sentiment_over_time.png")