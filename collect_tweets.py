"""Collect tweets from Twitter/X API using Tweepy."""

import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
import tweepy


def collect_tweets():
    """Collect tweets about Bitcoin and Ethereum using Twitter API v2."""
    load_dotenv()

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN not found in .env file")

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    query = "(Bitcoin OR BTC OR Ethereum OR ETH) lang:en -is:retweet"

    tweets_data = []

    max_results = 500
    batch_size = 100

    print(f"Collecting tweets with query: {query}")
    print(f"Target: up to {max_results} tweets")

    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["created_at", "public_metrics"],
        max_results=100,
        limit=5
    )

    for response in paginator:
        if response.data:
            for tweet in response.data:
                metrics = tweet.public_metrics
                tweets_data.append({
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "author_id": tweet.author_id,
                    "like_count": metrics.get("like_count", 0),
                    "retweet_count": metrics.get("retweet_count", 0),
                    "reply_count": metrics.get("reply_count", 0),
                })

            print(f"  Collected {len(tweets_data)} tweets so far...")

            if len(tweets_data) >= max_results:
                tweets_data = tweets_data[:max_results]
                break

    if not tweets_data:
        print("No tweets collected.")
        return

    df = pd.DataFrame(tweets_data)

    os.makedirs("data", exist_ok=True)
    output_path = "data/tweets_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved {len(df)} tweets to {output_path}")
    print(f"📊 Shape: {df.shape}")


if __name__ == "__main__":
    collect_tweets()
