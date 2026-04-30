import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
import tweepy


def collect_tweets():
    load_dotenv()

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN not found in .env file")

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    query = "(Bitcoin OR BTC) lang:en -is:retweet -is:reply -is:quote"

    tweets_data = []

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=40)

    print(f"Collecting tweets from {start_date.date()} to {end_date.date()}")
    print(f"Target: 100 tweets per day x 40 days = 4000 tweets")
    print(f"Query: {query}")
    print("=" * 60)

    for day in range(40):
        day_start = start_date + timedelta(days=day)
        day_end = day_start + timedelta(days=1)

        day_start_str = day_start.strftime("%Y-%m-%d")

        print(f"Day {day + 1}/40: {day_start_str} ...", end=" ")

        try:
            response = client.search_all_tweets(
                query=query,
                tweet_fields=["created_at", "public_metrics"],
                max_results=100,
                start_time=day_start,
                end_time=day_end
            )

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
                print(f"Collected {len(response.data)} tweets")
            else:
                print("No tweets found")

        except Exception as e:
            print(f"Error: {e}")

    print("=" * 60)

    if not tweets_data:
        print("No tweets collected.")
        return

    df = pd.DataFrame(tweets_data)

    os.makedirs("data", exist_ok=True)
    output_path = "data/tweets_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} tweets to {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    collect_tweets()
