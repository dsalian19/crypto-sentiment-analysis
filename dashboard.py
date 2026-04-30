import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

df = pd.read_csv("data/merged_final.csv")
df["date"] = pd.to_datetime(df["date"])

# Filter for BTC only and aggregate by date
df_btc = df[df["coin"] == "BTC"].groupby("date").agg({
    "avg_compound": "mean",
    "avg_weighted_sentiment": "mean",
    "tweet_volume": "sum",
    "daily_return": "first"
}).reset_index()

st.title("Bitcoin Sentiment Dashboard")

# Sentiment vs price over time with secondary y-axis
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_btc["date"], y=df_btc["avg_compound"],
                          name="Avg Sentiment", yaxis="y1"))
fig1.add_trace(go.Scatter(x=df_btc["date"], y=df_btc["daily_return"],
                          name="Daily Return", yaxis="y2"))
fig1.update_layout(
    title="Sentiment vs BTC Daily Return Over Time",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Sentiment", side="left"),
    yaxis2=dict(title="Daily Return (%)", overlaying="y", side="right"),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# Tweet volume
fig2 = px.bar(df_btc, x="date", y="tweet_volume",
              title="Daily Tweet Volume")
st.plotly_chart(fig2, use_container_width=True)

# Raw vs weighted sentiment
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_btc["date"], y=df_btc["avg_compound"],
                          name="Raw Sentiment", mode="lines"))
fig3.add_trace(go.Scatter(x=df_btc["date"], y=df_btc["avg_weighted_sentiment"],
                          name="Weighted Sentiment", mode="lines"))
fig3.update_layout(
    title="Raw vs Engagement-Weighted Sentiment",
    xaxis_title="Date",
    yaxis_title="Sentiment Score",
    hovermode="x unified"
)
st.plotly_chart(fig3, use_container_width=True)

# Correlation summary
corr, pvalue = stats.pearsonr(df_btc["avg_compound"], df_btc["daily_return"])
st.subheader("Correlation Analysis")
st.metric("Pearson Correlation (Sentiment vs Return)", f"{corr:.4f}")
st.metric("P-value", f"{pvalue:.4f}")

if pvalue < 0.05:
    st.success("Correlation is statistically significant (p < 0.05)")
else:
    st.warning("Correlation is not statistically significant (p >= 0.05)")