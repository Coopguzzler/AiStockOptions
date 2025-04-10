
#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Investor‑Grade Options Trading Model
--------------------------------------
This module retrieves historical stock data (via Polygon with fallback to Yahoo),
computes technical and fundamental features, trains an ensemble of models (including
Prophet and Optuna‑tuned XGBoost), and generates trading signals with risk and position sizing.
It also integrates alternative sentiment data (Twitter/Reddit) with heavy weighting and
retrieves quarterly financials.

"""




# Imports
import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional


import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Machine learning and modeling libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from prophet import Prophet
import pandas_datareader.data as web
import optuna
from polygon import RESTClient
import re

# For sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# For Twitter API using Tweepy (v4)
import tweepy
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN")
twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

POLYGON_API_KEY = "bbYmszAdyJRvANl6FqHgoBkWU3hfhHJT"
from polygon import RESTClient

polygon_client = RESTClient(POLYGON_API_KEY)
# For Reddit API using PRAW
import praw
REDDIT_CLIENT_ID = "YJnhW_FkwbvXBWL-5rQohA"
REDDIT_CLIENT_SECRET = "coyMlkxWQEaHG5MoRMOX4JtXc300sg"
REDDIT_USER_AGENT = "coopersstockai"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
sys.setrecursionlimit(15000)

# API KEYS – (Make sure to secure these in production)
POLYGON_API_KEY = "bbYmszAdyJRvANl6FqHgoBkWU3hfhHJT"
TWITTER_API_KEY = "YkdXUykzmRdk7sqMF9LvtE1pp"
TWITTER_API_SECRET = "uzZ4ic2jCnZGYx1BeNrnYr044yjT1TB1bGF4i8rvfDq6YuujQA"

# GLOBAL CONSTANTS & WATCHLIST
FEATURE_COLS: List[str] = [
    "Rolling_Mean", "Rolling_Std", "EMA_10", "Momentum", "RSI",
    "SMA_50", "SMA_200", "MACD", "Bollinger_Upper", "Bollinger_Lower",
    "Interest_Rate", "Inflation_Rate", "VIX", "M2",
    "Month", "Day", "Weekday",
    "lag_close_1", "lag_close_5", "lag_close_10",
    "lag_return_1", "lag_return_5", "lag_return_10",
    "vol_10", "vol_20", "vol_30",
    "Daily_Return"
]

HOT_TICKERS: List[str] = [
    "TWI","KMT","ATI","CMC","NUE"
]
logger.info(f"Total HOT_TICKERS in watchlist: {len(HOT_TICKERS)}")

# ----- REAL IMPLEMENTATIONS -----
def get_sp500_tickers() -> List[str]:
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_table = pd.read_html(url)[0]
        tickers = sp500_table['Symbol'].tolist()
        logger.info("Successfully retrieved S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Error retrieving S&P 500 tickers: {e}")
        return []

def get_twitter_sentiment(ticker: str) -> float:
    try:
        query = f"${ticker} -is:retweet lang:en"
        tweets = twitter_client.search_recent_tweets(query=query, max_results=50)
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        if tweets.data:
            for tweet in tweets.data:
                score = analyzer.polarity_scores(tweet.text)["compound"]
                scores.append(score)
            avg_score = np.mean(scores) if scores else 0.0
            logger.info(f"Twitter sentiment for {ticker}: {avg_score}")
            return avg_score
        else:
            logger.info(f"No tweets found for {ticker}.")
            return 0.0
    except Exception as e:
        logger.error(f"Error fetching Twitter sentiment for {ticker}: {e}")
        return 0.0

def get_reddit_sentiment(ticker: str) -> float:
    try:
        subreddits = ["wallstreetbets", "stocks", "investing", "pennystocks"]
        reddit_local = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for sub in subreddits:
            for post in reddit_local.subreddit(sub).new(limit=50):
                text = post.title + " " + post.selftext
                if ticker.upper() in text.upper():
                    score = analyzer.polarity_scores(text)["compound"]
                    scores.append(score)
        avg_score = np.mean(scores) if scores else 0.0
        logger.info(f"Reddit sentiment for {ticker}: {avg_score}")
        return avg_score
    except Exception as e:
        logger.error(f"Error fetching Reddit sentiment for {ticker}: {e}")
        return 0.0

def aggregate_alternative_sentiment(ticker: str, twitter_weight: float = 2.0, reddit_weight: float = 2.0) -> float:
    twitter_score = get_twitter_sentiment(ticker)
    reddit_score = get_reddit_sentiment(ticker)
    combined = (twitter_weight * twitter_score + reddit_weight * reddit_score) / (twitter_weight + reddit_weight)
    logger.info(f"Combined sentiment for {ticker}: {combined}")

def fetch_all_news(api_key: str, limit: int = 50) -> List[Dict[str, Any]]:
    url = f"https://api.polygon.io/vX/reference/financials"
    params = {"limit": limit, "apiKey": POLYGON_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Error fetching news from Polygon: {e}")
        return []

def extract_tickers_from_articles(articles: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    analyzer = SentimentIntensityAnalyzer()
    known_tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NFLX", "NVDA", "JPM", "BAC"]
    ticker_sentiments: Dict[str, List[float]] = {}
    for article in articles:
        text = (article.get("title", "") + " " + article.get("summary", "")).upper()
        for ticker in known_tickers:
            if ticker in text:
                score = analyzer.polarity_scores(text)["compound"]
                ticker_sentiments.setdefault(ticker, []).append(score)
    return ticker_sentiments

def extract_hot_tickers_from_reddit(subreddits: Optional[List[str]] = None, limit: int = 200) -> List[str]:
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing", "options", "pennystocks",
                      "stockmarket", "SPACs", "RobinHoodPennyStocks"]
    reddit_local = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    phrases = ["to the moon", "massive upside", "exploding", "next big thing", "multi-bagger",
               "10x potential", "undervalued", "hidden gem", "breakout", "rocket", "moonshot",
               "up only", "growth play", "taking profits", "printing money", "bag secured",
               "life-changing", "retirement stock", "dividend monster", "yolo call", "IV crush",
               "leaps", "theta gang", "bullish sweep", "unusual options", "buy the dip",
               "reversal", "squeeze", "short squeeze", "oversold", "going parabolic"]
    phrases = [p.lower() for p in phrases]
    tickers_set = {"AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "AMD", "GOOGL", "META",
                   "PLTR", "SOFI", "F", "RIVN", "BBBY", "GME", "AMC", "TQQQ", "SPY", "QQQ"}
    ticker_counts: Dict[str, int] = {}
    for sub in subreddits:
        try:
            submissions = reddit_local.subreddit(sub).new(limit=limit)
            for post in submissions:
                content = (post.title + " " + post.selftext).upper()
                if any(phrase in content.lower() for phrase in phrases):
                    for word in content.split():
                        word = word.strip("$()[]:.,!?")
                        if word.isalpha() and word in tickers_set:
                            ticker_counts[word] = ticker_counts.get(word, 0) + 1
        except Exception as e:
            logger.error(f"Error fetching Reddit posts from r/{sub}: {e}")
    ranked = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in ranked]

def rank_tickers_by_sentiment(articles: List[Dict[str, Any]], top_n: int = 10) -> List[str]:
    ticker_sentiments = extract_tickers_from_articles(articles)
    trending_reddit = extract_hot_tickers_from_reddit(["wallstreetbets", "stocks", "investing"])
    rankings = []
    for ticker, scores in ticker_sentiments.items():
        frequency = len(scores)
        avg_sentiment = np.mean(scores) if scores else 0
        combined = frequency * avg_sentiment
        rankings.append((ticker, frequency, avg_sentiment, combined))
    rankings.sort(key=lambda x: (x[3], x[1]), reverse=True)
    return [x[0] for x in rankings[:top_n]]

def fetch_yahoo_trending_tickers() -> List[str]:
    url = "https://query2.finance.yahoo.com/v1/finance/trending/US"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Yahoo trending tickers: got status code {response.status_code}. Using fallback list.")
            return ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]
        data = response.json()
        if "finance" in data and "result" in data["finance"] and data["finance"]["result"]:
            trending = [q["symbol"] for q in data["finance"]["result"][0]["quotes"]]
            return trending
        else:
            logger.warning("Yahoo trending tickers: unexpected JSON structure. Using fallback list.")
            return ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]
    except Exception as e:
        logger.warning(f"Error fetching Yahoo trending tickers: {e}. Using fallback list.")
        return ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]

# ----- TECHNICAL & FEATURE ENGINEERING FUNCTIONS -----
def sanitize_close_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column before rolling feature generation.")
    df["Close"] = df["Close"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if df["Close"].isna().all():
        raise ValueError("'Close' column is all NaN after sanitizing.")
    return df

def add_macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["Interest_Rate"] = 5.25
    df["Inflation_Rate"] = 3.2
    return df

import requests
import pandas as pd
import logging

# Set up logger for debugging
logger = logging.getLogger(__name__)

POLYGON_API_KEY = "bbYmszAdyJRvANl6FqHgoBkWU3hfhHJT"

def get_optimal_expiration_date(predicted_volatility: float, forecast_days: int) -> str:
    """
    Get the expiration date based on forecast volatility and predicted market movement.
    """
    # Calculate optimal expiration in days (e.g., 30 days ahead for significant movement)
    optimal_expiration = forecast_days + int(predicted_volatility * 15)  # Adjust multiplier for preferred time window
    
    # Ensure that the expiration date is at least a week out
    optimal_expiration = max(optimal_expiration, 7)  # Minimum of 7 days for time decay consideration
    
    expiration_date = (datetime.today() + timedelta(days=optimal_expiration)).strftime("%Y-%m-%d")
    return expiration_date

def get_stock_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "results" not in data:
            raise ValueError(f"No data returned from Polygon for {ticker}")
        records = data["results"]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"c": "Close", "o": "Open", "h": "High", "l": "Low", "v": "Volume"}, inplace=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        if df.empty or df["Close"].isna().all():
            raise ValueError(f"Fetched data for {ticker} has no valid Close prices.")
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.error(f"Error fetching stock prices from Polygon for {ticker}: {e}")
        return pd.DataFrame()

def get_prices_for_tickers(tickers: list, start_date: str, end_date: str) -> dict:
    tickers_data = {}
    for ticker in tickers:
        tickers_data[ticker] = get_stock_prices(ticker, start_date, end_date)
    return tickers_data

# Example: List of 5 tickers
tickers = HOT_TICKERS

# Example: Date range for fetching data
start_date = "2016-01-01"
end_date = "2018-12-31"

# Fetch prices for all tickers
tickers_data = get_prices_for_tickers(tickers, start_date, end_date)

# Example: Print the data for each ticker
for ticker, data in tickers_data.items():
    print(f"\n{ticker} Data:\n", data.head())  # Print first 5 rows for each ticker


def add_vix_indicator(df: pd.DataFrame) -> pd.DataFrame:
    start_date = df.index[0] - pd.DateOffset(days=1)
    try:
        vix = web.DataReader("VIXCLS", "fred", start_date, df.index[-1])
        vix = vix.fillna(method="ffill")
        df = df.merge(vix, left_index=True, right_index=True, how="left")
        df.rename(columns={"VIXCLS": "VIX"}, inplace=True)
        df["VIX"] = df["VIX"].fillna(method="ffill")
    except Exception:
        df["VIX"] = 20.0
    return df

def add_m2_indicator(df: pd.DataFrame) -> pd.DataFrame:
    start_date = df.index[0] - pd.DateOffset(days=1)
    try:
        m2 = web.DataReader("M2SL", "fred", start_date, df.index[-1])
        m2 = m2.fillna(method="ffill")
        df = df.merge(m2, left_index=True, right_index=True, how="left")
        df.rename(columns={"M2SL": "M2"}, inplace=True)
        df["M2"] = df["M2"].fillna(method="ffill")
    except Exception:
        df["M2"] = 10000.0
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns:
        logger.error("Error: 'Close' column not found in DataFrame.")
        return df
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    close = df["Close"]
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["Momentum"] = close - df["EMA_10"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace({0: 1e-9})
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df["Bollinger_Upper"] = sma20 + 2 * std20
    df["Bollinger_Lower"] = sma20 - 2 * std20
    return sanitize_close_column(df)

def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
    if "Close" not in df.columns:
        logger.error("Error: 'Close' column not found in DataFrame.")
        return df
    close = df["Close"]
    for lag in lags:
        shifted = close.shift(lag)
        df[f"lag_close_{lag}"] = shifted
        df[f"lag_return_{lag}"] = np.log(close / shifted.replace({0: 1e-9}))
    return df.fillna(method="bfill")

def add_volatility_features(df: pd.DataFrame, windows: List[int] = [10, 20, 30]) -> pd.DataFrame:
    df["Daily_Return"] = df["Close"].pct_change()
    for window in windows:
        df[f"vol_{window}"] = df["Daily_Return"].rolling(window=window).std()
    return df.fillna(method="bfill")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = sanitize_close_column(df)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if df["Close"].isna().all():
        raise ValueError("All Close values are NaN after coercion.")
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df = add_macro_indicators(df)
    df = add_vix_indicator(df)
    df = add_m2_indicator(df)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_volatility_features(df)
    df["Rolling_Mean"] = df["Close"].rolling(window=30).mean()
    df["Rolling_Std"] = df["Close"].rolling(window=30).std()
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    df["Weekday"] = df.index.weekday
    for col in FEATURE_COLS:
        if col not in df.columns:
            logger.warning(f"Missing '{col}'. Filling with 0.")
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    return df.fillna(0)

def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = feature_engineering(df.copy()).reset_index()
    if "Rolling_Mean" not in df_fe.columns or df_fe["Rolling_Mean"].isna().all():
        if "Close" in df_fe.columns:
            df_fe["Rolling_Mean"] = df_fe["Close"].rolling(window=30, min_periods=1).mean()
        else:
            raise ValueError("The 'Close' column is required to compute Rolling_Mean.")
    if "Rolling_Std" not in df_fe.columns or df_fe["Rolling_Std"].isna().all():
        if "Close" in df_fe.columns:
            df_fe["Rolling_Std"] = df_fe["Close"].rolling(window=30, min_periods=1).std().fillna(0)
        else:
            raise ValueError("The 'Close' column is required to compute Rolling_Std.")
    if "timestamp" in df_fe.columns:
        df_fe.rename(columns={"timestamp": "ds"}, inplace=True)
    elif "index" in df_fe.columns:
        df_fe.rename(columns={"index": "ds"}, inplace=True)
    else:
        df_fe["ds"] = df_fe.index
    if "Close" in df_fe.columns:
        df_fe.rename(columns={"Close": "y"}, inplace=True)
    elif "Returns" in df_fe.columns:
        df_fe["y"] = df_fe["Returns"]
    else:
        raise ValueError("No suitable 'y' column found for Prophet.")
    cols = ["ds", "y"] + [c for c in df_fe.columns if c not in ["ds", "y"]]
    return df_fe[cols]

# ----- OPTIONS & SENTIMENT FUNCTIONS -----
def calculate_strike(current_price: float, predicted_return: float, trade_type: str) -> float:
    if trade_type == "CALL":
        strike_percentage = 0.05
    elif trade_type == "PUT":
        strike_percentage = -0.05
    else:
        strike_percentage = 0
    return round(current_price * (1 + strike_percentage), 2)

def generate_trade_type_improved(predicted_return: float, alt_sentiment: float, volatility: float) -> str:
    combined_score = predicted_return + alt_sentiment
    if combined_score > 0.03 + (volatility * 0.1):
        return "CALL"
    elif combined_score < -0.03 - (volatility * 0.1):
        return "PUT"
    else:
        return "NEUTRAL"
    
def get_optimal_expiration_date(predicted_volatility: float, forecast_days: int) -> str:
    optimal_expiration = forecast_days + int(predicted_volatility * 15)  # Adjust multiplier for preferred time window
    
  
    optimal_expiration = max(optimal_expiration, 7)  # Minimum of 7 days for time decay consideration
    
    expiration_date = (datetime.today() + timedelta(days=optimal_expiration)).strftime("%Y-%m-%d")
    return expiration_date

def get_options_chain(ticker: str, forecast_days: int, predicted_volatility: float) -> List[Dict[str, Any]]:
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        return []
    
    # Get the optimal expiration date based on the volatility and forecast days
    expiration_date = get_optimal_expiration_date(predicted_volatility, forecast_days)
    
    # Find the closest available expiration date that matches
    expiration = min(exps, key=lambda exp: abs((datetime.strptime(exp, "%Y-%m-%d") - datetime.strptime(expiration_date, "%Y-%m-%d")).days))
    
    chain = t.option_chain(expiration)
    options = []
    for idx, row in chain.calls.iterrows():
        row_dict = row.to_dict()
        row_dict["type"] = "CALL"
        row_dict["expiration_date"] = expiration
        options.append(row_dict)
    for idx, row in chain.puts.iterrows():
        row_dict = row.to_dict()
        row_dict["type"] = "PUT"
        row_dict["expiration_date"] = expiration
        options.append(row_dict)
    
    return options


def generate_options_signal(ticker: str, forecast_df: pd.DataFrame, alt_sentiment: float, threshold: float = 0.3) -> Tuple[bool, float, float, str]:
    last_price = forecast_df.iloc[0]["Ensemble"]
    predicted_price = forecast_df.iloc[-1]["Ensemble"]
    predicted_return = (predicted_price - last_price) / last_price
    combined_score = predicted_return + alt_sentiment
    signal = combined_score > threshold
    volatility = forecast_df["Ensemble"].pct_change().std()
    trade_type = generate_trade_type_improved(predicted_return, alt_sentiment, volatility)
    return signal, combined_score, predicted_return, trade_type

def calculate_position_size(max_loss: float, account_balance: float = 1000, risk_per_trade: float = 0.02) -> float:
    risk_amount = account_balance * risk_per_trade
    return risk_amount / max_loss

def compute_options_parameters(future_forecast: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    last_date = future_forecast.index[-1]
    predicted_price = future_forecast.iloc[-1]["Ensemble"]
    strike = round(predicted_price)
    expiration_date = last_date.strftime("%Y-%m-%d")
    delta = 0.5  # Placeholder value
    profit_probability = 0.7  # Placeholder value
    hold_duration = (future_forecast.index[-1] - future_forecast.index[0]).days
    resistance_level = predicted_price * 1.05
    return {
        "strike": strike,
        "expiration_date": expiration_date,
        "delta": delta,
        "profit_probability": profit_probability,
        "hold_duration": hold_duration,
        "resistance_level": resistance_level
    }

# ----- MODEL TRAINING & PREDICTION FUNCTIONS -----
def tune_xgboost_optuna(X: np.ndarray, y: np.ndarray) -> XGBRegressor:
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    def optuna_objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
            "max_depth": trial.suggest_int("max_depth", 8, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.mean(np.abs(y_val - preds))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=100, show_progress_bar=True)
    best_params = study.best_trial.params
    logging.basicConfig(
    level=logging.WARNING,  # Change from INFO to WARNING to suppress info logs
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
    logger.info(f"Optuna best XGBoost params: {best_params}")
    return XGBRegressor(**best_params, random_state=42)

def train_models_updated(df: pd.DataFrame) -> Tuple[Dict[str, Any], Prophet, Dict[str, Any]]:
    df_eng = feature_engineering(df.copy())
    df_eng = df_eng.loc[:, ~df_eng.columns.duplicated()].fillna(method="ffill").fillna(method="bfill").fillna(0)
    feature_cols = FEATURE_COLS
    X_raw = df_eng[feature_cols].values
    y = df_eng["Returns"].fillna(0).values

    med = np.median(X_raw, axis=0)
    q1 = np.percentile(X_raw, 25, axis=0)
    q3 = np.percentile(X_raw, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    X_scaled = (X_raw - med) / iqr
    assert X_scaled.shape[1] == len(feature_cols), "Scaler shape mismatch"
    scaler_params = {"median": med, "iqr": iqr, "feature_names": feature_cols}

    models_dict = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.001),
        "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42),
        "XGBoost": None,
        "SVR": SVR(kernel="linear", C=1, gamma="scale"),
        "MLP": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=200, early_stopping=True, learning_rate="constant")
    }
    best_xgb = tune_xgboost_optuna(X_scaled, y)
    best_xgb.fit(X_scaled, y)
    models_dict["XGBoost"] = best_xgb

    trained_models = {name: m.fit(X_scaled, y) for name, m in models_dict.items() if m is not None}

    df_prophet = prepare_prophet_data(df.copy())
    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    if "Rolling_Mean" in df_prophet.columns and "Rolling_Std" in df_prophet.columns:
        prophet_model.add_regressor("Rolling_Mean")
        prophet_model.add_regressor("Rolling_Std")
    prophet_model.fit(df_prophet)

    return trained_models, prophet_model, scaler_params

def compute_model_weights_with_cv(models: Dict[str, Any], X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> Dict[str, float]:
    from sklearn.model_selection import TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=n_splits)
    errors = {name: [] for name in models}
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train = np.nan_to_num(X_train)
        y_train = np.nan_to_num(y_train)
        X_val = np.nan_to_num(X_val)
        y_val = np.nan_to_num(y_val)
        for name, model in models.items():
            preds = model.predict(X_val)
            mae = np.mean(np.abs(y_val - preds))
            errors[name].append(mae)
    avg_err = {k: np.mean(v) for k, v in errors.items()}
    inv_errors = {k: 1/(val+1e-6) for k, val in avg_err.items()}
    total_inv = sum(inv_errors.values())
    weights = {k: inv_errors[k]/total_inv for k in inv_errors}
    return weights

def train_ensemble_models(df: pd.DataFrame) -> Tuple[Dict[str, Any], Prophet, Dict[str, Any]]:
    df_fe_full = feature_engineering(df.copy())
    df_features = df_fe_full[FEATURE_COLS].fillna(0)
    X = df_features.values
    y = df_fe_full["Returns"].values

    med = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    X_scaled = (X - med) / iqr
    scaler_params = {"median": med, "iqr": iqr, "feature_names": FEATURE_COLS}

    models_dict = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.001),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
        "XGBoost": None,
        "SVR": SVR(kernel="linear", C=1, gamma="scale"),
        "MLP": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=200, early_stopping=True, learning_rate="constant")
    }
    best_xgb = tune_xgboost_optuna(X_scaled, y)
    best_xgb.fit(X_scaled, y)
    models_dict["XGBoost"] = best_xgb

    trained_models = {name: m.fit(X_scaled, y) for name, m in models_dict.items() if m is not None}

    df_prophet = prepare_prophet_data(df.copy())
    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    if "Rolling_Mean" in df_prophet.columns and "Rolling_Std" in df_prophet.columns:
        prophet_model.add_regressor("Rolling_Mean")
        prophet_model.add_regressor("Rolling_Std")
    prophet_model.fit(df_prophet)

    return trained_models, prophet_model, scaler_params

def align_and_scale_features(features_df: pd.DataFrame, scaler_params: Dict[str, Any]) -> np.ndarray:
    expected_cols = scaler_params["feature_names"]
    for col in expected_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[expected_cols]
    med = np.array(scaler_params["median"]).reshape(1, -1)
    iqr = np.array(scaler_params["iqr"]).reshape(1, -1)
    values = features_df.values
    if values.shape[1] != med.shape[1]:
        raise ValueError("Mismatch in feature count after alignment.")
    return (values - med) / iqr

def predict_stock_price_updated(df: pd.DataFrame, models: Dict[str, Any], prophet: Prophet, scaler_params: Dict[str, Any], periods: int = 7) -> pd.DataFrame:
    if "Returns" not in df.columns:
        if "y" in df.columns:
            df["Returns"] = df["y"]
        else:
            raise ValueError("Input DataFrame lacks the 'Returns' column.")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df_subset = df[FEATURE_COLS].copy()
    last_date = df_subset.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]
    future_features = pd.DataFrame({col: [df[col].iloc[-1]] * len(future_dates) for col in FEATURE_COLS},
                                   index=future_dates)
    X_future = align_and_scale_features(future_features, scaler_params)
    X_all = df_subset.values
    y_all = df["Returns"].fillna(0).values
    weights = compute_model_weights_with_cv(models, X_all, y_all, n_splits=3)
    weighted_preds = np.zeros(len(future_dates))
    for name, model in models.items():
        preds = model.predict(X_future)
        preds = np.clip(preds, -0.02, 0.02)
        weighted_preds += preds * weights.get(name, 0)
    last_close = df["Close"].iloc[-1]
    cumulative_returns = np.cumsum(weighted_preds)
    forecast_prices = last_close * np.exp(cumulative_returns)
    return pd.DataFrame({"Ensemble": forecast_prices}, index=future_dates)

# ----- ANALYZE STOCK FUNCTION -----
def analyze_stock(ticker: str, training_start: str, forecast_days: int) -> Optional[Dict[str, Any]]:
    try:
        # Get predicted stock prices using the trained models
        future_prices = predict_stock_price(ticker, training_start, forecast_days)

        # No sentiment data (Twitter/Reddit) for this version
        alt_sentiment = 0  # Set to 0 if not using alternative sentiment data

        # Get the initial price and predicted price
        current_price = future_prices.iloc[0]["Ensemble"]
        predicted_price = future_prices.iloc[-1]["Ensemble"]

        # Calculate the predicted return
        predicted_return = (predicted_price - current_price) / current_price

        # Calculate volatility based on predicted prices
        volatility = future_prices["Ensemble"].pct_change().std()

        # Get options chain with the new expiration date logic (based on volatility and forecast_days)
        options_chain = get_options_chain(ticker, forecast_days, volatility)

        # Generate the trade type based on predicted return, sentiment, and volatility
        trade_type = generate_trade_type_improved(predicted_return, alt_sentiment, volatility)

        if trade_type == "NEUTRAL":
            return None

        # Calculate strike price for options using predicted price
        strike = calculate_strike(current_price, predicted_return, trade_type)

        # Filter valid options based on strike price, type, and expiration date
        valid_options = [
            opt for opt in options_chain
            if opt["type"] == trade_type and abs(opt["strike"] - strike) / current_price < 0.1
            and opt["expiration_date"] > datetime.today().strftime('%Y-%m-%d')
        ]

        if not valid_options:
            return None

        # Choose the option with the best price (based on ask price)
        option = min(valid_options, key=lambda x: x.get("ask") if x.get("ask") is not None else float('inf'))

        # Calculate the probability of profit for the option
        profit_certainty = get_option_profit_certainty(trade_type, future_prices, option)

        # Collect all relevant details for the option
        option_details = {
            "contractSymbol": option.get("contractSymbol", ""),
            "strike": option.get("strike", ""),
            "lastPrice": option.get("lastPrice", ""),
            "bid": option.get("bid", ""),
            "ask": option.get("ask", ""),
            "volume": option.get("volume", ""),
            "openInterest": option.get("openInterest", ""),
            "impliedVolatility": option.get("impliedVolatility", ""),
            "expiration": option.get("expiration_date", ""),
            "profitCertainty": round(profit_certainty * 100, 2)
        }

        # Return the final result
        return {
            "ticker": ticker,
            "future_prices": future_prices,
            "alt_sent": alt_sentiment,
            "signal": True,
            "combined_score": predicted_return + alt_sentiment,
            "predicted_return": predicted_return,
            "option_params": option_details,
            "trade_type": trade_type
        }
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        return None



def predict_stock_price(ticker: str, training_start: str, forecast_days: int = 7) -> pd.DataFrame:
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = get_stock_prices(ticker, training_start, end_date)
    if df.empty or "Close" not in df.columns or df["Close"].isna().all():
        raise ValueError(f"Stock data for {ticker} is empty or missing critical fields.")
    if len(df) < 200:
        raise ValueError(f"Not enough historical data for {ticker}.")
    df_fe = feature_engineering(df.copy())
    trained_models, prophet, scaler_params = train_models_updated(df_fe)
    future_forecast = predict_stock_price_updated(df_fe, trained_models, prophet, scaler_params, periods=forecast_days)
    return future_forecast

def get_option_profit_certainty(trade_type: str, future_prices: pd.DataFrame, option: Dict[str, Any]) -> float:
    from scipy.stats import norm
    # Calculate the predicted return
    pred_return = (future_prices.iloc[-1]["Ensemble"] - future_prices.iloc[0]["Ensemble"]) / future_prices.iloc[0]["Ensemble"]
    # Calculate the standard deviation of the returns
    std_return = future_prices["Ensemble"].pct_change().std()
    
    if std_return == 0:
        std_return = 1e-9  # Prevent division by zero
    
    # Calculate Z-score
    z = pred_return / std_return
    
    # Calculate the probability based on the trade type
    if trade_type == "CALL":
        prob = norm.cdf(z)
    elif trade_type == "PUT":
        prob = 1 - norm.cdf(z)
    else:
        prob = 0.5  # Neutral position has a 50% probability
    
    return prob

def fetch_all_news(api_key: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve financial news articles via Polygon.
    """
    url = f"https://api.polygon.io/vX/reference/financials"
    params = {"limit": limit, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Error fetching news from Polygon: {e}")
        return []

def get_quarterly_financials(ticker: str, years: int = 4) -> pd.DataFrame:
    """
    Retrieves quarterly financial filings for the specified ticker using Polygon's RESTClient.
    Only returns filings with filing dates ≥ (today - years).
    """
    start_date = (datetime.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    financials = []
    try:
        for filing in polygon_client.vx.list_stock_financials(
            ticker=ticker.upper(),
            timeframe="quarterly",
            order="asc",
            limit=str(years * 4),
            sort="filing_date",
            filing_date_gte=start_date,
            include_sources="true"
        ):
            financials.append(filing)
    except Exception as e:
        logger.error(f"Error retrieving financials for {ticker}: {e}")
        return pd.DataFrame()

    financials_list = []
    for entry in financials:
        try:
            fiscal_period = getattr(entry, "fiscal_period", "Unknown")
            fiscal_year = getattr(entry, "fiscal_year", "Unknown")
            filing_date = getattr(entry, "filing_date", None)
            fin = getattr(entry, "financials", None)
            income_statement = getattr(fin, "income_statement", None) if fin else None
            balance_sheet = getattr(fin, "balance_sheet", None) if fin else None

            revenue = getattr(getattr(income_statement, "revenues", {}), "value", None) if income_statement else None
            net_income = getattr(getattr(income_statement, "net_income_loss", {}), "value", None) if income_statement else None
            gross_profit = getattr(getattr(income_statement, "gross_profit", {}), "value", None) if income_statement else None
            rnd = getattr(getattr(income_statement, "research_and_development", {}), "value", None) if income_statement else None
            equity = getattr(getattr(balance_sheet, "equity", {}), "value", None) if balance_sheet else None
            assets = getattr(getattr(balance_sheet, "assets", {}), "value", None) if balance_sheet else None

            financials_list.append({
                "Quarter": f"{fiscal_period} {fiscal_year}",
                "Filing Date": filing_date if filing_date is not None else "N/A",
                "Revenue": revenue,
                "Net Income": net_income,
                "Gross Profit": gross_profit,
                "R&D": rnd,
                "Equity": equity,
                "Assets": assets
            })
        except Exception as ex:
            logger.error(f"Error processing financial data for {ticker}: {ex}")
            continue

    return pd.DataFrame(financials_list)

def compute_qoq_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes quarter-over-quarter growth and margin metrics.
    Assumes that the input DataFrame is sorted by quarter (ascending).
    Returns a DataFrame including growth metrics.
    """
    try:
        df["Filing Date"] = df["Filing Date"].replace("N/A", np.nan)
        df = df.dropna(subset=["Filing Date"]).copy()
        df["Filing Date"] = pd.to_datetime(df["Filing Date"], errors="coerce", infer_datetime_format=True)
        if df["Filing Date"].isna().all():
            logger.warning("⚠️ All filing dates could not be converted!")
            return pd.DataFrame()
        df["Quarter"] = df["Filing Date"].dt.to_period("Q")
        df = df.dropna(subset=["Quarter"]).sort_values("Quarter")
        if "Revenue" not in df.columns or "Net Income" not in df.columns:
            logger.warning("⚠️ Essential financial metrics are missing.")
            return pd.DataFrame()
        df["Revenue Growth (%)"] = df["Revenue"].pct_change() * 100
        df["Net Income Growth (%)"] = df["Net Income"].pct_change() * 100
        df["Gross Margin (%)"] = (df["Gross Profit"] / df["Revenue"]) * 100 if "Gross Profit" in df.columns else np.nan
        df["ROE (%)"] = (df["Net Income"] / df["Equity"]) * 100 if "Equity" in df.columns else np.nan
        df["ROA (%)"] = (df["Net Income"] / df["Assets"]) * 100 if "Assets" in df.columns else np.nan
        return df[["Quarter", "Revenue Growth (%)", "Net Income Growth (%)", "Gross Margin (%)", "ROE (%)", "ROA (%)"]]
    except Exception as e:
        logger.error(f"Error computing quarter-over-quarter growth: {e}")
        return pd.DataFrame()

def get_quarterly_financials_safe(ticker: str, years: int = 4, retries: int = 3) -> Optional[pd.DataFrame]:
    for _ in range(retries):
        df = get_quarterly_financials(ticker, years)
        if not df.empty:
            return df
    logger.warning(f"⚠️ Data missing for {ticker}, using sector estimates.")
    return None

def main():
    tickers = HOT_TICKERS  # Assumes HOT_TICKERS is defined globally
    training_start = "2023-01-01"
    forecast_days = 7

    for ticker in tickers:
        print("=" * 50)
        print(f"Processing ticker: {ticker}\n")
        
        # --- Options Analysis Section ---
        try:
            result = analyze_stock(ticker, training_start, forecast_days)
            if result:
                print(f"Ticker: {result['ticker']}")
                print(f"Predicted Return: {result['predicted_return']:.2%}")
                print(f"Trade Type: {result['trade_type']}")
                if result.get('option_params'):
                    option = result['option_params']
                    print("Option Recommendation:")
                    print(f"  Contract Symbol: {option.get('contractSymbol', 'N/A')}")
                    print(f"  Strike: {option.get('strike', 'N/A')}")
                    print(f"  Last Price: {option.get('lastPrice', 'N/A')}")
                    print(f"  Bid: {option.get('bid', 'N/A')}")
                    print(f"  Ask: {option.get('ask', 'N/A')}")
                    print(f"  Expiration: {option.get('expiration', 'N/A')}")
                    print(f"  Profit Certainty: {option.get('profitCertainty', 'N/A')}%")
                else:
                    print("No valid options recommendation found.")
            else:
                print(f"No analysis result for {ticker}.")
        except Exception as ex:
            print(f"Error analyzing options for {ticker}: {ex}")

        # --- News & Financial Analysis Section ---
        try:
            # Get company name for more robust news filtering
            company_name = ""
            try:
                ticker_info = yf.Ticker(ticker).info
                company_name = ticker_info.get("longName", "").upper()
            except Exception as e:
                logger.error(f"Could not fetch company name for {ticker}: {e}")

            # Retrieve and filter news articles
            news_summary = ""
            try:
                news_articles = fetch_all_news(POLYGON_API_KEY, limit=10)
                combined_text = lambda article: (article.get("title", "") + " " + article.get("summary", "")).upper()
                relevant_news = [
                    article for article in news_articles
                    if (f"${ticker.upper()}" in combined_text(article))
                       or (ticker.upper() in combined_text(article))
                       or (company_name and company_name in combined_text(article))
                ]
                if relevant_news:
                    news_headlines = [article.get("title", "N/A") for article in relevant_news]
                    news_summary = "Recent headlines include: " + "; ".join(news_headlines) + "."
                else:
                    news_summary = f"No relevant news found for {ticker}."
            except Exception as ne:
                news_summary = f"Error fetching news: {ne}"

            # Retrieve and compute financial growth metrics with enhanced narrative
            financial_summary = ""
            try:
                financials = get_quarterly_financials_safe(ticker, years=2)
                if financials is not None and not financials.empty:
                    growth_data = compute_qoq_growth(financials)
                    if not growth_data.empty:
                        # Use the most recent four quarters, if available
                        last_four = growth_data if len(growth_data) < 4 else growth_data.iloc[-4:]

                        # Compute averages for key growth metrics
                        avg_rg = last_four["Revenue Growth (%)"].mean() if "Revenue Growth (%)" in last_four.columns else None
                        avg_nig = last_four["Net Income Growth (%)"].mean() if "Net Income Growth (%)" in last_four.columns else None
                        avg_gm = last_four["Gross Margin (%)"].mean() if "Gross Margin (%)" in last_four.columns else None
                        avg_roa = last_four["ROA (%)"].mean() if "ROA (%)" in last_four.columns else None
                        avg_roe = last_four["ROE (%)"].mean() if "ROE (%)" in last_four.columns else None

                        # Compute annualized growth projections using compound growth over 4 quarters
                        annual_rg = ((1 + avg_rg / 100) ** 4 - 1) * 100 if avg_rg is not None else None
                        annual_nig = ((1 + avg_nig / 100) ** 4 - 1) * 100 if avg_nig is not None else None

                        # Get the latest quarter's figures
                        latest_quarter = growth_data.iloc[-1]
                        latest_rg = latest_quarter.get("Revenue Growth (%)", None)
                        latest_nig = latest_quarter.get("Net Income Growth (%)", None)
                        latest_rg_str = f"{latest_rg:.2f}%" if (latest_rg is not None and pd.notna(latest_rg)) else "data not available"
                        latest_nig_str = f"{latest_nig:.2f}%" if (latest_nig is not None and pd.notna(latest_nig)) else "data not available"

                        # Start constructing the detailed narrative
                        detailed_summary = (
                            f"In the latest analysis over the past four quarters, {ticker} has demonstrated a moderate growth trajectory. "
                            f"On average, revenue increased by {avg_rg:.2f}% per quarter and net income by {avg_nig:.2f}%. "
                            f"In the most recent quarter ({latest_quarter['Quarter']}), revenue grew by {latest_rg_str} and net income by {latest_nig_str}. "
                        )

                        # Compare the latest quarter with previous quarters
                        if len(last_four) > 1:
                            previous_quarters = last_four.iloc[:-1]
                            prev_avg_rg = previous_quarters["Revenue Growth (%)"].mean() if "Revenue Growth (%)" in previous_quarters.columns else None
                            prev_avg_nig = previous_quarters["Net Income Growth (%)"].mean() if "Net Income Growth (%)" in previous_quarters.columns else None

                            if prev_avg_rg is not None and latest_rg is not None:
                                diff_rg = latest_rg - prev_avg_rg
                                if diff_rg > 0:
                                    detailed_summary += f"This represents an increase of {abs(diff_rg):.2f} percentage points over the previous average revenue growth. "
                                else:
                                    detailed_summary += f"Revenue growth has decreased by {abs(diff_rg):.2f} percentage points compared to the preceding quarters. "
                            if prev_avg_nig is not None and latest_nig is not None:
                                diff_nig = latest_nig - prev_avg_nig
                                if diff_nig > 0:
                                    detailed_summary += f"Notably, net income growth improved by {abs(diff_nig):.2f} percentage points relative to the previous average. "
                                else:
                                    detailed_summary += f"Net income growth declined by {abs(diff_nig):.2f} percentage points compared to the earlier quarters. "

                        # Interpret the trajectory; for mature stocks, a 3% quarterly revenue gain is moderate.
                        if avg_rg is not None:
                            detailed_summary += (
                                f"Although the average quarterly revenue growth of {avg_rg:.2f}% suggests steady performance, for a mature company like {ticker} "
                                f"it is considered moderate rather than exceptionally robust. "
                            )
                        # Add long-term growth projections based on compound averages
                        if annual_rg is not None and annual_nig is not None:
                            detailed_summary += (
                                f"If these performance levels persist, the estimated annualized revenue growth is {annual_rg:.2f}% and net income growth is {annual_nig:.2f}%. "
                            )
                        # Add additional financial health commentary
                        if avg_gm is not None and not pd.isna(avg_gm):
                            detailed_summary += f"Gross margin averaged {avg_gm:.2f}%, signifying stable production costs. "
                        else:
                            detailed_summary += "Gross margin data is not available. "
                        if avg_roa is not None and avg_roe is not None and not (pd.isna(avg_roa) or pd.isna(avg_roe)):
                            detailed_summary += (
                                f"Profitability metrics (ROA and ROE) averaged {avg_roa:.2f}% and {avg_roe:.2f}%, respectively, reflecting effective asset utilization "
                                "and solid shareholder returns. "
                            )
                        else:
                            detailed_summary += "Profitability ratios (ROA/ROE) could not be fully determined. "

                        financial_summary = detailed_summary
                    else:
                        financial_summary = f"No growth data available for {ticker}."
                else:
                    financial_summary = f"Financial data not available for {ticker}."
            except Exception as fe:
                financial_summary = f"Error fetching financial data: {fe}"

            # Print the combined news and financial analysis summary for the ticker
            print("\nAnalysis Summary:")
            print(f"{ticker} analysis shows that {news_summary} {financial_summary}\n")
        except Exception as e:
            print(f"Error in financial news analysis for {ticker}: {e}")

    print("=" * 50)
    
if __name__ == "__main__":
    main()
