
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


Required packages (install in Colab using pip):

Also, run the following once to download required NLTK data:

"""
# Upgrade numpy to a compatible version

# First, uninstall the current numpy version


# Uninstall the current numpy version



# Installations
!pip uninstall -y numpy pandas tensorflow
!pip install numpy==1.26.0 pandas==2.2.2
!pip install tensorflow==2.18.0
!pip install yfinance prophet scikit-learn xgboost optuna matplotlib seaborn nltk tweepy praw pyngrok altair blinker cachetools click protobuf pyarrow tenacity toml watchdog gitpython pandas_datareader

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

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"TensorFlow version: {tf.__version__}")
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


# For sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# For Twitter API using Tweepy (v4)
import tweepy
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN")
twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

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
    "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "GOOG", "META", "NVDA", "NFLX", "ADBE",
    "INTC", "AMD", "CRM", "PYPL", "QCOM", "AVGO", "TXN", "CSCO", "ORCL", "IBM",
    "NOW", "SHOP", "SQ", "ZM", "DOCU", "UBER", "LYFT", "SNOW", "CRWD", "NET",
    "FSLY", "PLTR", "SPOT", "RBLX", "WORK", "VRTX", "BIIB", "GME", "AMC", "BBBY",
    "SBUX", "DIS", "XOM", "CVX", "COP", "BP", "SLB", "LNG", "DUK", "NEE", "D", "SO",
    # ... (add more tickers as needed)
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
    return combined

def fetch_all_news(api_key: str, limit: int = 50) -> List[Dict[str, Any]]:
    url = "https://api.polygon.io/v2/reference/news"
    params = {"limit": limit, "apiKey": api_key}
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
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

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

def get_options_chain(ticker: str) -> List[Dict[str, Any]]:
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        return []
    expiration = exps[0]
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

def get_option_profit_certainty(trade_type: str, future_prices: pd.DataFrame, option: Dict[str, Any]) -> float:
    from scipy.stats import norm
    pred_return = (future_prices.iloc[-1]["Ensemble"] - future_prices.iloc[0]["Ensemble"]) / future_prices.iloc[0]["Ensemble"]
    std_return = future_prices["Ensemble"].pct_change().std()
    if std_return == 0:
        std_return = 1e-9
    z = pred_return / std_return
    if trade_type == "CALL":
        prob = norm.cdf(z)
    elif trade_type == "PUT":
        prob = 1 - norm.cdf(z)
    else:
        prob = 0.5
    return prob

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

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=50, show_progress_bar=False)
    best_params = study.best_trial.params
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

        # Calculate volatility
        volatility = future_prices["Ensemble"].pct_change().std()

        # Generate the trade type based on predicted return, sentiment, and volatility
        trade_type = generate_trade_type_improved(predicted_return, alt_sentiment, volatility)

        if trade_type == "NEUTRAL":
            return None

        # Calculate strike price for options using predicted price
        strike = calculate_strike(current_price, predicted_return, trade_type)

        # Get the options chain for the ticker
        options_chain = get_options_chain(ticker)

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

# ----- FINANCIAL STATEMENTS FUNCTIONS (omitted for brevity; include as needed) -----
def get_quarterly_financials(ticker: str, years: int = 5) -> pd.DataFrame:
    url = f"https://api.polygon.io/vX/reference/financials"
    params = {"ticker": ticker, "limit": years * 4, "timeframe": "quarterly", "apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    if r.status_code != 200 or "results" not in data:
        return pd.DataFrame()
    financials_list = []
    for entry in data["results"]:
        inc = entry.get("financials", {}).get("income_statement", {})
        bal = entry.get("financials", {}).get("balance_sheet", {})
        rd_val = inc.get("research_and_development", {}).get("value") if "research_and_development" in inc else None
        financials_list.append({
            "Quarter": f"{entry.get('fiscal_period', 'Unknown')} {entry.get('fiscal_year', 'Unknown')}",
            "Filing Date": entry.get("filing_date", "N/A"),
            "Revenue": inc.get("revenues", {}).get("value"),
            "Net Income": inc.get("net_income_loss", {}).get("value"),
            "Gross Profit": inc.get("gross_profit", {}).get("value"),
            "R&D": rd_val,
            "Equity": bal.get("equity", {}).get("value"),
            "Assets": bal.get("assets", {}).get("value")
        })
    return pd.DataFrame(financials_list)

def compute_qoq_growth(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["Filing Date"] = df["Filing Date"].replace("N/A", np.nan)
        df = df.dropna(subset=["Filing Date"]).copy()
        df["Filing Date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
        df["Quarter"] = df["Filing Date"].dt.to_period("Q")
        df = df.dropna(subset=["Quarter"]).sort_values("Quarter")
        if "Revenue" not in df.columns or "Net Income" not in df.columns:
            return pd.DataFrame()
        df["Revenue Growth (%)"] = df["Revenue"].pct_change() * 100
        df["Net Income Growth (%)"] = df["Net Income"].pct_change() * 100
        if "Gross Profit" in df.columns and df["Gross Profit"].notna().any():
            df["Gross Margin (%)"] = (df["Gross Profit"] / df["Revenue"]) * 100
        else:
            df["Gross Margin (%)"] = np.nan
        if "R&D" in df.columns and df["R&D"].notna().any():
            df["R&D as % of Revenue"] = (df["R&D"] / df["Revenue"]) * 100
            df["R&D Growth (%)"] = df["R&D"].pct_change() * 100
        else:
            df["R&D as % of Revenue"] = np.nan
            df["R&D Growth (%)"] = np.nan
        if "Equity" in df.columns and df["Equity"].notna().any():
            df["ROE (%)"] = (df["Net Income"] / df["Equity"]) * 100
        else:
            df["ROE (%)"] = np.nan
        if "Assets" in df.columns and df["Assets"].notna().any():
            df["ROA (%)"] = (df["Net Income"] / df["Assets"]) * 100
        else:
            df["ROA (%)"] = np.nan
        return df[["Quarter", "Revenue", "Net Income", "Gross Profit", "R&D",
                   "Revenue Growth (%)", "Net Income Growth (%)", "Gross Margin (%)",
                   "R&D as % of Revenue", "R&D Growth (%)", "ROE (%)", "ROA (%)"]]
    except Exception:
        return pd.DataFrame()

def get_quarterly_financials_safe(ticker: str, years: int = 5, retries: int = 3) -> Optional[pd.DataFrame]:
    for _ in range(retries):
        df = get_quarterly_financials(ticker, years)
        if not df.empty:
            return df
    return None

def get_one_week_data(ticker: str) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df

# ----- MAIN FUNCTION FOR TERMINAL OUTPUT -----
def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    training_start = "2020-01-01"
    forecast_days = 7
   


    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        try:
            result = analyze_stock(ticker, training_start, forecast_days)
        except Exception as ex:
            print(f"Error analyzing {ticker}: {ex}")
            continue

        if result:
            print(f"Ticker: {result['ticker']}")
            print(f"Predicted Return: {result['predicted_return']:.2%}")
            print(f"Trade Type: {result['trade_type']}")
            if result['option_params']:
                print("Option Recommendation:")
                option = result['option_params']
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
        print("-" * 50)

if __name__ == "__main__":
    main()


##############################
# STREAMLIT APP & NAVIGATION
##############################
# (If you need to clear query parameters, you can use st.query_params directly)
'''query_params = st.query_params

# (Note: st.set_page_config was already called at the top of the file.)

page = st.sidebar.selectbox("Select Page", ["Market Scan", "Financial Statements"])

# Sidebar: Option to use the full watchlist of HOT_TICKERS
use_watchlist = st.sidebar.checkbox("Use full HOT_TICKERS watchlist (500+ stocks)", value=True)
if use_watchlist:
    st.sidebar.write("### Full Watchlist (Sample)")
    st.sidebar.write(HOT_TICKERS[:20] + ["..."])

##############################
# MAIN APP LOGIC
##############################
if page == "Market Scan":
    st.title("💡 Market Scan (1‑Week Forecast with Real Options & Profit Certainty)")
    st.markdown("This scan ranks stocks using news, Yahoo trending data, and heavy alternative sentiment from Twitter and Reddit combined with technical analysis.")
    if st.button("Run Market Scan"):
        with st.spinner("Running market scan..."):
            horizon = 7
            training_start = "2018-01-01"
            news_articles = fetch_all_news(POLYGON_API_KEY, limit=50)
            news_tickers = rank_tickers_by_sentiment(news_articles, top_n=10)
            yahoo_trending = fetch_yahoo_trending_tickers()
            combined_tickers = list(set(news_tickers + yahoo_trending))
            st.write("### Candidate Tickers")
            st.write(combined_tickers)
            results = []
            for ticker in combined_tickers:
                st.write(f"Analyzing {ticker}...")
                res = analyze_stock(ticker, training_start, horizon)
                if res:
                    results.append(res)
            if results:
                results.sort(key=lambda x: x["combined_score"], reverse=True)
                top_results = results[:5]
                st.write("### Top 5 Options Trade Ideas")
                cols = st.columns(5)
                for idx, res in enumerate(top_results):
                    ticker_str = res["ticker"]
                    button_html = f"""
                    <div style="background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
                                -webkit-background-clip: text;
                                -webkit-text-fill-color: transparent;
                                font-weight: bold;
                                font-size: 24px;
                                cursor: pointer;"
                         onclick="window.location.href='?selected_ticker={ticker_str}'">
                         {ticker_str}
                    </div>
                    """
                    cols[idx].markdown(button_html, unsafe_allow_html=True)
                st.write("#### Detailed Options Trade Information for Top Pick")
                top_option = top_results[0]["option_params"]
                st.write(f"**Ticker:** {top_results[0]['ticker']}")
                st.write(f"**Trade Type:** {top_results[0]['trade_type']}")
                st.write(f"**Predicted Return (%):** {round(top_results[0]['predicted_return']*100,2)}")
                st.write(f"**Strike:** {top_option.get('strike', '')}")
                st.write(f"**Expiration:** {top_option.get('expiration', '')}")
                st.write(f"**Profit Certainty (%):** {top_option.get('profitCertainty', '')}")
                best_ticker = top_results[0]["ticker"]
                st.write(f"### 1‑Week Chart for {best_ticker}")
                one_week_df = get_one_week_data(best_ticker)
                if not one_week_df.empty:
                    st.line_chart(one_week_df["Close"])
                else:
                    st.warning(f"No 1‑week data found for {best_ticker}.")
            else:
                st.error("No valid stock analyses were generated from the market scan.")

elif page == "Financial Statements":
    selected_ticker = query_params.get("selected_ticker", [None])[0]
    if selected_ticker:
        st.title(f"📊 Financial Statements for {selected_ticker}")
        st.markdown("Detailed quarterly financials and QoQ metrics for the selected ticker.")
        if st.button("Back to Top 5"):
            st.query_params({})  # Clear query parameters
            st.experimental_rerun()
        df_statements = get_quarterly_financials_safe(selected_ticker, years=5, retries=3)
        if df_statements is None or df_statements.empty:
            st.warning(f"No financial data found for {selected_ticker}.")
        else:
            st.write("### Raw Quarterly Financials")
            st.dataframe(df_statements)
            df_growth = compute_qoq_growth(df_statements)
            if df_growth.empty:
                st.warning("Could not compute QoQ growth metrics for these statements.")
            else:
                def color_positive(val):
                    try:
                        val = float(val)
                    except:
                        return ""
                    color = "green" if val > 0 else "red"
                    return f"color: {color}"
                styled_df = df_growth.style.applymap(color_positive, subset=["Revenue Growth (%)", "Net Income Growth (%)", "R&D Growth (%)"])
                st.write("### QoQ Growth & Metrics")
                st.dataframe(styled_df)
    else:
        st.title("📊 Financial Statements – Top 5 Candidates")
        st.markdown("This scan ranks stocks from our watchlist. Click a ticker (in rainbow‑text) to view its detailed financials.")
        horizon = 7
        training_start = "2018-01-01"
        if use_watchlist:
            combined_tickers = list(set(HOT_TICKERS))
        else:
            combined_tickers = get_sp500_tickers()
        st.write("### Candidate Tickers from Watchlist")
        st.write(combined_tickers[:50] + ["..."])
        results = []
        for ticker in combined_tickers:
            st.write(f"Analyzing {ticker}...")
            res = analyze_stock(ticker, training_start, horizon)
            if res:
                results.append(res)
        if results:
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            top_results = results[:5]
            st.write("### Top 5 Candidates (Click a ticker below)")
            cols = st.columns(5)
            for idx, res in enumerate(top_results):
                ticker_str = res["ticker"]
                button_html = f"""
                <div style="background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            font-weight: bold;
                            font-size: 24px;
                            cursor: pointer;"
                     onclick="window.location.href='?selected_ticker={ticker_str}'">
                     {ticker_str}
                </div>
                """
                cols[idx].markdown(button_html, unsafe_allow_html=True)
        else:
            st.error("No valid stock analyses were generated from the market scan.")

##############################
# NEXT STEPS & SUGGESTIONS
##############################
st.write("### Next Steps & Suggestions")
st.markdown("""
- **Automated Trade Execution:** Integrate with your brokerage’s API for automated options trade execution.
- **Scheduled Model Runs:** Deploy on a cloud server with scheduled runs (e.g., AWS Lambda).
- **Enhanced Alternative Data:** Expand sentiment analysis to include additional data sources (replace placeholders with real API calls).
- **Risk Management Optimization:** Continuously refine your risk parameters and backtest your strategies.
- **Continuous Learning:** Use the Optuna dashboard to monitor hyperparameter tuning.

**Disclaimer**: All predictions and signals are for educational purposes only.
""")
