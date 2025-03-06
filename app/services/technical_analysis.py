import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 🏦 Fetch Stock Data
def fetch_stock_data(stock_name, period="10y", interval="1d"):
    stock = yf.Ticker(stock_name)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df

# 📈 Calculate Technical Indicators
def calculate_technical_indicators(df):
    if df is None or df.empty:
        return None

    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df["Close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD Calculation
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    return df

# 📊 Prepare ML Data
def prepare_ml_data(df):
    df["Price Change"] = df["Close"].pct_change() * 100
    df["Target"] = (df["Price Change"].shift(-1) > 0).astype(int)

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.fillna(value=0, inplace=True)

    features = ["SMA_20", "SMA_50", "EMA_20", "RSI", "MACD", "Signal", "Price Change"]

    X = df[features]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 🤖 Train Model
def train_ml_model(X, y):
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

# 🚀 Predict Next Move with Explanation
def predict_next_day(model, latest_data, df):
    latest_data = latest_data.reshape(1, -1)
    prediction = model.predict(latest_data)

    trend = "Bullish" if prediction[0] == 1 else "Bearish"
    reasons = []

    if df["RSI"].iloc[-1] > 70:
        reasons.append("RSI is above 70 (Overbought)")
    elif df["RSI"].iloc[-1] < 30:
        reasons.append("RSI is below 30 (Oversold)")

    if df["MACD"].iloc[-1] > df["Signal"].iloc[-1]:
        reasons.append("MACD is above the Signal line (Bullish)")
    else:
        reasons.append("MACD is below the Signal line (Bearish)")

    if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]:
        reasons.append("SMA_20 is above SMA_50 (Uptrend)")
    else:
        reasons.append("SMA_20 is below SMA_50 (Downtrend)")

    return {"trend": trend, "reasons": reasons}
