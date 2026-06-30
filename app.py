import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="StockLab", layout="wide")

st.title("StockLab")
st.subheader("AI-powered investing simulation lab")
st.caption(
    "Educational simulation only. Not financial advice. "
    "Model outputs are uncertain and may be wrong."
)

mode = st.sidebar.radio(
    "Choose simulation mode",
    ["Historical Lab", "AI Scenario Lab"]
)

starting_balance = st.sidebar.number_input(
    "Starting fake balance",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

tickers_input = st.sidebar.text_input(
    "Tickers",
    value="AAPL,MSFT,NVDA,SPY"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2024-01-01"))

strategy = st.sidebar.selectbox(
    "Strategy",
    ["Equal Weight", "Momentum Weight", "AI Confidence Weight"]
)

risk_level = st.sidebar.selectbox(
    "Risk level",
    ["Low", "Medium", "High"]
)

@st.cache_data
def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")

def calculate_equal_weight_portfolio(prices, balance):
    returns = prices.pct_change().dropna()
    weights = np.array([1 / len(prices.columns)] * len(prices.columns))
    portfolio_returns = returns.dot(weights)
    portfolio_value = balance * (1 + portfolio_returns).cumprod()
    return portfolio_value

def calculate_momentum_weights(prices):
    momentum = prices.pct_change(60).iloc[-1]
    momentum = momentum.clip(lower=0)
    if momentum.sum() == 0:
        return np.array([1 / len(momentum)] * len(momentum))
    return (momentum / momentum.sum()).values

def calculate_weighted_portfolio(prices, balance, weights):
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    portfolio_value = balance * (1 + portfolio_returns).cumprod()
    return portfolio_value

def max_drawdown(series):
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown.min()

def make_features(prices):
    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        df = pd.DataFrame(index=s.index)
        df["ticker"] = ticker
        df["return_7d"] = s.pct_change(7)
        df["return_30d"] = s.pct_change(30)
        df["return_90d"] = s.pct_change(90)
        df["volatility_30d"] = s.pct_change().rolling(30).std()
        df["future_return_30d"] = s.shift(-30) / s - 1
        rows.append(df)
    features = pd.concat(rows).dropna()
    return features

def train_simple_model(prices):
    features = make_features(prices)
    features["target"] = (features["future_return_30d"] > 0).astype(int)

    X = features[["return_7d", "return_30d", "return_90d", "volatility_30d"]]
    y = features["target"]

    if len(features) < 100:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    latest = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        row = {
            "ticker": ticker,
            "return_7d": s.pct_change(7).iloc[-1],
            "return_30d": s.pct_change(30).iloc[-1],
            "return_90d": s.pct_change(90).iloc[-1],
            "volatility_30d": s.pct_change().rolling(30).std().iloc[-1],
        }
        latest.append(row)

    latest_df = pd.DataFrame(latest).dropna()
    if latest_df.empty:
        return model, pd.DataFrame()

    probs = model.predict_proba(
        latest_df[["return_7d", "return_30d", "return_90d", "volatility_30d"]]
    )[:, 1]

    latest_df["confidence_up_next_30d"] = probs
    latest_df["model_accuracy_test"] = accuracy

    return model, latest_df

def generate_scenario(prices, days=126, scenario_type="Neutral"):
    last_prices = prices.iloc[-1]
    daily_returns = prices.pct_change().dropna()

    mean_returns = daily_returns.mean()
    vol = daily_returns.std()

    if scenario_type == "Bull":
        drift_multiplier = 1.5
    elif scenario_type == "Bear":
        drift_multiplier = -0.5
    elif scenario_type == "High Volatility":
        drift_multiplier = 1.0
        vol = vol * 2
    else:
        drift_multiplier = 1.0

    simulated_prices = pd.DataFrame(index=range(days), columns=prices.columns)
    simulated_prices.iloc[0] = last_prices

    for day in range(1, days):
        random_returns = np.random.normal(
            mean_returns * drift_multiplier,
            vol
        )
        simulated_prices.iloc[day] = simulated_prices.iloc[day - 1] * (1 + random_returns)

    simulated_prices.index = pd.date_range(
        start=prices.index[-1],
        periods=days,
        freq="B"
    )

    return simulated_prices.astype(float)

if st.button("Run StockLab Simulation"):
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    prices = load_prices(tickers, start_date, end_date)

    if prices.empty:
        st.error("No price data found. Try different tickers or dates.")
        st.stop()

    st.write("### Price Data")
    st.line_chart(prices)

    model, confidence_df = train_simple_model(prices)

    st.write("### AI Confidence Signals")
    if confidence_df is not None and not confidence_df.empty:
        display_df = confidence_df.copy()
        display_df["confidence_up_next_30d"] = (
            display_df["confidence_up_next_30d"] * 100
        ).round(1).astype(str) + "%"
        display_df["model_accuracy_test"] = (
            display_df["model_accuracy_test"] * 100
        ).round(1).astype(str) + "%"
        st.dataframe(display_df)
    else:
        st.info("Not enough data to train confidence model.")

    if mode == "Historical Lab":
        st.write("## Historical Lab Results")

        if strategy == "Equal Weight":
            portfolio = calculate_equal_weight_portfolio(prices, starting_balance)
        elif strategy == "Momentum Weight":
            weights = calculate_momentum_weights(prices)
            portfolio = calculate_weighted_portfolio(prices, starting_balance, weights)
            st.write("Momentum weights:")
            st.dataframe(pd.DataFrame({"ticker": prices.columns, "weight": weights}))
        else:
            if confidence_df is not None and not confidence_df.empty:
                weights_raw = confidence_df.set_index("ticker").reindex(prices.columns)[
                    "confidence_up_next_30d"
                ].fillna(1 / len(prices.columns)).values
                weights = weights_raw / weights_raw.sum()
            else:
                weights = np.array([1 / len(prices.columns)] * len(prices.columns))
            portfolio = calculate_weighted_portfolio(prices, starting_balance, weights)
            st.write("AI confidence weights:")
            st.dataframe(pd.DataFrame({"ticker": prices.columns, "weight": weights}))

        spy_prices = load_prices(["SPY"], start_date, end_date)
        spy_portfolio = calculate_equal_weight_portfolio(spy_prices, starting_balance)

        results = pd.DataFrame({
            "StockLab Strategy": portfolio,
            "SPY Benchmark": spy_portfolio.reindex(portfolio.index).ffill()
        }).dropna()

        st.line_chart(results)

        final_value = portfolio.iloc[-1]
        total_return = (final_value / starting_balance - 1) * 100
        dd = max_drawdown(portfolio) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Final fake balance", f"${final_value:,.2f}")
        col2.metric("Total return", f"{total_return:.2f}%")
        col3.metric("Max drawdown", f"{dd:.2f}%")

    else:
        st.write("## AI Scenario Lab Results")

        scenario_type = st.selectbox(
            "Scenario type",
            ["Neutral", "Bull", "Bear", "High Volatility"]
        )

        days = st.slider("Simulation days", 30, 252, 126)

        simulated_prices = generate_scenario(prices, days=days, scenario_type=scenario_type)

        st.write("### Model-generated market scenario")
        st.line_chart(simulated_prices)

        if strategy == "Equal Weight":
            scenario_portfolio = calculate_equal_weight_portfolio(
                simulated_prices,
                starting_balance
            )
        elif strategy == "Momentum Weight":
            weights = calculate_momentum_weights(prices)
            scenario_portfolio = calculate_weighted_portfolio(
                simulated_prices,
                starting_balance,
                weights
            )
        else:
            if confidence_df is not None and not confidence_df.empty:
                weights_raw = confidence_df.set_index("ticker").reindex(prices.columns)[
                    "confidence_up_next_30d"
                ].fillna(1 / len(prices.columns)).values
                weights = weights_raw / weights_raw.sum()
            else:
                weights = np.array([1 / len(prices.columns)] * len(prices.columns))
            scenario_portfolio = calculate_weighted_portfolio(
                simulated_prices,
                starting_balance,
                weights
            )

        st.write("### Portfolio timelapse result")
        st.line_chart(scenario_portfolio)

        final_value = scenario_portfolio.iloc[-1]
        total_return = (final_value / starting_balance - 1) * 100
        dd = max_drawdown(scenario_portfolio) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Final fake balance", f"${final_value:,.2f}")
        col2.metric("Scenario return", f"{total_return:.2f}%")
        col3.metric("Scenario max drawdown", f"{dd:.2f}%")

        st.warning(
            "This is a model-generated scenario, not a prediction, guarantee, "
            "or recommendation to buy or sell securities."
        )