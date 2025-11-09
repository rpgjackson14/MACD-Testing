import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime as datetime
import yfinance as yf
import matplotlib.dates as mdates

plt.style.use("dark_background")


#Global Variables
#Settings
symbol = "SPY"
todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
file_path = symbol + "_data.csv"

print ("Today's date is: ", todays_date)

#MACD Setup
short_window = 12
long_window = 26
signal_window = 9


# # # Functions # # #

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def normalize_csv_headers(file_path: str) -> None:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return
    try:
        df = pd.read_csv(file_path, header=[0, 1])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df.to_csv(file_path, index=False)
            return
    except Exception:
        pass
    # fallback: read and rewrite to ensure one header row
    try:
        df = pd.read_csv(file_path)
        df.to_csv(file_path, index=False)
    except Exception:
        return

# Data Scrape Function

def scrape_data(symbol: str, file_path: str) -> None:
    data = yf.download(symbol, period="max", interval="1d", progress=False, auto_adjust=False)
    # flatten any MultiIndex columns and ensure a 'Date' column exists in CSV
    data = flatten_columns(data)
    data.reset_index().to_csv(file_path, index=False)
    print("Downloaded data to", file_path)

# check if data is up to date

def get_csv_last_date(file_path: str):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    # Read CSV, tolerate either a Date column or a Date index
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
    except Exception:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = df.rename_axis("Date").reset_index()
    if df.empty or "Date" not in df.columns:
        return None
    return pd.to_datetime(df["Date"].max()).normalize()

def get_latest_online_date(symbol: str):
    recent = yf.download(symbol, period="30d", interval="1d", progress=False, auto_adjust=False)
    if recent is None or recent.empty:
        return None
    return pd.to_datetime(recent.index.max()).normalize()

def download_gap(symbol: str, start_date, file_path: str):
    # start_date: the first date we need (inclusive)
    data = yf.download(symbol, start=start_date, interval="1d", progress=False, auto_adjust=False)
    if data is None or data.empty:
        return False
    data = flatten_columns(data)
    data = data.reset_index()  # ensure a Date column exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            existing = pd.read_csv(file_path, parse_dates=["Date"])
        except ValueError:
            existing = pd.read_csv(file_path, parse_dates=True, index_col=0).rename_axis("Date").reset_index()
        merged = pd.concat([existing, data], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Date"]).sort_values("Date")
    else:
        merged = data.drop_duplicates(subset=["Date"]).sort_values("Date")
    merged.to_csv(file_path, index=False)
    return True

def backfill_earlier_history(symbol: str, file_path: str) -> None:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return
    try:
        existing = pd.read_csv(file_path, parse_dates=["Date"]).sort_values("Date")
    except Exception:
        return
    if existing.empty or "Date" not in existing.columns:
        return
    earliest = pd.to_datetime(existing["Date"].min()).normalize()
    # If we already go far back (e.g., before 1995), skip
    if earliest <= pd.Timestamp("1990-01-01"):
        return
    # Fetch data strictly before the earliest date we have
    older = yf.download(symbol, start="1900-01-01", end=(earliest - pd.Timedelta(days=1)).date().isoformat(), interval="1d", progress=False, auto_adjust=False)
    if older is None or older.empty:
        return
    older = flatten_columns(older).reset_index()
    merged = pd.concat([older, existing], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"]).sort_values("Date")
    merged.to_csv(file_path, index=False)

def ensure_data_up_to_date(symbol: str, file_path: str):
    last_csv_date = get_csv_last_date(file_path)

    # If we have no CSV yet, get a full baseline
    if last_csv_date is None:
        baseline = yf.download(symbol, period="max", interval="1d", progress=False, auto_adjust=False)
        if baseline is not None and not baseline.empty:
            baseline = flatten_columns(baseline)
            baseline.reset_index().to_csv(file_path, index=False)
        return

    # Always attempt an earlier backfill in case the CSV was seeded with limited history
    backfill_earlier_history(symbol, file_path)

    latest_online = get_latest_online_date(symbol)
    if latest_online is None:
        return  # no network/new data; keep existing file

    if last_csv_date >= latest_online:
        return  # already up to date (earlier backfill already attempted)

    # Download from the day after our last CSV date
    start = (last_csv_date + pd.Timedelta(days=1)).date().isoformat()
    download_gap(symbol, start, file_path)



if __name__ == "__main__":
    ensure_data_up_to_date(symbol, file_path)
    normalize_csv_headers(file_path)
    stock_data = pd.read_csv(file_path, parse_dates=["Date"])
    if not stock_data.empty:
        print(f"Loaded {len(stock_data)} rows from {stock_data['Date'].min().date()} through {stock_data['Date'].max().date()}")

def evaluate_params(data: pd.DataFrame, fast: int, slow: int, sig: int):
    d = data.copy()
    d = d.sort_values("Date")
    d.set_index("Date", inplace=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["EMA_fast"] = d["Close"].ewm(span=fast, adjust=False).mean()
    d["EMA_slow"] = d["Close"].ewm(span=slow, adjust=False).mean()
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["Signal"] = d["MACD"].ewm(span=sig, adjust=False).mean()
    d["hist_difference"] = d["MACD"] - d["Signal"]
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["trenddirection"] = np.where(d["Close"] > d["EMA200"], "up", "down")
    buy = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        if (
            (d["hist_difference"].iloc[i] > 0)
            and (d["hist_difference"].iloc[i - 1] < 0)
            and (d["MACD"].iloc[i] > d["Signal"].iloc[i])
            and (d["Signal"].iloc[i] < 0)
            and (d["trenddirection"].iloc[i] == "up")
        ):
            buy[i] = 1
    d["buy_signal"] = buy
    stop_loss_pct = 0.02
    take_profit_pct = 0.05
    initial_capital = 100000.0
    cash = initial_capital
    shares = 0
    in_pos = False
    entry_price = 0.0
    entry_date = None
    trades = []
    peak = initial_capital
    eq_series = []
    for idx, row in d.iterrows():
        price = row["Close"]
        if not in_pos and row["buy_signal"] == 1 and pd.notna(price):
            sh = int(cash // price)
            if sh > 0:
                shares = sh
                cash -= shares * price
                entry_price = price
                entry_date = idx
                in_pos = True
        elif in_pos and pd.notna(price):
            sl = entry_price * (1 - stop_loss_pct)
            tp = entry_price * (1 + take_profit_pct)
            reason = None
            if price <= sl:
                reason = "loss"
            elif price >= tp:
                reason = "win"
            if reason is not None:
                cash += shares * price
                pnl = (price - entry_price) * shares
                trades.append((entry_date, idx, entry_price, price, reason, pnl))
                shares = 0
                in_pos = False
                entry_price = 0.0
                entry_date = None
        eq = cash + (shares * price if in_pos and pd.notna(price) else 0)
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak > 0 else 0
        eq_series.append(eq)
    if in_pos:
        last_price = d["Close"].iloc[-1]
        cash += shares * last_price
        pnl = (last_price - entry_price) * shares
        trades.append((entry_date, d.index[-1], entry_price, last_price, "close", pnl))
        shares = 0
        in_pos = False
    final_eq = cash
    total_return = (final_eq - initial_capital) / initial_capital
    wins = sum(1 for t in trades if t[4] == "win")
    losses = sum(1 for t in trades if t[4] == "loss")
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    max_dd = 0.0
    running_peak = -np.inf
    for v in eq_series:
        running_peak = max(running_peak, v)
        if running_peak > 0:
            max_dd = min(max_dd, (v - running_peak) / running_peak)
    return {
        "fast": fast,
        "slow": slow,
        "signal": sig,
        "final_equity": final_eq,
        "total_return": total_return,
        "trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }

if __name__ == "__main__" and not stock_data.empty:
    results = []
    fasts = [8, 10, 12, 16]
    slows = [20, 26, 35]
    signals = [5, 9]
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            for sg in signals:
                r = evaluate_params(stock_data, f, s, sg)
                results.append(r)
    if results:
        df_results = pd.DataFrame(results).sort_values("final_equity", ascending=False)
        print("\nMACD window sweep (top 10):")
        print(df_results.head(10).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# Prepare dataframe for indicators
df = stock_data.copy()
df.set_index("Date", inplace=True)
# coerce numeric columns in case of stray strings
for c in ["Open", "High", "Low", "Close", "Volume"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

#calculating EMAs and MACDs
df["EMA_12"] = df["Close"].ewm(span=short_window, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=long_window, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

# Additional indicators and signals from tutorial
df["hist_difference"] = df["MACD"] - df["Signal"]
df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
df["trenddirection"] = np.where(df["Close"] > df["EMA200"], "up", "down")

# Buy signal: histogram crosses above 0, signal below 0, MACD > Signal, and uptrend
df["buy_signal"] = 0
for i in range(1, len(df)):
    if (
        (df["hist_difference"].iloc[i] > 0)
        and (df["hist_difference"].iloc[i - 1] < 0)
        and (df["MACD"].iloc[i] > df["Signal"].iloc[i])
        and (df["Signal"].iloc[i] < 0)
        and (df["trenddirection"].iloc[i] == "up")
    ):
        df.loc[df.index[i], "buy_signal"] = 1

# simple backtest
stop_loss_pct = 0.02
take_profit_pct = 0.05
initial_capital = 100000.0
cash = initial_capital
shares = 0
in_pos = False
entry_price = 0.0
entry_date = None
trades = []
portfolio_values = []
for idx, row in df.iterrows():
    price = row["Close"]
    if not in_pos and row["buy_signal"] == 1 and pd.notna(price):
        sh = int(cash // price)
        if sh > 0:
            shares = sh
            cash -= shares * price
            entry_price = price
            entry_date = idx
            in_pos = True
    elif in_pos and pd.notna(price):
        sl = entry_price * (1 - stop_loss_pct)
        tp = entry_price * (1 + take_profit_pct)
        reason = None
        if price <= sl:
            reason = "loss"
        elif price >= tp:
            reason = "win"
        if reason is not None:
            cash += shares * price
            pnl = (price - entry_price) * shares
            trades.append((entry_date, idx, entry_price, price, reason, pnl))
            shares = 0
            in_pos = False
            entry_price = 0.0
            entry_date = None
    eq = cash + (shares * price if in_pos and pd.notna(price) else 0)
    portfolio_values.append((idx, eq))
if in_pos:
    last_price = df["Close"].iloc[-1]
    cash += shares * last_price
    pnl = (last_price - entry_price) * shares
    trades.append((entry_date, df.index[-1], entry_price, last_price, "close", pnl))
    shares = 0
    in_pos = False
equity_series = pd.Series([v for _, v in portfolio_values], index=[t for t, _ in portfolio_values], name="Portfolio")
#Plotting Figures
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 9))
fig.patch.set_facecolor("black")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("black")

#Top Plot
ax1.plot(df.index, df["Close"], label="Close", color="blue")
ax1.plot(df.index, df["EMA200"], label="EMA200", color="magenta", linewidth=1.2)
# mark buy signals
buys = df[df["buy_signal"] == 1]
if not buys.empty:
    ax1.scatter(buys.index, buys["Close"], marker="^", color="lime", s=40, label="Buy")
ax1.legend(facecolor="black", edgecolor="gray")

#middle plot
ax2.plot(df.index, df["MACD"], label="MACD", color="cyan")
ax2.plot(df.index, df["Signal"], label="Signal", color="orange")
ax2.legend(facecolor="black", edgecolor="gray")

# bottom plot: MACD histogram (MACD - Signal)
colors = np.where(df["hist_difference"] >= 0, "#00ff00", "#ff3333")
ax3.bar(df.index, df["hist_difference"], color=colors, width=1.0, label="Histogram")
ax3.legend(facecolor="black", edgecolor="gray")

# portfolio figure
fig2 = plt.figure(figsize=(16, 9))
fig2.patch.set_facecolor("black")
axp = fig2.add_subplot(111)
axp.set_facecolor("black")
line_color = "green" if equity_series.iloc[-1] >= equity_series.iloc[0] else "red"
axp.plot(equity_series.index, equity_series.values, color=line_color, linewidth=2.5, label="Portfolio Value")
axp.set_title("Portfolio Value Over Time", color="white", fontsize=16)
axp.set_xlabel("Date", color="white")
axp.set_ylabel("Portfolio Value", color="white")
axp.grid(True, color="#333333", linestyle="--", alpha=0.3)
axp.legend(facecolor="black", edgecolor="gray", loc="upper left")
axp.xaxis.set_major_locator(mdates.YearLocator())
axp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig2.autofmt_xdate()

if len(trades) > 0:
    total_trades = len(trades)
    wins = sum(1 for t in trades if t[4] == "win")
    losses = total_trades - wins
    accuracy = wins / total_trades * 100
    print(f"Trades: {total_trades} | Wins: {wins} | Losses: {losses} | Accuracy: {accuracy:.2f}%")

plt.show()
