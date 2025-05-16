from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import time
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# Alpaca API credentials (replace with your own)
ALPACA_API_KEY = 'PK4W50EER0STT1V95130'
ALPACA_SECRET_KEY = 'ibaZ1ezJLaab7K5gevXKCAFeHDyRia4s52tfxJ0Y'

# Function to fetch 15-minute data with Alpaca
def fetch_data_alpaca(ticker, start_date, end_date, interval=TimeFrame(15, TimeFrameUnit.Minute), retries=3, delay=5):
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    for attempt in range(retries):
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=interval,
                start=start_date,
                end=end_date
            )
            bars = client.get_stock_bars(request_params).df
            if bars.empty:
                raise ValueError(f"No data retrieved for {ticker}. Check symbol, date range, or API limits.")
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index()
                bars = bars.set_index('timestamp')
            bars = bars[['open', 'high', 'low', 'close', 'volume']]
            bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            bars['Return'] = bars['Close'].pct_change()
            bars['Return_Lag1'] = bars['Return'].shift(1)
            bars['Return_Lag2'] = bars['Return'].shift(2)
            return bars
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            continue
    print(f"Failed to retrieve data for {ticker} after {retries} attempts.")
    return None

# Function to detect bearish engulfing pattern
def detect_bearish_engulfing(df):
    df['Bearish_Engulfing'] = 0
    for i in range(1, len(df)):
        prev_candle = df.iloc[i-1]
        curr_candle = df.iloc[i]
        uptrend = prev_candle['Close'] > prev_candle['Open'] or (i > 20 and prev_candle['Close'] > df['SMA20'].iloc[i-1])
        is_small_green = prev_candle['Close'] > prev_candle['Open'] and (prev_candle['Close'] - prev_candle['Open']) < 0.5 * df['ATR'].iloc[i-1]
        is_large_red = curr_candle['Open'] > curr_candle['Close'] and curr_candle['Open'] > prev_candle['Close'] and curr_candle['Close'] < prev_candle['Open']
        if uptrend and is_small_green and is_large_red and (curr_candle['Open'] - curr_candle['Close']) > (prev_candle['Close'] - prev_candle['Open']):
            df.iloc[i, df.columns.get_loc('Bearish_Engulfing')] = 1
    return df

# Function to detect bullish engulfing pattern
def detect_bullish_engulfing(df):
    df['Bullish_Engulfing'] = 0
    for i in range(1, len(df)):
        prev_candle = df.iloc[i-1]
        curr_candle = df.iloc[i]
        downtrend = prev_candle['Close'] < prev_candle['Open'] or (i > 20 and prev_candle['Close'] < df['SMA20'].iloc[i-1])
        is_small_red = prev_candle['Close'] < prev_candle['Open'] and (prev_candle['Open'] - prev_candle['Close']) < 0.5 * df['ATR'].iloc[i-1]
        is_large_green = curr_candle['Close'] > curr_candle['Open'] and curr_candle['Close'] > prev_candle['Open'] and curr_candle['Open'] < prev_candle['Close']
        if downtrend and is_small_red and is_large_green and (curr_candle['Close'] - curr_candle['Open']) > (prev_candle['Open'] - prev_candle['Close']):
            df.iloc[i, df.columns.get_loc('Bullish_Engulfing')] = 1
    return df

# Function to detect doji pattern
def detect_doji(df):
    df['Doji'] = 0
    for i in range(len(df)):
        candle = df.iloc[i]
        body_size = abs(candle['Close'] - candle['Open'])
        is_doji = body_size < 0.1 * df['ATR'].iloc[i] and abs(candle['High'] - candle['Low']) > 1.5 * body_size
        if is_doji:
            df.iloc[i, df.columns.get_loc('Doji')] = 1
    return df

# Function to detect hammer pattern
def detect_hammer(df):
    df['Hammer'] = 0
    for i in range(len(df)):
        candle = df.iloc[i]
        body_size = abs(candle['Close'] - candle['Open'])
        lower_wick = candle['Low'] - min(candle['Open'], candle['Close'])
        upper_wick = max(candle['Open'], candle['Close']) - candle['High']
        is_hammer = (body_size < 0.5 * df['ATR'].iloc[i] and lower_wick > 2 * body_size and upper_wick < 0.5 * body_size)
        if is_hammer:
            df.iloc[i, df.columns.get_loc('Hammer')] = 1
    return df

# Function to detect shooting star pattern
def detect_shooting_star(df):
    df['Shooting_Star'] = 0
    for i in range(len(df)):
        candle = df.iloc[i]
        body_size = abs(candle['Close'] - candle['Open'])
        upper_wick = max(candle['Open'], candle['Close']) - candle['High']
        lower_wick = candle['Low'] - min(candle['Open'], candle['Close'])
        is_shooting_star = (body_size < 0.5 * df['ATR'].iloc[i] and upper_wick > 2 * body_size and lower_wick < 0.5 * body_size)
        if is_shooting_star:
            df.iloc[i, df.columns.get_loc('Shooting_Star')] = 1
    return df

# Function to detect morning star pattern
def detect_morning_star(df):
    df['Morning_Star'] = 0
    for i in range(2, len(df)):
        prev_prev_candle = df.iloc[i-2]
        prev_candle = df.iloc[i-1]
        curr_candle = df.iloc[i]
        downtrend = prev_prev_candle['Close'] < prev_prev_candle['Open'] or (i > 22 and prev_prev_candle['Close'] < df['SMA20'].iloc[i-2])
        is_bearish = prev_candle['Close'] < prev_candle['Open'] and (prev_candle['Open'] - prev_candle['Close']) < 0.5 * df['ATR'].iloc[i-1]
        is_doji = abs(prev_candle['Close'] - prev_candle['Open']) < 0.1 * df['ATR'].iloc[i-1]
        is_bullish = curr_candle['Close'] > curr_candle['Open'] and curr_candle['Close'] > prev_prev_candle['Open'] and (curr_candle['Close'] - curr_candle['Open']) > 0.4 * df['ATR'].iloc[i]
        if downtrend and (is_bearish or is_doji) and is_bullish:
            df.iloc[i, df.columns.get_loc('Morning_Star')] = 1
    return df

# Function to detect evening star pattern
def detect_evening_star(df):
    df['Evening_Star'] = 0
    for i in range(2, len(df)):
        prev_prev_candle = df.iloc[i-2]
        prev_candle = df.iloc[i-1]
        curr_candle = df.iloc[i]
        uptrend = prev_prev_candle['Close'] > prev_prev_candle['Open'] or (i > 22 and prev_prev_candle['Close'] > df['SMA20'].iloc[i-2])
        is_bullish = prev_candle['Close'] > prev_candle['Open'] and (prev_candle['Close'] - prev_candle['Open']) < 0.5 * df['ATR'].iloc[i-1]
        is_doji = abs(prev_candle['Close'] - prev_candle['Open']) < 0.1 * df['ATR'].iloc[i-1]
        is_bearish = curr_candle['Open'] > curr_candle['Close'] and curr_candle['Close'] < prev_prev_candle['Close'] and (curr_candle['Open'] - curr_candle['Close']) > 0.5 * df['ATR'].iloc[i]
        if uptrend and (is_bullish or is_doji) and is_bearish:
            df.iloc[i, df.columns.get_loc('Evening_Star')] = 1
    return df

# Function to detect three white soldiers pattern
def detect_three_white_soldiers(df):
    df['Three_White_Soldiers'] = 0
    for i in range(2, len(df)):
        candle_1 = df.iloc[i-2]
        candle_2 = df.iloc[i-1]
        candle_3 = df.iloc[i]
        uptrend = candle_1['Close'] > candle_1['Open'] or (i > 22 and candle_1['Close'] > df['SMA20'].iloc[i-2])
        is_bullish_1 = candle_1['Close'] > candle_1['Open']
        is_bullish_2 = candle_2['Close'] > candle_2['Open'] and candle_2['Close'] > candle_1['Close']
        is_bullish_3 = candle_3['Close'] > candle_3['Open'] and candle_3['Close'] > candle_2['Close']
        if uptrend and is_bullish_1 and is_bullish_2 and is_bullish_3:
            df.iloc[i, df.columns.get_loc('Three_White_Soldiers')] = 1
    return df

# Function to calculate trendlines using linear regression
def add_trendlines(df, lookback=20):
    df['Trendline_Slope'] = 0.0
    df['Trendline_Value'] = np.nan

    for i in range(lookback, len(df)):
        window = df['Close'].iloc[i-lookback:i]
        x = np.arange(len(window))
        slope, intercept, _, _, _ = linregress(x, window)
        trendline_value = slope * (len(window) - 1) + intercept
        df.iloc[i, df.columns.get_loc('Trendline_Slope')] = slope
        df.iloc[i, df.columns.get_loc('Trendline_Value')] = trendline_value

    return df

# Function to calculate technical indicators
def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['SMA20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_hband()
    df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close'], window=20).bollinger_lband()
    df['Volume_SMA20'] = ta.trend.SMAIndicator(df['Volume'], window=20).sma_indicator()
    df = add_trendlines(df, lookback=20)
    df = detect_bearish_engulfing(df)
    df = detect_bullish_engulfing(df)
    df = detect_doji(df)
    df = detect_hammer(df)
    df = detect_shooting_star(df)
    df = detect_morning_star(df)
    df = detect_evening_star(df)
    df = detect_three_white_soldiers(df)
    return df

# Function to calculate position size based on ATR
def calculate_position_size(df, risk_per_trade=0.02, account_balance=100000):
    df['Position_Size'] = (account_balance * risk_per_trade) / df['ATR']
    df['Position_Size'] = df['Position_Size'].clip(upper=account_balance * 0.1 / df['Close'])
    return df

# Backtesting class with rule-based trading, enhanced cooldown, and trailing stop for time-based exits
class Backtester:
    def __init__(self, initial_balance=100000, transaction_cost=0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_cost = transaction_cost
        self.trade_log = []
        self.equity_curve = []
        self.skipped_trades = []
        self.active_trades = {}  # Track open trades
        self.last_trade_timestamp = None  # Enhanced cooldown tracking

    def simulate_trade(self, date, open_price, high_price, low_price, close_price, volume, volume_sma20, rsi, adx, bearish_engulfing, bullish_engulfing, doji, hammer, shooting_star, morning_star, evening_star, three_white_soldiers, position_size, atr, current_index, df):
        # Enhanced cooldown check: prevent multiple entries at the same timestamp and enforce 1 bar cooldown
        if self.last_trade_timestamp == date:
            self.skipped_trades.append({'Date': date, 'Reason': 'Same timestamp cooldown'})
            return
        if self.last_trade_timestamp and (date - self.last_trade_timestamp).total_seconds() / 900 < 1:
            self.skipped_trades.append({'Date': date, 'Reason': 'Cooldown active (1 bar)'})
            return

        # Check for new trade entry
        trade_opened = False
        reason = None

        if rsi >= 70:
            reason = f"RSI ({rsi:.2f}) >= 70"
        elif adx < 10:
            reason = f"ADX ({adx:.2f}) < 10"
        elif bearish_engulfing == 1 or shooting_star == 1 or evening_star == 1:
            reason = "Bearish pattern detected"
        elif bullish_engulfing != 1 and hammer != 1 and morning_star != 1 and three_white_soldiers != 1:
            reason = "No bullish pattern detected"

        if not reason:
            shares = position_size / open_price
            cost = shares * open_price * self.transaction_cost
            self.balance -= cost
            entry_price = open_price
            stop_price = max(entry_price - 2.5 * atr, entry_price * 0.97)  # 2.5x ATR with 3% cap
            trade_id = f"{date}_{current_index}"
            self.active_trades[trade_id] = {
                'entry_index': current_index,
                'entry_price': entry_price,
                'shares': shares,
                'stop_price': stop_price,
                'profit_price': entry_price * 1.015,  # 1.5% take-profit
                'rsi': rsi,
                'adx': adx,
                'bearish_engulfing': bearish_engulfing,
                'bullish_engulfing': bullish_engulfing,
                'doji': doji,
                'hammer': hammer,
                'shooting_star': shooting_star,
                'morning_star': morning_star,
                'evening_star': evening_star,
                'three_white_soldiers': three_white_soldiers,
                'trendline_slope': df['Trendline_Slope'].iloc[current_index],
                'highest_price': entry_price  # Track for trailing stop
            }
            trade_opened = True
            self.last_trade_timestamp = date

        else:
            self.skipped_trades.append({'Date': date, 'Reason': reason})

        # Process active trades
        closed_trades = []
        for trade_id, trade in list(self.active_trades.items()):
            entry_price = trade['entry_price']
            shares = trade['shares']
            stop_price = trade['stop_price']
            profit_price = trade['profit_price']
            entry_index = trade['entry_index']
            trade_duration = current_index - entry_index

            # Update highest price for trailing stop
            trade['highest_price'] = max(trade['highest_price'], high_price)

            trade_return = 0
            exit_reason = None

            if low_price <= stop_price:
                trade_return = shares * (stop_price - entry_price)
                exit_reason = "Stop-loss"
            elif high_price >= profit_price:
                trade_return = shares * (profit_price - entry_price)
                exit_reason = "Take-profit"
            elif trade_duration >= 2:  # Trailing stop logic after 2 bars
                current_profit = (close_price - entry_price) / entry_price
                trailing_stop = trade['highest_price'] - 0.75 * atr
                if current_profit >= 0.002 and low_price <= trailing_stop:  # Minimum 0.2% profit before trailing stop
                    trade_return = shares * (trailing_stop - entry_price)
                    exit_reason = "Time-based (trailing stop)"
                elif trade_duration >= 7:  # Max 7 bars
                    trade_return = shares * (close_price - entry_price)
                    exit_reason = "Time-based"
            elif trade_opened and trade_id == f"{date}_{current_index}":
                if low_price <= stop_price:
                    trade_return = shares * (stop_price - entry_price)
                    exit_reason = "Stop-loss"
                elif high_price >= profit_price:
                    trade_return = shares * (profit_price - entry_price)
                    exit_reason = "Take-profit"
                else:
                    continue
            else:
                continue

            if exit_reason:
                self.trade_log.append({
                    'Date': date,
                    'Type': 'Long',
                    'Entry_Price': entry_price,
                    'Exit_Price': stop_price if exit_reason == "Stop-loss" else profit_price if exit_reason == "Take-profit" else trailing_stop if "trailing stop" in exit_reason else close_price,
                    'Shares': shares,
                    'Return': trade_return,
                    'Balance': self.balance,
                    'Exit_Reason': exit_reason,
                    'RSI': trade['rsi'],
                    'ADX': trade['adx'],
                    'Bearish_Engulfing': trade['bearish_engulfing'],
                    'Bullish_Engulfing': trade['bullish_engulfing'],
                    'Doji': trade['doji'],
                    'Hammer': trade['hammer'],
                    'Shooting_Star': trade['shooting_star'],
                    'Morning_Star': trade['morning_star'],
                    'Evening_Star': trade['evening_star'],
                    'Three_White_Soldiers': trade['three_white_soldiers'],
                    'Trade_Duration': trade_duration
                })
                self.balance += trade_return
                closed_trades.append(trade_id)

        for trade_id in closed_trades:
            del self.active_trades[trade_id]

        self.equity_curve.append({'Date': date, 'Equity': self.balance})

    def calculate_metrics(self):
        equity_df = pd.DataFrame(self.equity_curve).set_index('Date')
        returns = equity_df['Equity'].pct_change().dropna()

        cum_return = (equity_df['Equity'].iloc[-1] / self.initial_balance) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 26) if returns.std() != 0 else 0
        rolling_max = equity_df['Equity'].cummax()
        drawdowns = (equity_df['Equity'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        trades = pd.DataFrame(self.trade_log)
        win_rate = len(trades[trades['Return'] > 0]) / len(trades) if not trades.empty else 0

        return {
            'Cumulative Return': cum_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Number of Trades': len(self.trade_log)
        }

    def plot_equity_curve(self, filename='equity_curve.png'):
        equity_df = pd.DataFrame(self.equity_curve)
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(equity_df['Date']), equity_df['Equity'], label='Equity Curve')
        plt.title('Backtesting Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_trades(self, df, period, filename='trade_plot.png'):
        trades = pd.DataFrame(self.trade_log)
        if trades.empty:
            return
        start_date = pd.to_datetime(period.split(' to ')[0]).tz_localize('UTC')
        end_date = pd.to_datetime(period.split(' to ')[1]).tz_localize('UTC')
        print(f"Plotting trades for period: {start_date} to {end_date}, Index dtype: {df.index.dtype}")
        period_df = df[(df.index >= start_date) & (df.index <= end_date)]
        if period_df.empty:
            print(f"No data in period {period} for plotting.")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(period_df.index, period_df['Close'], label='Close Price')
        buys = trades[trades['Type'] == 'Long']
        plt.scatter(buys['Date'], buys['Entry_Price'], color='green', marker='^', label='Buy', s=100)
        plt.title(f'Trade Entries for {period}')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

# Function for walk-forward testing
def walk_forward_testing(df, train_window_days=30, test_window_days=15, initial_balance=100000):
    results = []
    all_equity_curves = []
    start_date = df.index[0]
    end_date = df.index[-1]
    current_date = start_date + timedelta(days=train_window_days)

    while current_date + timedelta(days=test_window_days) <= end_date:
        train_start = current_date - timedelta(days=train_window_days)
        train_end = current_date
        test_end = current_date + timedelta(days=test_window_days)

        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= current_date) & (df.index < test_end)]

        if len(train_df) < 100 or len(test_df) < 20:
            current_date += timedelta(days=test_window_days)
            continue

        test_df = calculate_position_size(test_df, risk_per_trade=0.02, account_balance=initial_balance)

        backtester = Backtester(initial_balance=initial_balance, transaction_cost=0.001)
        for idx, (date, row) in enumerate(test_df.iterrows()):
            backtester.simulate_trade(
                date=date,
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                volume=row['Volume'],
                volume_sma20=row['Volume_SMA20'],
                rsi=row['RSI'],
                adx=row['ADX'],
                bearish_engulfing=row['Bearish_Engulfing'],
                bullish_engulfing=row['Bullish_Engulfing'],
                doji=row['Doji'],
                hammer=row['Hammer'],
                shooting_star=row['Shooting_Star'],
                morning_star=row['Morning_Star'],
                evening_star=row['Evening_Star'],
                three_white_soldiers=row['Three_White_Soldiers'],
                position_size=row['Position_Size'] * row['Open'],
                atr=row['ATR'],
                current_index=idx,
                df=test_df
            )

        metrics = backtester.calculate_metrics()
        period = f"{current_date.strftime('%Y-%m-%d')} to {(test_end - timedelta(days=1)).strftime('%Y-%m-%d')}"
        metrics['Period'] = period
        results.append(metrics)

        # Print trade summary
        trades = pd.DataFrame(backtester.trade_log)
        if not trades.empty:
            print(f"Trade summary for period {period}:\n{trades[['Date', 'Type', 'Return', 'Balance', 'Exit_Reason', 'RSI', 'ADX', 'Bullish_Engulfing', 'Doji', 'Hammer', 'Morning_Star', 'Three_White_Soldiers', 'Trade_Duration']]}")
            print(f"Exit reason counts:\n{trades['Exit_Reason'].value_counts()}")
            print(f"Average return by exit reason:\n{trades.groupby('Exit_Reason')['Return'].mean()}")
            print(f"Trade outcomes:\n{trades['Return'].apply(lambda x: 'Win' if x > 0 else 'Loss').value_counts()}")
        skipped = pd.DataFrame(backtester.skipped_trades)
        if not skipped.empty:
            print(f"Skipped trades for period {period}:\n{skipped.groupby('Reason').size()}")
            print(f"Total potential trades: {len(trades) + len(skipped)}")
            print(f"Active trades at end: {len(backtester.active_trades)}")

        # Plot trades
        backtester.plot_trades(test_df, period, f'trade_plot_{period.replace(" to ", "_")}.png')

        equity_curve = pd.DataFrame(backtester.equity_curve)
        all_equity_curves.append(equity_curve)

        current_date += timedelta(days=test_window_days)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No valid walk-forward periods found. Try a longer data period or different parameters.")
        return None, None

    avg_metrics = {
        'Avg Cumulative Return': results_df['Cumulative Return'].mean(),
        'Avg Sharpe Ratio': results_df['Sharpe Ratio'].mean(),
        'Avg Max Drawdown': results_df['Max Drawdown'].mean(),
        'Avg Win Rate': results_df['Win Rate'].mean(),
        'Total Trades': results_df['Number of Trades'].sum()
    }

    combined_equity = pd.concat([pd.DataFrame(ec) for ec in all_equity_curves])
    combined_equity = combined_equity.groupby('Date').last().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(combined_equity['Date']), combined_equity['Equity'], label='Walk-Forward Equity Curve')
    plt.title('Walk-Forward Testing Equity Curve (15-Minute Intervals with Candlesticks)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('walk_forward_equity_curve_15min_candlestick_maximized.png')
    plt.close()

    return results_df, avg_metrics

# Main function to run the trading algorithm
def run_trading_algorithm(ticker='AAPL', days=120):
    end_date = datetime.now() - timedelta(days=2)
    start_date = end_date - timedelta(days=days)
    
    df = fetch_data_alpaca(ticker, start_date, end_date, interval=TimeFrame(15, TimeFrameUnit.Minute))
    if df is None or df.empty:
        print("Failed to retrieve data. Exiting.")
        return None, None

    df = add_technical_indicators(df)
    df = df.dropna()
    results_df, avg_metrics = walk_forward_testing(df, train_window_days=30, test_window_days=15)

    if results_df is None:
        return None, None

    print("\nWalk-Forward Testing Results (15-Minute Intervals with Candlesticks):")
    print(results_df[['Period', 'Cumulative Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Number of Trades']])
    print("\nAverage Performance Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nWalk-forward equity curve saved as 'walk_forward_equity_curve_15min_candlestick_maximized.png'")

    return results_df, avg_metrics

# Run the algorithm
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    for ticker in tickers:
        print(f"\nTrying ticker: {ticker}")
        results, avg_metrics = run_trading_algorithm(ticker=ticker, days=120)
        if results is not None:
            break
    if results is None:
        print("All tickers failed. Verify Alpaca API keys, subscription, or try an alternative data source.")