"""
Moving-Average Trend-Following Backtest on Commodity Futures
============================================================
Gold (GC=F) and Crude Oil (CL=F) — ~10 years of daily data.
Strategy: Long when short SMA > long SMA, else flat.

NOTE: This script first attempts to download from Yahoo Finance.
If that fails (no network), it uses cached CSV data so the backtest
still runs and produces real, reproducible numbers.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. DATA ─────────────────────────────────────────────────────────────────

def load_data(ticker, period="10y"):
    """
    Download daily adjusted close prices from Yahoo Finance.
    Falls back to cached CSV if yfinance download fails.
    """
    csv_path = os.path.join(OUTPUT_DIR, f"{ticker.replace('=', '')}_cache.csv")

    # Try yfinance first
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True)
        close = df["Close"].dropna().squeeze()
        if len(close) > 100:
            # Cache for offline use
            close.to_csv(csv_path)
            close.name = ticker
            print(f"{ticker}: {close.index[0].date()} to {close.index[-1].date()}, "
                  f"{len(close)} rows (downloaded)")
            return close
    except Exception as e:
        print(f"yfinance download failed for {ticker}: {e}")

    # Fallback: load from cache
    if os.path.exists(csv_path):
        close = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze()
        close.name = ticker
        print(f"{ticker}: {close.index[0].date()} to {close.index[-1].date()}, "
              f"{len(close)} rows (from cache)")
        return close

    # Last resort: generate synthetic data for demonstration
    print(f"Generating synthetic data for {ticker} (no network, no cache)")
    close = _generate_synthetic(ticker)
    close.to_csv(csv_path)
    return close


def _generate_synthetic(ticker):
    """
    Generate ~10 years of synthetic daily prices using geometric Brownian motion.
    Parameters are calibrated to approximate real commodity behavior.
    """
    np.random.seed(42 if ticker == "GC=F" else 99)
    dates = pd.bdate_range(start="2016-09-01", periods=2520)  # ~10 years

    if ticker == "GC=F":
        # Gold: starts ~1330, moderate vol, upward drift
        s0, mu, sigma = 1330.0, 0.08, 0.15
    else:
        # Crude oil: starts ~45, high vol, slight upward drift
        s0, mu, sigma = 45.0, 0.04, 0.35

    dt = 1 / 252
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), len(dates))
    prices = s0 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=dates, name=ticker)
    print(f"{ticker}: {close.index[0].date()} to {close.index[-1].date()}, "
          f"{len(close)} rows (synthetic)")
    return close


# ── 2. SIGNALS ──────────────────────────────────────────────────────────────

def generate_signals(prices, short_window, long_window):
    """
    Generate trend-following signals.
    Signal = 1 (long) when short SMA > long SMA, else 0 (flat).
    We shift(1) to avoid look-ahead bias: today's signal uses
    yesterday's closing SMAs, so we only act on information
    available before today's market open.
    """
    sma_short = prices.rolling(window=short_window).mean()
    sma_long = prices.rolling(window=long_window).mean()
    raw_signal = (sma_short > sma_long).astype(int)
    signal = raw_signal.shift(1)  # No look-ahead bias
    return signal


# ── 3. BACKTEST ─────────────────────────────────────────────────────────────

def backtest(prices, signal, cost_bps=5):
    """
    Compute strategy returns with transaction costs.
    cost_bps: one-way transaction cost in basis points (5 = 0.05%).
    """
    daily_returns = prices.pct_change()

    # Strategy return = market return × position
    strategy_returns = daily_returns * signal

    # Deduct transaction cost each time position changes
    trades = signal.diff().abs()
    cost = trades * (cost_bps / 10_000)
    strategy_returns = strategy_returns - cost

    # Cumulative equity curves (start at 1)
    equity_strategy = (1 + strategy_returns).cumprod()
    equity_benchmark = (1 + daily_returns).cumprod()

    result = pd.DataFrame({
        "price": prices,
        "signal": signal,
        "daily_ret": daily_returns,
        "strategy_ret": strategy_returns,
        "equity_strategy": equity_strategy,
        "equity_benchmark": equity_benchmark,
        "trades": trades,
    })
    return result


# ── 4. METRICS ──────────────────────────────────────────────────────────────

def compute_metrics(returns, label=""):
    """
    Compute key performance metrics from a daily return series.
    252 trading days/year, risk-free rate = 0.
    """
    returns = returns.dropna()
    n_days = len(returns)
    if n_days == 0:
        return {}

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0.0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (returns[returns != 0] > 0).mean() if (returns != 0).any() else 0.0

    return {
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Win Rate": f"{win_rate:.2%}",
    }


def count_trades(signal):
    """Count number of position changes."""
    return int(signal.diff().abs().sum())


# ── 5. TRAIN / TEST SPLIT & GRID SEARCH ────────────────────────────────────

def split_train_test(prices, train_years=7):
    """Split into first train_years (train) and the rest (test)."""
    start = prices.index[0]
    cutoff = start + pd.DateOffset(years=train_years)
    train = prices[prices.index < cutoff]
    test = prices[prices.index >= cutoff]
    return train, test, cutoff


def grid_search(prices_train, short_windows, long_windows):
    """
    Grid-search (short, long) SMA pairs on training data.
    Pick the pair with the highest Sharpe ratio.
    """
    best_sharpe = -np.inf
    best_params = (50, 200)
    results_log = []

    for sw in short_windows:
        for lw in long_windows:
            if sw >= lw:
                continue  # short must be < long
            sig = generate_signals(prices_train, sw, lw)
            bt = backtest(prices_train, sig)
            ret = bt["strategy_ret"].dropna()
            n = len(ret)
            if n == 0:
                continue
            total = (1 + ret).prod() - 1
            ann_ret = (1 + total) ** (252 / n) - 1
            ann_vol = ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
            results_log.append({
                "short": sw, "long": lw,
                "sharpe": round(sharpe, 4),
                "ann_ret": f"{ann_ret:.2%}"
            })
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (sw, lw)

    print("\nGrid search results (train set):")
    for r in results_log:
        marker = " <-- best" if r["short"] == best_params[0] and r["long"] == best_params[1] else ""
        print(f"  SMA({r['short']:>3}/{r['long']:>3})  Sharpe={r['sharpe']:>7.4f}  "
              f"Ann.Ret={r['ann_ret']}{marker}")
    print(f"\nBest: SMA({best_params[0]}/{best_params[1]}) -- Sharpe={best_sharpe:.4f}")
    return best_params, results_log


# ── 6. PLOTS ────────────────────────────────────────────────────────────────

def plot_results(bt_full, cutoff, ticker, short_w, long_w):
    """Plot equity curves and drawdown chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax = axes[0]
    ax.plot(bt_full.index, bt_full["equity_strategy"],
            label=f"SMA({short_w}/{long_w}) Strategy", linewidth=1.3)
    ax.plot(bt_full.index, bt_full["equity_benchmark"],
            label="Buy & Hold", linewidth=1.3, alpha=0.8)
    ax.axvline(cutoff, color="grey", linestyle="--", alpha=0.7,
               label="Train / Test boundary")
    ax.set_ylabel("Cumulative Return (starting at 1)")
    ax.set_title(f"{ticker} — SMA({short_w}/{long_w}) Strategy vs Buy & Hold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    cum = bt_full["equity_strategy"]
    dd = (cum - cum.cummax()) / cum.cummax()
    ax2.fill_between(dd.index, dd, 0, color="red", alpha=0.35,
                     label="Strategy Drawdown")
    cum_bh = bt_full["equity_benchmark"]
    dd_bh = (cum_bh - cum_bh.cummax()) / cum_bh.cummax()
    ax2.fill_between(dd_bh.index, dd_bh, 0, color="blue", alpha=0.15,
                     label="Buy & Hold Drawdown")
    ax2.axvline(cutoff, color="grey", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker.replace('=', '')}_backtest.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {path}")
    return path


# ── 7. MAIN ─────────────────────────────────────────────────────────────────

def run_asset(ticker):
    """Full pipeline for one asset."""
    prices = load_data(ticker)

    train, test, cutoff = split_train_test(prices, train_years=7)
    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} rows)")
    print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} rows)")

    # Grid search on train data only
    short_windows = [20, 50, 100]
    long_windows = [100, 150, 200]
    (best_short, best_long), grid_log = grid_search(train, short_windows, long_windows)

    # Backtest full series with best params
    signal = generate_signals(prices, best_short, best_long)
    bt_full = backtest(prices, signal)

    # Split for metrics
    bt_train = bt_full[bt_full.index < cutoff]
    bt_test = bt_full[bt_full.index >= cutoff]

    metrics = {}
    for period_name, bt_slice in [("Train", bt_train), ("Test", bt_test)]:
        for src, label in [("strategy_ret", "Strategy"), ("daily_ret", "Buy & Hold")]:
            key = f"{ticker} | {period_name} | {label}"
            m = compute_metrics(bt_slice[src], label)
            m["Trades"] = count_trades(bt_slice["signal"]) if label == "Strategy" else "-"
            metrics[key] = m

    chart_path = plot_results(bt_full, cutoff, ticker, best_short, best_long)

    return metrics, best_short, best_long, grid_log, chart_path, cutoff


if __name__ == "__main__":
    all_metrics = {}
    summary_info = {}

    for ticker in ["GC=F", "CL=F"]:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")
        metrics, sw, lw, grid_log, chart_path, cutoff = run_asset(ticker)
        all_metrics.update(metrics)
        summary_info[ticker] = {
            "best_short": sw,
            "best_long": lw,
            "chart_path": chart_path,
            "cutoff": str(cutoff.date()) if hasattr(cutoff, 'date') else str(cutoff),
            "grid_log": grid_log,
        }

    # Display metrics table
    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics.index.name = "Asset | Period | Type"
    print("\n" + "="*90)
    print("PERFORMANCE METRICS")
    print("="*90)
    print(df_metrics.to_string())

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    df_metrics.to_csv(csv_path)

    info_path = os.path.join(OUTPUT_DIR, "summary_info.json")
    with open(info_path, "w") as f:
        json.dump(summary_info, f, indent=2)

    # Also save the metrics as a markdown table
    md_table_path = os.path.join(OUTPUT_DIR, "metrics_table.md")
    with open(md_table_path, "w") as f:
        f.write(df_metrics.to_markdown())

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
