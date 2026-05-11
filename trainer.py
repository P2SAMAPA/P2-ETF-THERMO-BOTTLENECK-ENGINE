"""
Main trainer: for each universe, test three windows, select best (max correlation of collapse score with future drawdown),
then compute current collapse warning.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from bottleneck_analyzer import ThermoBottleneck
import push_results

def compute_future_drawdown(returns, lookahead):
    """Drawdown over next `lookahead` days."""
    cum_ret = np.exp(np.cumsum(returns))
    future_min = np.minimum.accumulate(cum_ret)
    drawdown = (future_min - cum_ret) / cum_ret
    return drawdown

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df)
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        # For each window, compute time series of collapse score and correlation with future drawdown
        best_window = None
        best_corr = -np.inf
        best_metrics = None

        for win in config.WINDOWS:
            print(f"  Testing window = {win} days")
            lambda_series = []
            drawdown_series = []
            # We'll loop over available dates to compute rolling score
            # Limit to last 5 years to speed up
            total_len = len(returns)
            start_idx = max(0, total_len - 5*252)   # last 5 years
            for i in range(start_idx + win, total_len - config.LOOKAHEAD):
                # For each ticker? We need a market proxy – use average of all tickers for now
                # But collapse score should be per ETF? The engine computes per ETF, then we aggregate.
                # To choose window, we use the average collapse score across tickers.
                scores = []
                for ticker in tickers:
                    if ticker not in returns.columns:
                        continue
                    ret_series = returns[ticker].iloc[i-win:i].values
                    macro_aligned = macro.loc[returns.index[i-win:i]]
                    if macro_aligned.isna().any().any():
                        continue
                    mac_vals = macro_aligned.values
                    if len(ret_series) < win or len(mac_vals) < win:
                        continue
                    tb = ThermoBottleneck(window=win,
                                          latent_dim=config.VIB_LATENT_DIM,
                                          beta=config.VIB_BETA,
                                          te_lag=config.TE_LAG,
                                          mep_bins=config.MEP_BINS)
                    score = tb.compute_metrics(ret_series, mac_vals)
                    if score is not None:
                        scores.append(score)
                if scores:
                    avg_score = np.mean(scores)
                    lambda_series.append(avg_score)
                    # Future drawdown over next LOOKAHEAD days for the equal‑weighted portfolio
                    port_returns = returns.iloc[i:i+config.LOOKAHEAD].mean(axis=1).values
                    drawdown = compute_future_drawdown(port_returns, config.LOOKAHEAD)
                    drawdown_series.append(np.min(drawdown))   # most negative drawdown
            if len(lambda_series) < 10:
                continue
            corr = np.corrcoef(lambda_series, drawdown_series)[0,1]
            print(f"    Correlation with {config.LOOKAHEAD}-day drawdown: {corr:.3f}")
            if abs(corr) > best_corr:
                best_corr = abs(corr)
                best_window = win
                # Re‑fit on the last full window for final output
                # We'll compute for each ticker separately
                best_metrics = {}
                for ticker in tickers:
                    if ticker not in returns.columns:
                        continue
                    ret_series = returns[ticker].iloc[-win:].values
                    macro_aligned = macro.loc[returns.index[-win:]]
                    if macro_aligned.isna().any().any():
                        continue
                    mac_vals = macro_aligned.values
                    if len(ret_series) < win or len(mac_vals) < win:
                        continue
                    tb = ThermoBottleneck(window=win,
                                          latent_dim=config.VIB_LATENT_DIM,
                                          beta=config.VIB_BETA,
                                          te_lag=config.TE_LAG,
                                          mep_bins=config.MEP_BINS)
                    score = tb.compute_metrics(ret_series, mac_vals)
                    if score is not None:
                        best_metrics[ticker] = {
                            "collapse_score": score,
                            "ib_compression": tb.ib_compression_,
                            "transfer_entropy_macro": tb.tmep_,
                            "mep_rate": tb.mep_rate_
                        }

        if best_window is None:
            print(f"  No valid window for {universe_name}")
            continue

        print(f"  Selected window: {best_window} days (|corr|={best_corr:.3f})")
        # Determine collapse warning for the whole universe: if any ETF exceeds threshold, flag universe
        collapse_warning = any(v["collapse_score"] > config.COLLAPSE_THRESHOLD for v in best_metrics.values())
        # Sort ETFs by collapse score descending (most at risk)
        sorted_etfs = sorted(best_metrics.items(), key=lambda x: x[1]["collapse_score"], reverse=True)
        top_risk = [{"ticker": t, "collapse_score": v["collapse_score"]} for t, v in sorted_etfs[:config.TOP_N_DESTAB]]  # reuse top_n as risk count

        universe_results = {
            "selected_window": best_window,
            "collapse_warning": collapse_warning,
            "threshold": config.COLLAPSE_THRESHOLD,
            "average_collapse_score": np.mean([v["collapse_score"] for v in best_metrics.values()]),
            "collapse_risk_etfs": top_risk,
            "all_tickers": best_metrics
        }
        all_results[universe_name] = universe_results

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/thermo_bottleneck_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Thermo-Bottleneck analysis complete ===")

if __name__ == "__main__":
    main()
