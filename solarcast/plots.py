import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colours follow the validated categorical palette (fixed order, never cycled).
# Aqua and yellow are below 3:1 contrast on the light surface, so every bar chart carries value labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dc", "#fcfcfb"

def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

def rmse_by_horizon(metrics: pd.DataFrame, models: list[str], ax=None):
    # Grouped bars: test RMSE per horizon for up to four models, value-labelled
    ax = ax or plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)[1]
    horizons = list(dict.fromkeys(metrics["Horizon"]))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = [metrics.loc[(metrics["Horizon"] == h) & (metrics["Model"] == model), "RMSE (W/m^2)"].iloc[0] for h in horizons]
        x = np.arange(len(horizons)) + (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x, vals, width * 0.92, color=SERIES[i], label=model)
        ax.bar_label(bars, fmt="%.0f", padding=2, color=INK, fontsize=8)
    labels = [f"{h}\n({metrics.loc[metrics['Horizon'] == h, 'Forecast inputs'].iloc[0]})" if "Forecast inputs" in metrics else h for h in horizons]
    ax.set_xticks(range(len(horizons)), labels)
    _style(ax, "Test RMSE by horizon (lower is better)", "RMSE (W/m^2)")
    ax.legend(frameon=False, ncols=len(models), loc="upper left", bbox_to_anchor=(0, -0.18))
    return ax

def week_timeseries(frame: pd.DataFrame, series: list[str], title: str, ax=None):
    # frame: index = target time, column "Truth" plus the series to overlay
    ax = ax or plt.subplots(figsize=(11, 4), facecolor=SURFACE)[1]
    ax.plot(frame.index, frame["Truth"], color=INK, linewidth=2, label="NSRDB truth")
    for i, name in enumerate(series):
        ax.plot(frame.index, frame[name], color=SERIES[i], linewidth=2, label=name)
    _style(ax, title, "GHI (W/m^2)")
    ax.legend(frameon=False, ncols=len(series) + 1, loc="upper left", bbox_to_anchor=(0, -0.12))
    return ax

def feature_importance(model, top: int = 15, ax=None):
    # XGBoost gain summed over lags, so each base feature appears once
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    base = gain.groupby(lambda name: re.sub(r"_lag\d+$", "", name)).sum()
    base = (base / base.sum()).sort_values().tail(top)
    ax = ax or plt.subplots(figsize=(8, 5), facecolor=SURFACE)[1]
    bars = ax.barh(base.index, base.values, color=SERIES[0], height=0.7)
    ax.bar_label(bars, fmt="%.2f", padding=2, color=INK, fontsize=8)
    _style(ax, "XGBoost feature importance (share of total gain)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return ax
