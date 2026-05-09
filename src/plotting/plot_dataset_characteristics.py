"""Plot dataset characteristic figures for Section 4.2 in the paper."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _select_representative_window(df: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """Select a representative time window whose statistics are close to full-season statistics."""
    steps_per_day = int((24 * 60) / 5)
    window_steps = window_days * steps_per_day
    if len(df) <= window_steps:
        return df.copy()

    cl = df["CL"]
    twb = df["Twb"]

    cl_mean_global = float(cl.mean())
    cl_std_global = float(cl.std(ddof=0))
    twb_mean_global = float(twb.mean())
    twb_std_global = float(twb.std(ddof=0))

    rolling = pd.DataFrame(
        {
            "cl_mean": cl.rolling(window_steps).mean(),
            "cl_std": cl.rolling(window_steps).std(ddof=0),
            "twb_mean": twb.rolling(window_steps).mean(),
            "twb_std": twb.rolling(window_steps).std(ddof=0),
        }
    )

    eps = 1e-6
    score = (
        (rolling["cl_mean"] - cl_mean_global).abs() / (abs(cl_mean_global) + eps)
        + (rolling["cl_std"] - cl_std_global).abs() / (abs(cl_std_global) + eps)
        + (rolling["twb_mean"] - twb_mean_global).abs() / (abs(twb_mean_global) + eps)
        + (rolling["twb_std"] - twb_std_global).abs() / (abs(twb_std_global) + eps)
    )

    valid_score = score.iloc[window_steps - 1 :]
    best_end_idx = int(valid_score.idxmin())
    best_start_idx = best_end_idx - window_steps + 1
    return df.iloc[best_start_idx : best_end_idx + 1].reset_index(drop=True)


def plot_dataset_overview(df: pd.DataFrame, output_path: Path) -> None:
    """Plot long-horizon trends and intra-day statistics for CL and Twb."""
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), dpi=300, constrained_layout=True)

    rep_df = _select_representative_window(df, window_days=14)

    # Use a 2-hour centered moving average to improve readability in representative-window curves.
    smooth_window = 24
    cl_smooth = rep_df["CL"].rolling(window=smooth_window, center=True, min_periods=1).mean()
    twb_smooth = rep_df["Twb"].rolling(window=smooth_window, center=True, min_periods=1).mean()

    # Subplot (a): representative-period trends.
    ax0 = axes[0]
    ax0.plot(rep_df["time"], cl_smooth, color="#1d4e89", linewidth=1.3, alpha=0.95, label="Cooling Load (CL)")
    ax0.set_ylabel("CL (kW)")
    ax0.set_title("(a) Representative 14-day time series", loc="left", fontsize=11, pad=6)
    ax0.grid(alpha=0.25, linestyle="--")
    ax0.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax0_t = ax0.twinx()
    ax0_t.plot(rep_df["time"], twb_smooth, color="#b23a48", linewidth=1.1, alpha=0.85, label="Wet-bulb Temp (Twb)")
    ax0_t.set_ylabel("Twb (degC)")

    lines, labels = ax0.get_legend_handles_labels()
    lines_t, labels_t = ax0_t.get_legend_handles_labels()
    ax0.legend(
        lines + lines_t,
        labels + labels_t,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    # Subplot (b): average intra-day profile.
    ax1 = axes[1]
    hour_index = df["time"].dt.hour + df["time"].dt.minute / 60.0
    profile = (
        df.assign(hour=hour_index)
        .groupby("hour")[["CL", "Twb"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    hour = profile["hour"].to_numpy()
    cl_mean = profile[("CL", "mean")].to_numpy()

    ax1.plot(hour, cl_mean, color="#1d4e89", linewidth=2.0, label="CL mean")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("CL (kW)")
    ax1.set_title("(b) Intra-day mean profile", loc="left", fontsize=11, pad=6)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.set_xlim(0, 24)

    ax1_t = ax1.twinx()
    twb_mean = profile[("Twb", "mean")].to_numpy()
    ax1_t.plot(hour, twb_mean, color="#b23a48", linewidth=1.8, label="Twb mean")
    ax1_t.set_ylabel("Twb (degC)")

    lines, labels = ax1.get_legend_handles_labels()
    lines_t, labels_t = ax1_t.get_legend_handles_labels()
    ax1.legend(
        lines + lines_t,
        labels + labels_t,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_and_joint_pattern(df: pd.DataFrame, output_path: Path) -> None:
    """Plot marginal distributions and CL-Twb joint pattern."""
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), dpi=300, constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # (a) CL distribution
    ax_a.hist(df["CL"], bins=45, color="#2f6690", alpha=0.85, edgecolor="white")
    ax_a.set_title("(a) Distribution of cooling load (CL)", loc="left", fontsize=11)
    ax_a.set_xlabel("CL (kW)")
    ax_a.set_ylabel("Frequency")
    ax_a.grid(alpha=0.2, linestyle="--")

    # (b) Twb distribution
    ax_b.hist(df["Twb"], bins=35, color="#b23a48", alpha=0.82, edgecolor="white")
    ax_b.set_title("(b) Distribution of wet-bulb temperature (Twb)", loc="left", fontsize=11)
    ax_b.set_xlabel("Twb (degC)")
    ax_b.set_ylabel("Frequency")
    ax_b.grid(alpha=0.2, linestyle="--")

    # (c) CL-Twb dependence
    hb = ax_c.hexbin(df["Twb"], df["CL"], gridsize=40, cmap="YlGnBu", mincnt=1)
    ax_c.set_title("(c) Joint pattern: Twb vs CL", loc="left", fontsize=11)
    ax_c.set_xlabel("Twb (degC)")
    ax_c.set_ylabel("CL (kW)")
    cb = fig.colorbar(hb, ax=ax_c)
    cb.set_label("Count")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path("d:/code/projects/Event-DQN")
    data_path = root / "data" / "raw_data.csv"
    output_dir = root / "docs" / "pics"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(data_path)

    out1 = output_dir / "fig4_dataset_temporal_characteristics.svg"
    out2 = output_dir / "fig5_dataset_distribution_and_correlation.svg"

    plot_dataset_overview(df, out1)
    plot_distribution_and_joint_pattern(df, out2)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
