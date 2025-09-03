## MODULES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.dates import DateFormatter
from src.python.parse import read_data


# Visualize the trend
def viz_trend(sub_tbl):
    # Crosstab
    ct = pd.crosstab(
        index=sub_tbl["pubDateTime"].dt.date,
        columns=[sub_tbl["is_unrest"], sub_tbl["is_violent"]]
    )

    # Relabel columns and order for legend
    label_map = {
        (True, True): "Unrest | Violence",
        (True, False): "Unrest | No Violence",
        (False, True): "No Unrest | Violence",
        (False, False): "No Unrest | No Violence"
    }
    ct = ct[[ (True, True), (True, False), (False, True), (False, False) ]]  # ensure order
    ct.columns = [label_map[col] for col in ct.columns]

    # Melt for plotting
    df_long = ct.reset_index().melt(
        id_vars="pubDateTime",
        var_name="Condition",
        value_name="Count"
    )
    df_long["pubDateTime"] = pd.to_datetime(df_long["pubDateTime"])

    # Apply cubic spline smoothing
    smoothed_list = []
    for cond, group in df_long.groupby("Condition"):
        group = group.sort_values("pubDateTime")
        x = np.arange(len(group))
        y = group["Count"].values
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_smooth = make_interp_spline(x, y, k=2)(x_smooth)

        # Map back to approximate dates
        dates_smooth = np.interp(x_smooth, x, group["pubDateTime"].map(pd.Timestamp.timestamp))
        dates_smooth = pd.to_datetime(dates_smooth, unit='s')

        tmp = pd.DataFrame({
            "pubDateTime": dates_smooth,
            "Condition": cond,
            "Smoothed": y_smooth
        })
        # Keep original points for dot/annotation
        tmp["Count"] = np.interp(x_smooth, x, y)
        smoothed_list.append(tmp)

    smooth_df = pd.concat(smoothed_list)

    # Set theme
    sns.set_theme(style="white", context="talk")

    # Define colors: Unrest | Violence red, others light gray
    color_map = {
        "Unrest | Violence": "#E10600",  # red
        "Unrest | No Violence": "#8C8C8C",
        "No Unrest | Violence": "#BFBFBF",
        "No Unrest | No Violence": "#464646"
    }

    # Plot
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=smooth_df,
        x="pubDateTime", y="Smoothed",
        hue="Condition",
        palette=color_map,
        linewidth=2.5
    )

    # Add dot only for 'Unrest | Violence' at the end
    for cond in ["Unrest | Violence", "No Unrest | No Violence"]:
        group = smooth_df[smooth_df["Condition"] == cond]
        last_point = group.iloc[-1]
        plt.scatter(
            last_point["pubDateTime"],
            last_point["Smoothed"],
            s=500,  # larger dot
            color=color_map[cond],
            zorder=10
        )
        plt.text(
            last_point["pubDateTime"],
            last_point["Smoothed"],
            str(int(last_point["Count"])),
            color="white",
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
            zorder=11
        )

    # Titles and subtitle
    plt.title(
        "Trends of Political Unrest & Violence Reported in News Articles",
        loc="left", fontsize=18, weight="bold", pad=20
    )

    # Axis labels
    plt.xlabel("Ongoing protests and demonstrations in 2025", fontsize=12)
    plt.ylabel("Number of News Articles", fontsize=12)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(DateFormatter('%d %b'))
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)

    # Legend top-right
    plt.legend(title="", loc="upper right", frameon=False, fontsize=11)

    # Remove gridlines
    sns.despine(left=True, bottom=True)
    plt.grid(False)
    plt.tight_layout()

    return plt
