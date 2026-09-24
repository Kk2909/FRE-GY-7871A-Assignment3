from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RISK_PATH = Path("data/processed/daily_iran_war_risk.csv")
VARIANCE_PATH = Path(
    "outputs/tables/variance_regime_diagnostics.csv"
)
ESTIMATES_PATH = Path(
    "outputs/tables/war_risk_estimates.csv"
)

FIGURE_DIRECTORY = Path("outputs/figures")


def prepare_style():
    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 13,
            "font.family": "sans-serif",
        }
    )


def plot_risk_index():
    risk = pd.read_csv(
        RISK_PATH,
        parse_dates=["market_date"],
    )

    high = risk[risk["high_news_day"]].copy()

    fig, axis = plt.subplots(
        figsize=(12, 5),
    )

    axis.plot(
        risk["market_date"],
        risk["war_risk_index"],
        linewidth=1.4,
        label="Daily Iran War Risk Index",
    )

    axis.scatter(
        high["market_date"],
        high["war_risk_index"],
        color="crimson",
        s=35,
        zorder=3,
        label="High-news day",
    )

    axis.axhline(
        risk["war_risk_index"].quantile(0.90),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="90th percentile",
    )

    axis.set_title(
        "Daily Iran War Risk Index in 2026"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Standardized risk index")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY / "daily_war_risk_index.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_variance_ratios():
    diagnostics = pd.read_csv(VARIANCE_PATH)

    labels = {
        "d_five_year_yield": "5-Year Yield",
        "d_ten_year_yield": "10-Year Yield",
        "r_sp500": "S&P 500",
        "r_global_equity": "Global Equity",
        "r_investment_grade_bonds": "Investment-Grade Bonds",
        "r_high_yield_bonds": "High-Yield Bonds",
        "r_inflation_linked_bonds": "Inflation-Linked Bonds",
        "r_brent_oil": "Brent Oil",
        "r_gold": "Gold",
        "r_dollar_index": "U.S. Dollar Index",
        "r_vix": "VIX",
        "r_ig_credit_proxy": "IG Credit Excess-Return Proxy",
        "r_hy_credit_proxy": "HY Credit Excess-Return Proxy",
    }

    diagnostics["label"] = (
        diagnostics["variable"].map(labels)
    )

    diagnostics = diagnostics.sort_values(
        "variance_ratio",
        ascending=True,
    )

    colors = np.where(
        diagnostics["variance_ratio"] >= 1,
        "#305C89",
        "#A6A6A6",
    )

    fig, axis = plt.subplots(
        figsize=(9, 6),
    )

    axis.barh(
        diagnostics["label"],
        diagnostics["variance_ratio"],
        color=colors,
    )

    axis.axvline(
        1,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axis.set_title(
        "Market Variance on High-News Relative to Low-News Days"
    )
    axis.set_xlabel("Variance ratio")
    axis.set_ylabel("")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY / "variance_ratios.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_market_effects():
    estimates = pd.read_csv(ESTIMATES_PATH)

    effects = estimates[
        estimates["variable"] != "d_ten_year_yield"
    ].copy()

    effects = effects.sort_values(
        "effect_of_10bp_yield_increase"
    )

    estimate = effects[
        "effect_of_10bp_yield_increase"
    ].to_numpy()

    lower_error = np.maximum(
        estimate
        - effects["confidence_low"].to_numpy(),
        0,
    )

    upper_error = np.maximum(
        effects["confidence_high"].to_numpy()
        - estimate,
        0,
    )

    significant = (
        effects["statistically_significant"]
        .astype(bool)
        .to_numpy()
    )

    colors = np.where(
        significant,
        "#A61B29",
        "#4C78A8",
    )

    fig, axis = plt.subplots(
        figsize=(10, 6),
    )

    positions = np.arange(len(effects))

    axis.errorbar(
        estimate,
        positions,
        xerr=np.vstack(
            [lower_error, upper_error]
        ),
        fmt="none",
        ecolor="#555555",
        capsize=3,
        linewidth=1.2,
    )

    axis.scatter(
        estimate,
        positions,
        color=colors,
        s=55,
        zorder=3,
    )

    axis.axvline(
        0,
        color="black",
        linewidth=1,
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(effects["label"])

    axis.set_title(
        "Estimated Market Response to an Iran War Risk Shock"
    )
    axis.set_xlabel(
        "Percent response to a 10 bp decline in the 5-year yield"
    )
    axis.set_ylabel("")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY / "estimated_market_effects.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_variance_explained():
    estimates = pd.read_csv(ESTIMATES_PATH)

    estimates = estimates.sort_values(
        "variance_share_percent",
        ascending=True,
    )

    fig, axis = plt.subplots(
        figsize=(9, 6),
    )

    axis.barh(
        estimates["label"],
        estimates["variance_share_percent"],
        color="#3A6EA5",
    )

    axis.set_title(
        "Variance Attributable to Iran War Risk"
    )
    axis.set_xlabel(
        "Share of high-news-day variance (%)"
    )
    axis.set_ylabel("")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY / "variance_explained.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_style()
    plot_risk_index()
    plot_variance_ratios()
    plot_market_effects()
    plot_variance_explained()

    print("Figures created:")
    print(
        FIGURE_DIRECTORY
        / "daily_war_risk_index.png"
    )
    print(
        FIGURE_DIRECTORY
        / "variance_ratios.png"
    )
    print(
        FIGURE_DIRECTORY
        / "estimated_market_effects.png"
    )
    print(
        FIGURE_DIRECTORY
        / "variance_explained.png"
    )


if __name__ == "__main__":
    main()