from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/estimation_sample.csv")
OUTPUT_PATH = Path("outputs/tables/war_risk_estimates.csv")

ANCHOR = "d_five_year_yield"
ANCHOR_SHOCK = 0.10
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

TARGETS = {
    "d_ten_year_yield": "10-Year Treasury Yield",
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


def calculate_estimates(
    high: pd.DataFrame,
    low: pd.DataFrame,
    target: str,
) -> dict:
    x_high = high[ANCHOR].to_numpy(dtype=float)
    x_low = low[ANCHOR].to_numpy(dtype=float)

    y_high = high[target].to_numpy(dtype=float)
    y_low = low[target].to_numpy(dtype=float)

    # Center each variable within its volatility regime.
    x_high = x_high - x_high.mean()
    x_low = x_low - x_low.mean()
    y_high = y_high - y_high.mean()
    y_low = y_low - y_low.mean()

    variance_x_high = np.mean(x_high**2)
    variance_x_low = np.mean(x_low**2)

    variance_y_high = np.mean(y_high**2)
    variance_y_low = np.mean(y_low**2)

    covariance_high = np.mean(x_high * y_high)
    covariance_low = np.mean(x_low * y_low)

    delta_variance_x = (
        variance_x_high - variance_x_low
    )

    delta_variance_y = (
        variance_y_high - variance_y_low
    )

    delta_covariance = (
        covariance_high - covariance_low
    )

    estimator_1 = (
        delta_covariance / delta_variance_x
        if abs(delta_variance_x) > 1e-12
        else np.nan
    )

    estimator_2 = (
        delta_variance_y / delta_covariance
        if abs(delta_covariance) > 1e-12
        else np.nan
    )

    x = np.concatenate([x_high, x_low])
    y = np.concatenate([y_high, y_low])

    regime_sign = np.concatenate(
        [
            np.ones(len(x_high)),
            -np.ones(len(x_low)),
        ]
    )

    instrument_1 = regime_sign * x
    instrument_2 = regime_sign * y

    instruments = np.column_stack(
        [instrument_1, instrument_2]
    )

    ztz_inverse = np.linalg.pinv(
        instruments.T @ instruments
    )

    xz = x @ instruments
    zy = instruments.T @ y

    denominator = (
        xz @ ztz_inverse @ xz.T
    )

    numerator = (
        xz @ ztz_inverse @ zy
    )

    combined_estimator = (
        numerator / denominator
        if abs(denominator) > 1e-12
        else np.nan
    )

    predicted_variance_change = (
        combined_estimator**2
        * delta_variance_x
    )

    variance_share_high = (
        predicted_variance_change
        / variance_y_high
        if variance_y_high > 0
        else np.nan
    )

    return {
        "estimator_1": estimator_1,
        "estimator_2": estimator_2,
        "combined_estimator": combined_estimator,
        "delta_variance_anchor": delta_variance_x,
        "delta_variance_target": delta_variance_y,
        "delta_covariance": delta_covariance,
        "variance_share_high": variance_share_high,
    }


def bootstrap_effect(
    high: pd.DataFrame,
    low: pd.DataFrame,
    target: str,
    random_generator: np.random.Generator,
) -> tuple[float, float, float]:
    estimates = []

    for _ in range(N_BOOTSTRAP):
        high_sample = high.iloc[
            random_generator.integers(
                0,
                len(high),
                len(high),
            )
        ]

        low_sample = low.iloc[
            random_generator.integers(
                0,
                len(low),
                len(low),
            )
        ]

        result = calculate_estimates(
            high_sample,
            low_sample,
            target,
        )

        estimate = (
            result["combined_estimator"]
            * ANCHOR_SHOCK
        )

        if np.isfinite(estimate):
            estimates.append(estimate)

    estimates = np.asarray(estimates)

    standard_error = np.std(
        estimates,
        ddof=1,
    )

    confidence_low = np.percentile(
        estimates,
        2.5,
    )

    confidence_high = np.percentile(
        estimates,
        97.5,
    )

    return (
        standard_error,
        confidence_low,
        confidence_high,
    )


def main():
    data = pd.read_csv(INPUT_PATH)

    high = data[
        data["regime"] == "high"
    ].copy()

    low = data[
        data["regime"] == "low"
    ].copy()

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    for target, label in TARGETS.items():
        pair_data = data[
            ["regime", ANCHOR, target]
        ].dropna()

        high_pair = pair_data[
            pair_data["regime"] == "high"
        ]

        low_pair = pair_data[
            pair_data["regime"] == "low"
        ]

        estimate = calculate_estimates(
            high_pair,
            low_pair,
            target,
        )

        normalized_effect = (
            estimate["combined_estimator"]
            * ANCHOR_SHOCK
        )

        standard_error, confidence_low, confidence_high = (
            bootstrap_effect(
                high_pair,
                low_pair,
                target,
                random_generator,
            )
        )

        statistically_significant = not (
            confidence_low <= 0 <= confidence_high
        )

        results.append(
            {
                "variable": target,
                "label": label,
                "high_observations": len(high_pair),
                "low_observations": len(low_pair),
                "estimator_1": estimate["estimator_1"],
                "estimator_2": estimate["estimator_2"],
                "combined_estimator": estimate[
                    "combined_estimator"
                ],
                "effect_of_10bp_yield_increase": normalized_effect,
                "bootstrap_standard_error": standard_error,
                "confidence_low": confidence_low,
                "confidence_high": confidence_high,
                "statistically_significant": statistically_significant,
                "variance_share_high": estimate[
                    "variance_share_high"
                ],
                "delta_covariance": estimate[
                    "delta_covariance"
                ],
            }
        )

    results = pd.DataFrame(results)

    results["variance_share_percent"] = (
        100 * results["variance_share_high"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    display = results[
        [
            "label",
            "estimator_1",
            "estimator_2",
            "combined_estimator",
            "effect_of_10bp_yield_increase",
            "confidence_low",
            "confidence_high",
            "statistically_significant",
            "variance_share_percent",
        ]
    ].copy()

    numeric_columns = display.select_dtypes(
        include="number"
    ).columns

    display[numeric_columns] = (
        display[numeric_columns].round(4)
    )

    print(
        "Estimated response to a 10-basis-point "
        "increase in the five-year Treasury yield:\n"
    )

    print(display.to_string(index=False))

    print(
        "\nYield effects are measured in percentage points. "
        "All other effects are percentage log returns."
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()