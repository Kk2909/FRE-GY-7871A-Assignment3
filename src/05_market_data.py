from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2025-12-15"
END_DATE = "2026-09-20"

LEVELS_PATH = Path("data/raw/market_levels.csv")
CHANGES_PATH = Path("data/processed/market_changes.csv")

YIELD_SERIES = {
    "^FVX": "five_year_yield",
    "^TNX": "ten_year_yield",
}

PRICE_SERIES = {
    "^GSPC": "sp500",
    "ACWI": "global_equity",
    "LQD": "investment_grade_bonds",
    "HYG": "high_yield_bonds",
    "TIP": "inflation_linked_bonds",
    "BZ=F": "brent_oil",
    "GC=F": "gold",
    "DX-Y.NYB": "dollar_index",
    "^VIX": "vix",
}


def download_series(
    ticker: str,
    column_name: str,
) -> pd.Series:
    for attempt in range(1, 4):
        try:
            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
            )

            if data.empty:
                raise ValueError("No observations returned.")

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            price_column = (
                "Adj Close"
                if "Adj Close" in data.columns
                else "Close"
            )

            series = data[price_column].copy()
            series.index = pd.to_datetime(series.index)

            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            series.name = column_name

            return series.dropna().sort_index()

        except Exception as error:
            print(
                f"  Attempt {attempt} failed for "
                f"{ticker}: {error}"
            )
            time.sleep(10 * attempt)

    raise RuntimeError(
        f"Unable to download {ticker}"
    )


def main():
    downloaded_series = []

    all_series = {
        **YIELD_SERIES,
        **PRICE_SERIES,
    }

    print("Downloading Yahoo Finance market data")

    for ticker, column_name in all_series.items():
        series = download_series(
            ticker,
            column_name,
        )

        downloaded_series.append(series)

        print(
            f"  {ticker:<10} "
            f"{column_name:<28} "
            f"{len(series):>4} observations"
        )

        time.sleep(1)

    levels = pd.concat(
        downloaded_series,
        axis=1,
    ).sort_index()

    levels.index.name = "date"

    LEVELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(LEVELS_PATH)

    changes = pd.DataFrame(index=levels.index)

    # Treasury yields are measured in percentage points.
    changes["d_five_year_yield"] = (
        levels["five_year_yield"].diff()
    )

    changes["d_ten_year_yield"] = (
        levels["ten_year_yield"].diff()
    )

    # Prices and market indices use percentage log returns.
    for column in PRICE_SERIES.values():
        changes[f"r_{column}"] = (
            100
            * np.log(
                levels[column]
                / levels[column].shift(1)
            )
        )

    # Corporate bond excess-return proxies for credit conditions.
    # Negative values indicate weaker corporate credit performance
    # relative to inflation-linked Treasury bonds.
    
    changes["r_ig_credit_proxy"] = (
        changes["r_investment_grade_bonds"]
        - changes["r_inflation_linked_bonds"]
    )

    changes["r_hy_credit_proxy"] = (
        changes["r_high_yield_bonds"]
        - changes["r_inflation_linked_bonds"]
    )

    changes = changes.loc[
        "2026-02-28":"2026-09-18"
    ]

    changes.index.name = "date"

    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    changes.to_csv(CHANGES_PATH)

    summary = pd.DataFrame(
        {
            "observations": levels.notna().sum(),
            "first_date": levels.apply(
                lambda column: column.first_valid_index()
            ),
            "last_date": levels.apply(
                lambda column: column.last_valid_index()
            ),
        }
    )

    print("\nMarket level summary:")
    print(summary.to_string())

    print("\nUsable daily changes:")
    print(changes.notna().sum().to_string())

    print(f"\nLevels saved to: {LEVELS_PATH}")
    print(f"Changes saved to: {CHANGES_PATH}")


if __name__ == "__main__":
    main()