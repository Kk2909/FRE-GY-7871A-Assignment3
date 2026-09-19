# Iran War Risk and Global Financial Markets

This project studies the sensitivity of financial markets to Iran War Risk from January 1 through September 18, 2026. It combines NLP analysis of news coverage with identification through heteroskedasticity, following Rigobon and Sack’s analysis of war risk.

## Methodology

Iran-related news articles are scored for relevance, escalation, de-escalation, and war-news intensity using sentence embeddings and conflict-related language. These scores are aggregated into a daily Iran War Risk Index.

High-news days are matched with nearby low-news days. Changes in market volatility between the two regimes are then used to identify the financial-market response to an increase in war risk.

A direct NLP regression is also estimated as an alternative approach using all available daily observations.

## Financial Variables

The analysis includes:

- Five-year and ten-year U.S. Treasury yields
- S&P 500
- Global equity market
- Investment-grade, high-yield, and inflation-linked bonds
- Brent crude oil
- Gold
- U.S. Dollar Index
- VIX

## Main Files

- `notebooks/Assignment3_Iran_War_Risk.ipynb`: final analysis and results
- `src/`: data collection, NLP, econometric estimation, and output scripts
- `data/processed/`: processed news and market datasets
- `outputs/tables/`: final tables and estimation results
- `outputs/figures/`: final figures

## Reproduction

Create and activate a Python environment, install the requirements, and run the scripts in numerical order:

```powershell
python -m pip install -r requirements.txt
python src/01_collect_iran_news.py
python src/02_clean_news.py
python src/03_score_war_risk.py
python src/04_build_daily_risk_index.py
python src/05_market_data.py
python src/06_build_estimation_sample.py
python src/07_check_variance_regimes.py
python src/08_estimate_war_risk_effects.py
python src/09_create_figures.py
python src/10_create_regime_tables.py
python src/11_nlp_regression.py
python src/12_create_tables2_3.py
python src/13_build_final_notebook.py

```

## References

- Rigobon, R. (2003). Identification Through Heteroskedasticity.
- Rigobon, R., and Sack, B. (2005). The Effects of War Risk on U.S. Financial Markets.