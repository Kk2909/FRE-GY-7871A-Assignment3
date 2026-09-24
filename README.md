# Iran War Risk and Global Financial Markets

This project studies the sensitivity of global financial markets to Iran War Risk from February 28 through September 18, 2026. It combines NLP analysis of news coverage with identification through heteroskedasticity, following Rigobon and Sack's analysis of war risk.

## Methodology

Iran-related news articles are scored for relevance, escalation, de-escalation, and war-news intensity using sentence-transformer embeddings. These article-level measures are aggregated into a daily Iran War Risk Index.

The top 10% of dates are classified as high-news dates and matched with nearby low-news dates. Changes in financial-market variances and covariances between these regimes are used to estimate market sensitivity to a latent war-risk factor.

Because heteroskedasticity identifies factor loadings only up to scale and sign, the factor is normalized to a 10-basis-point increase in the five-year Treasury yield. This reflects the Iran-specific inflation, energy-price, debt-risk, and weaker-flight-to-quality channels described in the assignment guidance.

The project also includes:

- Direct directional NLP regressions
- NLP intensity regressions
- Bad-news, good-news, and low-news regime analysis
- Credit excess-return proxies
- Iran 2026 versus Iraq 2003 comparison
- Novel-language validation sample
- Evaluation of alternative identification methods

## Financial Variables

The analysis includes:

- Five-year and ten-year U.S. Treasury yields
- S&P 500
- Global equity market
- Investment-grade corporate bonds
- High-yield corporate bonds
- Inflation-linked Treasury bonds
- Brent crude oil
- Gold
- U.S. Dollar Index
- VIX
- Investment-grade credit excess-return proxy
- High-yield credit excess-return proxy

The credit variables are calculated as corporate-bond ETF returns minus inflation-linked Treasury ETF returns. They are credit-condition proxies, not observed credit spreads.

## Main Results

The final sample contains:

- 146 daily Iran War Risk observations
- 142 merged news and market dates
- 15 high-news dates
- 15 matched low-news dates
- 30 bad-war-news dates
- 6 good-war-news dates
- 106 no- or low-war-news dates

The five-year Treasury yield variance is approximately 2.35 times higher on high-news dates.

Under normalization to a 10-basis-point increase in the five-year Treasury yield:

- The ten-year Treasury yield increases by approximately 9.05 basis points and is statistically significant.
- Brent oil increases by approximately 5.32%, although the confidence interval includes zero.
- S&P 500 and global equities decline, but the estimates are statistically insignificant.
- Inflation-linked bonds decline significantly.
- VIX increases, but with a wide confidence interval.
- The investment-grade credit proxy weakens, although neither credit proxy is statistically significant.

The direct NLP regressions find that escalation language is significantly associated with a higher ten-year Treasury yield and higher Brent oil prices. Greater news intensity is also significantly associated with larger absolute Brent oil movements.

## Main Files

- `notebooks/Assignment3_Iran_War_Risk.ipynb`: final executed analysis and results
- `src/`: news processing, NLP, econometric estimation, and output scripts
- `data/processed/`: processed news and market datasets
- `outputs/tables/`: final tables and estimation results
- `outputs/figures/`: final figures
- `outputs/tables/ai_methodology_evaluation.md`: evaluation of alternative identification methods
- `outputs/tables/nlp_validation_sample.csv`: stratified headline-validation sample

## Reproduction

Create and activate a Python environment, install the requirements, and run:

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
python src/14_three_regime_analysis.py
python src/15_nlp_validation.py
python src/13_build_final_notebook.py
jupyter nbconvert --to notebook --execute notebooks/Assignment3_Iran_War_Risk.ipynb --inplace --ExecutePreprocessor.timeout=600
```

## Interpretation

Identification through heteroskedasticity is the primary replication method. Direct NLP regressions provide directional evidence using the full daily sample, while the three-regime analysis is an exploratory extension.

The results should be interpreted as conditional market sensitivities under the model's assumptions rather than unconditional proof that Iran War Risk caused every observed financial-market movement.

## References

- Rigobon, R. (2003). Identification Through Heteroskedasticity.
- Rigobon, R., and Sack, B. (2005). The Effects of War Risk on U.S. Financial Markets.
- Caldara, D., and Iacoviello, M. (2022). Measuring Geopolitical Risk.