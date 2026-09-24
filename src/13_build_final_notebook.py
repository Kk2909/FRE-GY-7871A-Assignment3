from pathlib import Path

import nbformat as nbf


OUTPUT_PATH = Path(
    "notebooks/Assignment3_Iran_War_Risk.ipynb"
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main():
    notebook = nbf.v4.new_notebook()

    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3.11",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }

    notebook["cells"] = [
        markdown(
            r"""
# Assignment 3: Iran War Risk and Global Financial Markets

This project estimates the sensitivity of global financial variables to Iran War Risk from **February 28, 2026 through September 18, 2026**.

The analysis combines:

1. NLP-based identification of high-war-news dates
2. Identification through heteroskedasticity
3. Direct NLP regressions
4. A three-regime bad-news, good-news, and low-news extension
5. Credit excess-return proxies
6. An evaluation of alternative identification methods
7. Validation considerations for novel conflict language
"""
        ),
        markdown(
            r"""
## Research Question

> How sensitive are global financial variables to changes in Iran War Risk during 2026?

Iran War Risk is not directly observable. News coverage provides information about military escalation, nuclear developments, sanctions, diplomatic negotiations, ceasefires, and threats to regional energy infrastructure.

The primary analysis replicates the framework in *The Effects of War Risk on U.S. Financial Markets*. NLP identifies dates with unusually intense Iran-related information, and changes in financial-market variances and covariances across high- and low-news regimes are used for identification.
"""
        ),
        markdown(
            r"""
## Analysis Structure

The analysis proceeds as follows:

1. Collect and clean Iran-related news.
2. Measure relevance, escalation, de-escalation, and intensity using sentence embeddings.
3. Construct a daily Iran War Risk Index.
4. Identify high-news dates and matched low-news dates.
5. Test whether the variance of the reference variable rises in the high-news regime.
6. Estimate market sensitivities through heteroskedasticity.
7. Normalize the latent factor to a 10-basis-point increase in the five-year Treasury yield.
8. Estimate direct NLP regressions using all overlapping dates.
9. Separate bad-war-news, good-war-news, and low-news regimes.
10. Compare the 2026 Iran setting with the 2003 Iraq setting.
11. Evaluate alternative identification methods and NLP limitations.
"""
        ),
        code(
            """
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 120)
pd.set_option(
    "display.float_format",
    lambda value: f"{value:,.4f}",
)

ROOT = Path.cwd()

if not (ROOT / "data").exists():
    ROOT = ROOT.parent

news = pd.read_csv(
    ROOT / "data/processed/iran_news_scored.csv"
)

daily_risk = pd.read_csv(
    ROOT / "data/processed/daily_iran_war_risk.csv",
    parse_dates=["market_date"],
)

estimation_sample = pd.read_csv(
    ROOT / "data/processed/estimation_sample.csv",
    parse_dates=["date"],
)

market_changes = pd.read_csv(
    ROOT / "data/processed/market_changes.csv",
    parse_dates=["date"],
)

table_1 = pd.read_csv(
    ROOT / "outputs/tables/table1_high_and_low_news_regimes.csv"
)

table_2 = pd.read_csv(
    ROOT / "outputs/tables/table2_estimated_market_effects.csv"
)

table_3 = pd.read_csv(
    ROOT / "outputs/tables/table3_variance_explained.csv"
)

direct_nlp = pd.read_csv(
    ROOT / "outputs/tables/direct_nlp_regressions.csv"
)

variance_diagnostics = pd.read_csv(
    ROOT / "outputs/tables/variance_regime_diagnostics.csv"
)

appendix_daily = pd.read_csv(
    ROOT / "outputs/tables/appendix_all_daily_war_risk.csv"
)

three_regime_summary = pd.read_csv(
    ROOT / "outputs/tables/three_regime_market_summary.csv"
)

three_regime_regressions = pd.read_csv(
    ROOT / "outputs/tables/three_regime_regressions.csv"
)

validation_sample = pd.read_csv(
    ROOT / "outputs/tables/nlp_validation_sample.csv"
)

ai_evaluation_path = (
    ROOT / "outputs/tables/ai_methodology_evaluation.md"
)

print("All final datasets and tables loaded successfully.")
"""
        ),
        markdown(
            r"""
## Data Coverage

The raw article collection begins before the final estimation period so the original corpus is preserved. The daily risk index, thresholds, matching procedure, and final market analysis are restricted to February 28, 2026 onward.

Because February 28, 2026 was a Saturday, the first applicable financial-market date is March 2, 2026.
"""
        ),
        code(
            """
coverage = pd.DataFrame(
    {
        "Dataset": [
            "Scored news articles in raw corpus",
            "Daily risk-index dates in final period",
            "Market dates in final period",
            "Matched high-news observations",
            "Matched low-news observations",
            "Variables in variance diagnostics",
            "Three-regime market dates",
            "NLP validation sample",
        ],
        "Observations": [
            len(news),
            len(daily_risk),
            len(market_changes),
            estimation_sample["regime"].eq("high").sum(),
            estimation_sample["regime"].eq("low").sum(),
            len(variance_diagnostics),
            three_regime_summary[
                three_regime_summary["variable"]
                == "r_vix"
            ]["observations"].sum(),
            len(validation_sample),
        ],
    }
)

coverage
"""
        ),
        markdown(
            r"""
The final daily risk dataset contains **146 dates**. The merged market and news dataset contains **142 dates**. The heteroskedasticity sample contains **15 high-news dates and 15 matched low-news dates**.

Several market-return series have 138 usable observations because of non-trading days and missing price changes, while the VIX has 142 usable observations.
"""
        ),
        markdown(
            r"""
## News Collection

Iran-related news was collected across three broad categories:

- Military conflict
- Nuclear developments
- Diplomatic negotiations

The corpus contains articles from a broad range of major news outlets. Google News RSS may cap the number of results returned for an individual search. Article counts should therefore be interpreted as the volume observed in the collected sample rather than a complete census of all published news.

The primary daily index gives more weight to semantic event intensity than to raw article counts, reducing—but not eliminating—the effect of capped search results.
"""
        ),
        markdown(
            r"""
## NLP Methodology

The sentence-transformer model `all-MiniLM-L6-v2` converts each headline and RSS description into a semantic embedding.

Each article is compared with seed statements representing:

- Iran-war relevance
- Military escalation
- Diplomatic de-escalation

For article \(i\):

\[
EventStrength_i
=
\max(
Escalation_i,
Deescalation_i
)
\]

\[
WarNewsIntensity_i
=
Relevance_i
\times
EventStrength_i
\]

The maximum of escalation and de-escalation similarity is used because both major escalation and major de-escalation news can produce substantial revisions in perceived war risk.
"""
        ),
        code(
            """
score_columns = [
    "relevance_score",
    "escalation_score",
    "deescalation_score",
    "war_news_intensity",
    "war_risk_direction",
]

news[score_columns].describe().round(4)
"""
        ),
        code(
            """
signal_counts = (
    news["dominant_signal"]
    .value_counts()
    .rename_axis("Dominant Signal")
    .reset_index(name="Articles")
)

signal_counts
"""
        ),
        markdown(
            r"""
## Daily Iran War Risk Index

Articles published after 4:00 p.m. Eastern Time are assigned to the following business day. Weekend news is assigned to the next business day.

The daily index is:

\[
RiskIndex_t
=
0.80 Z(Top10Intensity_t)
+
0.20 Z(UncertaintyRate_t)
\]

Using the mean intensity of the ten strongest articles reduces dependence on total article counts. The top 10% of daily index values are classified as high-news dates. Dates below the median are eligible to serve as matched low-news observations.
"""
        ),
        code(
            """
display(
    Image(
        filename=str(
            ROOT
            / "outputs/figures/daily_war_risk_index.png"
        )
    )
)
"""
        ),
        markdown(
            r"""
## Table 1: High- and Low-News Regimes

Table 1 contains the 30 dates used in the heteroskedasticity analysis:

- 15 high-news dates
- 15 uniquely matched low-news dates

The matching criterion minimizes distance in trading days, with lower risk-index values used as a tie-breaker. The median matching distance is one trading day and the maximum is four trading days.
"""
        ),
        code(
            """
table_1[
    [
        "Date",
        "Regime",
        "NLP Risk Score",
        "News Direction",
        "Article Count",
        "Primary Event",
        "Primary Source",
    ]
]
"""
        ),
        code(
            """
appendix_summary = pd.DataFrame(
    {
        "Appendix Measure": [
            "Daily rows",
            "First date",
            "Last date",
            "High-variance dates",
            "Matched low-variance dates",
            "Other dates",
        ],
        "Value": [
            len(appendix_daily),
            appendix_daily["Date"].min(),
            appendix_daily["Date"].max(),
            appendix_daily["Regime"]
            .eq("High variance")
            .sum(),
            appendix_daily["Regime"]
            .eq("Matched low variance")
            .sum(),
            appendix_daily["Regime"]
            .eq("Other")
            .sum(),
        ],
    }
)

appendix_summary
"""
        ),
        markdown(
            r"""
## Financial Variables

The analysis includes:

- Five-year and ten-year U.S. Treasury yields
- S&P 500
- Global equities
- Investment-grade corporate bonds
- High-yield corporate bonds
- Inflation-linked Treasury bonds
- Brent crude oil
- Gold
- U.S. Dollar Index
- VIX
- Investment-grade credit excess-return proxy
- High-yield credit excess-return proxy

Yield changes are measured in percentage points. Prices and indices are measured using percentage log returns.

The credit variables are not actual option-adjusted spreads. They are calculated as corporate-bond ETF returns minus inflation-linked Treasury ETF returns:

\[
IGProxy_t = r_{LQD,t} - r_{TIP,t}
\]

\[
HYProxy_t = r_{HYG,t} - r_{TIP,t}
\]

A negative proxy value indicates corporate credit underperformance relative to Treasury inflation-protected bonds and is directionally consistent with weaker credit conditions or wider spreads.
"""
        ),
        markdown(
            r"""
## Identification Through Heteroskedasticity

Let:

\[
\Delta X_t = D z_t + \mu_t
\]

where \(z_t\) is the unobservable Iran War Risk factor, \(D\) contains financial-market sensitivities, and \(\mu_t\) represents other financial shocks.

Define:

\[
\Omega_H = Var(\Delta X_t \mid H)
\]

\[
\Omega_L = Var(\Delta X_t \mid L)
\]

The covariance difference is:

\[
\Delta\Omega = \Omega_H - \Omega_L
\]

If the main difference between regimes is a change in the variance of Iran War Risk, the change in covariance identifies the relative factor loadings.

Using the five-year Treasury yield as the reference variable:

\[
\widehat d_j =
\frac{
Cov_H(\Delta x_1,\Delta x_j)
-
Cov_L(\Delta x_1,\Delta x_j)
}{
Var_H(\Delta x_1)
-
Var_L(\Delta x_1)
}
\]

Heteroskedasticity identifies the factor only up to scale and sign. The factor is therefore normalized to a **10-basis-point increase in the five-year Treasury yield**, consistent with the Iran-specific inflation, energy-price, debt-risk, and weaker-flight-to-quality channels described in the assignment guidance.

Directional NLP regressions are used as an additional interpretation check rather than treating the sign normalization as independently identified.
"""
        ),
        markdown(
            r"""
## Variance-Regime Validation

The reference-variable variance must be higher in the high-news regime for the identification strategy to be informative.
"""
        ),
        code(
            """
variance_display = variance_diagnostics[
    [
        "variable",
        "high_observations",
        "low_observations",
        "variance_low",
        "variance_high",
        "variance_difference",
        "variance_ratio",
        "levene_pvalue",
    ]
].copy()

variance_display.round(4)
"""
        ),
        markdown(
            r"""
The five-year Treasury yield variance ratio is approximately **2.35**, satisfying the required variance-direction condition. Seven of the thirteen variables have higher variance on high-news dates.

The individual Levene tests are not statistically significant, which is unsurprising with only 15 matched observations per regime. The analysis should therefore be interpreted cautiously because the sample provides limited power for detecting variance differences.
"""
        ),
        code(
            """
display(
    Image(
        filename=str(
            ROOT
            / "outputs/figures/variance_ratios.png"
        )
    )
)
"""
        ),
        markdown(
            r"""
## Table 2: Estimated Impact of an Increase in Iran War Risk

The estimated effects are normalized to a 10-basis-point increase in the five-year Treasury yield. Bootstrap confidence intervals are reported for the combined estimator.
"""
        ),
        code(
            """
table_2
"""
        ),
        markdown(
            r"""
## Interpretation of Table 2

The central heteroskedasticity results are:

- The ten-year Treasury yield increases by approximately **9.05 basis points**, with a 95% confidence interval of approximately 3.63 to 12.08 basis points.
- The S&P 500 response is approximately **-0.47%**, but is not statistically significant.
- Global equities decline by approximately **-0.75%**, but the estimate is imprecise.
- Investment-grade and high-yield bond returns are negative but insignificant.
- Inflation-linked bonds decline by approximately **-0.32%**, with a confidence interval below zero.
- Brent oil increases by approximately **5.32%**, but the confidence interval includes zero.
- VIX increases by approximately **6.37%**, but with a wide confidence interval.
- The investment-grade credit excess-return proxy is negative, which is directionally consistent with weaker credit conditions, but it is not statistically significant.
- The high-yield credit proxy is close to zero and statistically insignificant.

These estimates provide economically meaningful signs for several variables, but the small matched sample creates substantial uncertainty.
"""
        ),
        code(
            """
display(
    Image(
        filename=str(
            ROOT
            / "outputs/figures/estimated_market_effects.png"
        )
    )
)
"""
        ),
        markdown(
            r"""
## Table 3: Variance Explained by Iran War Risk

For financial variable \(j\):

\[
\widehat{\Delta Var_j}
=
\widehat d_j^2
\left[
Var_H(\Delta x_1)
-
Var_L(\Delta x_1)
\right]
\]

The share of high-news-day variance explained is:

\[
Share_{j,H}
=
\frac{
\widehat{\Delta Var_j}
}{
Var_H(\Delta x_j)
}
\]
"""
        ),
        code(
            """
table_3
"""
        ),
        markdown(
            r"""
The estimated variance contributions are largest for the ten-year Treasury yield, inflation-linked bonds, and Brent oil. A large estimated variance contribution does not imply that the directional coefficient is precisely estimated. Brent oil, for example, has a large variance contribution but a wide directional confidence interval.
"""
        ),
        code(
            """
display(
    Image(
        filename=str(
            ROOT
            / "outputs/figures/variance_explained.png"
        )
    )
)
"""
        ),
        markdown(
            r"""
## Alternative Method: Direct NLP Regressions

The direct directional regression is:

\[
\Delta x_{j,t}
=
\alpha_j
+
\beta_j SignedRisk_t
+
\varepsilon_{j,t}
\]

The intensity regression is:

\[
|\Delta x_{j,t}|
=
\alpha_j
+
\gamma_j RiskIntensity_t
+
\varepsilon_{j,t}
\]

The NLP variables are standardized, and HC3 heteroskedasticity-consistent standard errors are used.
"""
        ),
        markdown(
            r"""
### Directional NLP Results
"""
        ),
        code(
            """
direction_columns = [
    "label",
    "direction_coefficient",
    "direction_standard_error",
    "direction_p_value",
    "direction_r_squared",
    "direction_significant",
]

if "observations" in direct_nlp.columns:
    direction_columns.insert(1, "observations")

direction_table = direct_nlp[
    direction_columns
].copy()

direction_table.round(4)
"""
        ),
        markdown(
            r"""
A one-standard-deviation increase in escalation language is associated with:

- A positive but insignificant change in the five-year Treasury yield.
- A statistically significant increase in the ten-year Treasury yield.
- A statistically significant increase in Brent oil.
- Negative S&P 500 and global-equity coefficients, although neither is statistically significant.
- A negative investment-grade credit-proxy coefficient, although it is statistically insignificant.

These results support the assignment's expectation that Iran escalation may increase longer-term yields and oil prices while placing downward pressure on risk assets.
"""
        ),
        markdown(
            r"""
### NLP Intensity and Absolute Market Movement
"""
        ),
        code(
            """
intensity_columns = [
    "label",
    "intensity_coefficient",
    "intensity_standard_error",
    "intensity_p_value",
    "intensity_r_squared",
    "intensity_significant",
]

if "observations" in direct_nlp.columns:
    intensity_columns.insert(1, "observations")

intensity_table = direct_nlp[
    intensity_columns
].copy()

intensity_table.round(4)
"""
        ),
        markdown(
            r"""
Higher news intensity is significantly associated with a larger absolute Brent oil movement. The gold intensity coefficient is positive and marginally significant at approximately the 10% level, but not at the 5% level.

The low \(R^2\) values indicate that Iran-news measures explain only a small share of daily market variation. This is expected because daily prices also respond to monetary policy, macroeconomic releases, earnings, and other geopolitical events.
"""
        ),
        markdown(
            r"""
## Three-Regime Extension

The direction-neutral high/low split remains the primary heteroskedasticity design. A separate extension classifies dates into:

1. Bad war news
2. Good war news
3. No or low war news

A date must be above the 75th percentile of the daily risk index to enter a directional high-news regime. Among those dates, a nonnegative weighted direction is classified as bad war news and a negative direction as good war news.
"""
        ),
        code(
            """
regime_counts = (
    three_regime_summary[
        three_regime_summary["variable"] == "r_vix"
    ][["regime", "observations"]]
    .reset_index(drop=True)
)

regime_counts
"""
        ),
        code(
            """
selected_variables = [
    "d_five_year_yield",
    "d_ten_year_yield",
    "r_sp500",
    "r_global_equity",
    "r_brent_oil",
    "r_gold",
    "r_vix",
    "r_ig_credit_proxy",
    "r_hy_credit_proxy",
]

three_regime_summary[
    three_regime_summary["variable"].isin(
        selected_variables
    )
][
    [
        "label",
        "regime",
        "observations",
        "mean",
        "median",
        "variance",
        "standard_deviation",
    ]
].round(4)
"""
        ),
        code(
            """
three_regime_regressions[
    three_regime_regressions["variable"].isin(
        selected_variables
    )
][
    [
        "label",
        "term",
        "coefficient",
        "standard_error",
        "p_value",
        "confidence_low",
        "confidence_high",
        "r_squared",
        "observations",
    ]
].round(4)
"""
        ),
        markdown(
            r"""
The three-regime sample contains approximately 30 bad-news dates, 6 good-news dates, and 106 no- or low-news dates. The directional groups are therefore highly unbalanced.

Most three-regime coefficients are statistically insignificant. The significant good-news coefficient for gold is based on only five usable return observations and should not be generalized.

The extension is useful for illustrating direction, but it is not stronger than the primary high/low heteroskedasticity design because the good-news group is very small.
"""
        ),
        markdown(
            r"""
## Iran 2026 Versus Iraq 2003

| Variable | Iraq War 2003 benchmark | Iran 2026 expectation | Iran 2026 evidence |
|---|---|---|---|
| Treasury yields | Yields fell as investors sought safety | Yields may rise because of inflation, debt concerns, and weaker flight-to-quality demand | Ten-year yield rises significantly in both the normalized heteroskedasticity model and direct NLP regression |
| Oil | Oil increased with perceived supply risk | Oil should increase, although greater U.S. production may reduce the domestic macroeconomic effect | Brent rises in both main approaches; direct NLP coefficient is significant |
| Credit spreads | Spreads widened | Credit conditions should weaken | IG credit proxy is negative but insignificant; HY proxy is inconclusive |
| Equities | Equities fell | Equities should fall | S&P 500 and global-equity estimates are negative in the primary and direct NLP specifications but insignificant |
| Gold | Limited movement in the benchmark | A safe-haven or inflation response is possible | Directional estimates are unstable; intensity is associated with larger gold movements at approximately the 10% level |
| VIX | Risk and uncertainty increased | VIX should increase | Heteroskedasticity estimate is positive but imprecise |

U.S. crude-oil production is substantially higher in 2026 than it was around the Iraq War period—approximately 14 million barrels per day compared with roughly 6 million barrels per day. Greater domestic production may reduce direct U.S. vulnerability to imported supply disruptions.

However, Iran-related conflict can still affect global energy prices through the Strait of Hormuz, regional production, shipping insurance, and the possibility of broader military escalation.
"""
        ),
        markdown(
            r"""
## Novel Language and NLP Validation

Fixed legacy dictionaries may fail to recognize new conflict terminology, indirect references, newly named operations, or evolving diplomatic language.

Sentence embeddings improve on exact keyword matching because semantically similar phrases can receive similar scores even when they do not contain the same words. Nevertheless, embeddings do not eliminate errors arising from:

- Mixed escalation and de-escalation headlines
- Sarcasm or rhetorical language
- Indirect references
- Headlines requiring broader context
- Newly introduced military or diplomatic terminology
- Descriptions that discuss conflict historically rather than as a new event

A stratified 50-article validation sample was created from:

- 20 highest-intensity articles
- 15 articles near the intensity threshold
- 15 randomly selected articles

The manual-label columns are intentionally left separate from the model output. No manual accuracy statistic is reported until those labels are completed.
"""
        ),
        code(
            """
validation_sample[
    [
        "sample_type",
        "published_utc",
        "title_clean",
        "source",
        "predicted_signal",
        "war_news_intensity",
        "war_risk_direction",
        "manual_label",
        "classification_correct",
        "novel_phrasing",
        "notes",
    ]
].head(15)
"""
        ),
        markdown(
            r"""
## Evaluation of Alternative Identification Methods

Identification through heteroskedasticity is appropriate as the **primary replication method**, but it is not independently conclusive.

| Method | Main advantage | Main weakness | Role in this project |
|---|---|---|---|
| Identification through heteroskedasticity | Addresses simultaneity without requiring perfect observation of the latent shock | Strong regime-stability assumptions, small-sample instability, and sign normalization | Primary replication |
| Direct NLP regression | Uses nearly all dates and provides directional interpretation | Omitted-variable bias and NLP measurement error | Main robustness test |
| Three-regime analysis | Separates bad and good war news | Very small good-news group | Exploratory extension |
| Traditional event study | Clear interpretation around precisely timed events | Contamination and timing requirements | Useful for selected major events |
| Local projections | Estimates dynamic responses over several horizons | Requires more observations | Future extension |
| Structural VAR | Models joint market dynamics and feedback | Strong ordering restrictions and overparameterization | Less suitable for the short sample |
| External-instrument IV | Potentially strong causal interpretation | A credible instrument and exclusion restriction are difficult to establish | Theoretically attractive but impractical here |
| Narrative identification | Uses expert-reviewed surprise events | Subjective and produces few observations | Useful for validation |

The preferred strategy is therefore:

1. Use heteroskedasticity identification as the primary replication.
2. Use direct NLP regressions as the main robustness analysis.
3. Use the three-regime analysis as a directional extension.
4. Use manual headline review to assess NLP classification quality.
5. Consider a narrow event-study appendix for a few precisely timed events.

Conclusions are strongest when the different approaches produce economically consistent patterns.
"""
        ),
        code(
            """
if ai_evaluation_path.exists():
    evaluation_text = ai_evaluation_path.read_text(
        encoding="utf-8"
    )
    display(Markdown(evaluation_text))
else:
    print(
        "Separate methodology evaluation file "
        "was not found."
    )
"""
        ),
        markdown(
            r"""
## Assumptions

The heteroskedasticity interpretation depends on the following assumptions:

1. Iran War Risk is the principal factor whose variance changes between the selected regimes.
2. Financial-market factor loadings remain stable across regimes.
3. Iran War Risk is sufficiently independent of other structural shocks.
4. The variances of unrelated shocks do not change systematically across regimes.
5. High-news dates are not systematically contaminated by monetary-policy, inflation, employment, or unrelated geopolitical announcements.
6. The factor structure is approximately linear.
7. The NLP index correctly separates relatively high- and low-information dates.

The direct NLP regressions additionally require the signed NLP score to contain meaningful directional information and not merely reflect market reactions already incorporated into news coverage.
"""
        ),
        markdown(
            r"""
## Limitations

The main limitations are:

- The final period covers only February 28 through September 18, 2026.
- The heteroskedasticity design uses only 15 matched pairs.
- Some high-news return series contain only 13 usable observations.
- Google News RSS may cap results.
- NLP is applied to headlines and short descriptions rather than full articles.
- Novel or mixed language may be misclassified.
- The good-news regime contains only six dates.
- The five-year Treasury yield replaces the two-year yield used in the original paper.
- The sign of the latent factor is imposed through normalization.
- LQD and HYG returns are not actual credit spreads.
- The credit excess-return measures are only proxies.
- Macroeconomic and monetary-policy events may overlap with Iran news.
- Individual variance-difference tests have limited power.
- Several confidence intervals are wide.
- Direct NLP regressions have low explanatory power.
- Results depend on the NLP seeds, thresholds, and matching procedure.

The estimates should be interpreted as conditional market sensitivities under the model assumptions, not as unconditional proof that Iran War Risk caused every observed market movement.
"""
        ),
        markdown(
            r"""
## Conclusion

The updated analysis identifies 15 high-news dates and 15 matched low-news dates between February 28 and September 18, 2026. The five-year Treasury yield variance is approximately 2.35 times higher in the high-news regime, supporting the variance-direction condition required for heteroskedasticity-based identification.

Under normalization to a 10-basis-point increase in the five-year Treasury yield, the ten-year yield increases significantly by approximately 9.05 basis points. Equity responses are negative, Brent oil and VIX responses are positive, and investment-grade credit conditions weaken, although these estimates are generally imprecise.

The direct NLP regressions provide important complementary evidence. Escalation language is associated with a significant increase in the ten-year Treasury yield and Brent oil. Equity coefficients are negative but insignificant. Higher news intensity is associated with significantly larger absolute Brent movements.

The three-regime analysis provides directional context but is limited by only six good-news dates. Credit excess-return proxies are directionally informative but statistically insignificant.

Overall, the evidence suggests that Iran War Risk in 2026 is most clearly reflected in longer-term Treasury yields and oil prices. The results differ from the traditional Iraq-era flight-to-quality pattern because Iran risk may operate through inflation, energy supply, sovereign-debt concerns, and weaker demand for Treasuries as safe assets.

Identification through heteroskedasticity remains the appropriate primary replication method, while direct NLP regressions and the three-regime analysis provide necessary robustness and interpretation.
"""
        ),
        markdown(
            r"""
## References

Rigobon, R. (2003). Identification through Heteroskedasticity. *The Review of Economics and Statistics*, 85(4), 777–792.

Rigobon, R., and Sack, B. (2003). The Effects of War Risk on U.S. Financial Markets. *NBER Working Paper No. 9609*.

Caldara, D., and Iacoviello, M. (2022). Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194–1225.
"""
        ),
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nbf.write(
        notebook,
        OUTPUT_PATH,
    )

    print(f"Final notebook created: {OUTPUT_PATH}")
    print(
        f"Notebook cells: "
        f"{len(notebook['cells'])}"
    )


if __name__ == "__main__":
    main()