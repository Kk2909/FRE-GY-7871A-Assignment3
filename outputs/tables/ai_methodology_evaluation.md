## Evaluation of Identification Through Heteroskedasticity

Identification through heteroskedasticity is an appropriate **primary methodology for replicating the war-risk paper**, but it is not necessarily the best standalone method for establishing causality in this application. Given the short 2026 sample, it should be combined with direct NLP regressions and a three-regime analysis.

### Identification through heteroskedasticity

The method compares changes in market variances and covariances between high-war-news and low-war-news regimes. If war-risk shocks become more volatile on high-news days while the relationships among other market shocks remain stable, the change in covariance can identify financial-market sensitivity to the latent war-risk factor.

Its principal advantage is that the exact magnitude and direction of every news story do not need to be measured. NLP only needs to identify days on which war-related information is unusually important. This is valuable because individual stories may contain mixed escalation and de-escalation signals.

The method also helps address simultaneity. Treasury yields, oil prices, equities, credit markets, and volatility can respond to war risk at the same time, making an ordinary regression difficult to interpret. Heteroskedasticity-based identification uses the change in the covariance structure rather than assuming that one observed market variable is completely exogenous.

However, the method relies on strong assumptions:

1. War-risk shocks must be more volatile on high-news days than on low-news days.
2. The structural relationships between war risk and financial variables must remain stable across regimes.
3. The variances and covariance structures of other shocks should not change systematically between the two regimes.
4. High-news days should not consistently coincide with monetary-policy announcements, inflation releases, employment reports, or unrelated geopolitical events.
5. The NLP classification must separate high- and low-information days sufficiently well.
6. A one-factor representation must be adequate even though war news can affect markets through several channels, including inflation, risk aversion, energy supply, and flight-to-quality demand.

The approach identifies relative factor loadings only up to scale and sign. Therefore, normalizing the factor to a 10-basis-point increase in the five-year Treasury yield is an interpretation choice, not a direction learned purely from the variance shift. Directional NLP evidence should be used to assess whether that normalization is economically reasonable.

The small number of high-news and matched low-news observations also produces wide confidence intervals and potentially unstable estimates. For this reason, the results should be described as conditional evidence of market sensitivity rather than definitive causal estimates.

## Comparison with Alternative Methods

| Method                                    | Main advantage                                                               | Main limitation                                                                            | Suitability for this project                           |
| ----------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| Identification through heteroskedasticity | Addresses simultaneity without requiring a perfectly observed war-risk shock | Strong regime-stability assumptions; small-sample instability; sign normalization required | Best primary method for replicating the assigned paper |
| Traditional event study                   | Provides intuitive market responses around clearly defined events            | Requires precise event timing and uncontaminated windows                                   | Useful only for a selected set of major announcements  |
| Direct NLP regression                     | Uses the full daily sample and incorporates escalation direction             | Vulnerable to omitted variables, reverse causality, and measurement error                  | Strongest practical robustness test                    |
| Local projections                         | Estimates responses over several future horizons                             | Requires substantially more observations and careful controls                              | Better as a future extension than a principal method   |
| Structural VAR                            | Models interactions and dynamic feedback among markets                       | Sensitive to variable ordering and other identification restrictions                       | Not ideal for the short 2026 sample                    |
| External-instrument IV                    | Can provide stronger causal identification with a valid instrument           | Finding a relevant instrument that affects markets only through Iran War Risk is difficult | Attractive in theory but impractical here              |
| Narrative identification                  | Uses expert-reviewed events to isolate plausibly exogenous shocks            | Subjective and time-intensive; may produce very few events                                 | Useful as a validation exercise                        |
| Three-regime analysis                     | Separates bad, good, and low war news and makes directional effects visible  | Directional groups are small and NLP classification errors matter                          | Valuable exploratory extension                         |

### 1. Traditional event studies

An event study would estimate market movements within narrow windows around major Iran-related announcements or military events. This provides a clear and easily interpreted comparison of pre-event and post-event prices.

Its weakness is event contamination. Important Iran news may occur on the same day as Federal Reserve announcements, inflation data, earnings releases, or other geopolitical developments. Daily data also provide weak timing precision because both the news and the market response may occur within the same day.

An event study is therefore useful for a small set of major, precisely timed events but should not replace the primary heteroskedasticity analysis.

### 2. Direct NLP regressions

Direct regressions relate daily financial-market changes to observed NLP measures such as escalation direction and news intensity. They use substantially more observations than the matched high/low design and provide directional interpretation.

For example, the regression can directly test whether escalation language is associated with higher Treasury yields, higher oil prices, lower equity returns, and higher volatility.

However, these regressions may suffer from omitted-variable bias. Market conditions may affect the volume and tone of news coverage, and both news language and financial prices may respond to the same underlying event. Consequently, the regressions measure conditional same-day associations unless stronger identification is introduced.

Despite this limitation, direct NLP regressions are the most useful robustness method for the current project because they use the full sample and provide the directional information that heteroskedasticity alone cannot identify.

### 3. Local projections

Local projections could estimate the response of each market variable over several days following a war-risk shock. This would help determine whether the effect is temporary, delayed, or persistent.

The principal problem is sample size. A February-to-September daily sample is short, and estimating multiple horizons would quickly reduce statistical power. Overlapping observations would also require appropriate standard errors.

Local projections would become more useful with a longer sample or higher-frequency data.

### 4. Structural VAR models

A structural vector autoregression could jointly model war risk, Treasury yields, oil, equities, and other variables. It would capture feedback and dynamic interactions.

Its conclusions would depend heavily on the chosen ordering, sign restrictions, or contemporaneous restrictions. With many financial variables and a short sample, the system would also be overparameterized and unstable.

A structural VAR is therefore less suitable than a direct regression or local projection for the current dataset.

### 5. External-instrument IV

An external-instrument approach could identify war-risk shocks using an instrument correlated with Iran-related escalation but unrelated to other determinants of global financial markets.

If a credible instrument existed, this could provide stronger causal identification. In practice, military announcements, sanctions, or geopolitical surprises may directly affect energy markets, inflation expectations, or general risk sentiment. This makes the exclusion restriction difficult to defend.

External-instrument IV is theoretically attractive but not currently practical without a carefully justified instrument.

### 6. Narrative identification

Narrative identification would manually classify major events according to their timing, surprise content, and likely direction. This could distinguish unexpected escalation from events that markets had already anticipated.

Manual review can improve the interpretation of novel conflict language and identify false NLP classifications. However, the classification involves researcher judgment and may leave only a very small number of usable events.

It is most useful as validation for the NLP classifications and for constructing a limited event-study appendix.

### 7. Three-regime analysis

A three-regime design separates:

* Bad war news
* Good war news
* No or low war news

This is more informative than a direction-neutral high/low split when the purpose is to examine whether bad news raises yields and oil prices while lowering equities.

Its main weakness is the small and unbalanced directional sample. In the present analysis, there are considerably fewer good-news dates than bad-news dates. Estimates for the good-news regime are therefore especially imprecise. The results should be interpreted as exploratory rather than causal.

## Recommended Methodological Strategy

Identification through heteroskedasticity should remain the **primary method** because the assignment requests a replication of the original war-risk paper and because the technique does not require the latent war-risk shock to be measured perfectly.

It should not be used alone. The preferred empirical strategy is:

1. **Primary specification:** Direction-neutral high-news versus matched low-news identification through heteroskedasticity.
2. **Primary robustness test:** Direct daily NLP regressions using escalation direction and news intensity.
3. **Directional extension:** Three-regime bad-news, good-news, and no-news analysis.
4. **Validation:** Manual review of a stratified sample of headlines to assess novel language and classification errors.
5. **Optional extension:** A narrow event study of a few major, precisely timed Iran-related events.

Local projections, structural VARs, and external-instrument IV are less suitable for the present short sample. They would require more observations, stronger identifying assumptions, or a credible external instrument.

## Conclusion

Identification through heteroskedasticity is the best **primary replication method**, but not an independently conclusive causal design. Its usefulness comes from exploiting variance changes without requiring perfect measurement of each war-risk shock. Its limitations arise from the small matched sample, possible contamination by other market events, structural-stability assumptions, and sign-normalization ambiguity.

The most defensible conclusion should therefore be based on convergence across methods. Evidence is stronger when heteroskedasticity estimates, direct NLP regressions, and three-regime comparisons produce economically consistent patterns. When the methods disagree, the differences should be reported and interpreted rather than forcing a single conclusion.
