# Memo: Churn model results  (TEMPLATE: fill in after running on the real data)

**To:** Head of Retention  **From:** <you>  **Date:** <date>

## Recommendation
Contact the top <X>% of active customers ranked by churn risk (about <N> customers).
Expected profit: £<A> vs £<B> for contacting everyone and £0 for doing nothing.

## What we did
Predicted which active customers will not purchase in the next 90 days, using purchase history only
(recency, frequency, spend, order-gap patterns, cancellations). Validated on a later period the model
never saw (cutoff 2011-09-09).

## Key results (from reports/metrics.json)
- ROC AUC <>, lift in the top 10% <>x versus random
- Beats the "longest silent first" rule by £<>   <- be honest if it does not
- Main drivers (reports/shap_summary.png): <recency, order-gap ratio, ...>

## Where it works less well
<segments from metrics.json "segments", e.g. one-time buyers>

## Risks and next steps
1. Run an A/B test with a holdout group to measure the real offer success rate.
2. Re-train quarterly; monitor feature drift.
3. Add seasonality and product-category features once more history is available.
