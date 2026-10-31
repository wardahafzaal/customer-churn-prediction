# Business Brief: Customer Churn Prevention

## 1. Problem
An online gift-ware retailer (many customers are wholesalers) loses repeat buyers silently: there is no
"cancel" event, customers just stop ordering. The retention team can run one outreach campaign per quarter
and wants a ranked list of customers to contact.

**Decision supported:** which active customers receive a retention offer next quarter.

## 2. Definitions
- **Active customer:** at least one purchase in the 180 days before the cutoff date.
- **Churn:** no purchase in the 90 days after the cutoff date.
- **Why 90 days:** (fill in after checking the distribution of gaps between orders; the 90th percentile of
  inter-order gaps is a good starting point)

## 3. Economics (edit `src/config.py` to match)
| Item | Default | Meaning |
|---|---|---|
| OFFER_COST | £10 | cost of contacting one customer with an offer |
| VALUE_IF_SAVED | £80 | margin-adjusted value of a customer who stays |
| SAVE_RATE | 20% | share of would-be churners the offer retains |

Break-even probability = cost / (save rate x value) = 62.5%. Contact a customer only above this.
These are placeholders: replace them with real numbers and state your source.

## 4. Success metric
Expected campaign profit on a held-out future period, compared with (a) contacting nobody,
(b) contacting everybody, (c) a simple "longest-silent customers first" rule. AUC is reported but is not the goal.

## 5. Data and exclusions
UCI Online Retail II, Dec 2009 to Dec 2011. Excluded: guest checkouts (no customer ID), non-product codes
(postage, bank charges, manual adjustments), invalid prices, duplicate rows.

## 6. Risks and assumptions
- Only two years of history; seasonality (Q4 peak) is barely represented.
- Wholesalers behave differently from consumers; check segment performance.
- Offer success rate is assumed, not measured. Recommend an A/B test (holdout group) before scaling.
- Customer-level features can drift; monitor recency and frequency distributions monthly.
