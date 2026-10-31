"""Central configuration. Edit values here, not inside other scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_FILE = RAW_DIR / "online_retail_II.xlsx"
CLEAN_FILE = PROCESSED_DIR / "transactions_clean.parquet"
SCORES_FILE = PROCESSED_DIR / "scores.parquet"
MODEL_FILE = MODELS_DIR / "churn_model.joblib"

SEED = 42

# ---- Churn definition -------------------------------------------------------
CHURN_WINDOW_DAYS = 90        # churn = no purchase in the 90 days after the cutoff
ACTIVE_LOOKBACK_DAYS = 180    # population = customers who bought in the 180 days BEFORE the cutoff

# ---- Time-based split (each cutoff = one snapshot of customers) ------------
# Label windows never overlap the next split's cutoff, so there is no label leakage.
TRAIN_CUTOFFS = ["2010-06-09", "2010-09-09", "2010-12-09", "2011-03-09"]
VAL_CUTOFF = "2011-06-09"     # label window ends 2011-09-07
TEST_CUTOFF = "2011-09-09"    # label window ends 2011-12-08 (data ends 2011-12-09)

# ---- Business economics (replace with your own numbers from docs/business_brief.md)
OFFER_COST = 10.0             # GBP cost of contacting one customer with a retention offer
VALUE_IF_SAVED = 80.0        # GBP margin-adjusted value of a retained customer
SAVE_RATE = 0.20              # share of would-be churners the offer actually retains

# ---- Risk tiers used by dashboard / scoring ---------------------------------
RISK_HIGH = 0.70
RISK_MEDIUM = 0.40


def break_even_probability() -> float:
    """Contact a customer only if p * SAVE_RATE * VALUE_IF_SAVED > OFFER_COST."""
    return OFFER_COST / (SAVE_RATE * VALUE_IF_SAVED)
