"""
Step 2-3: build snapshots, train baselines + LightGBM, calibrate, evaluate on the business metric.

    python src/train.py
Outputs: models/churn_model.joblib, reports/metrics.json, reports/*.png
"""
import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import (CLEAN_FILE, MODEL_FILE, MODELS_DIR, OFFER_COST, REPORTS_DIR, SAVE_RATE, SEED,
                    TEST_CUTOFF, TRAIN_CUTOFFS, VAL_CUTOFF, VALUE_IF_SAVED, break_even_probability)
from features import FEATURES, build_dataset, build_snapshot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")


# ---------------------------------------------------------------- metrics ---
def profit_at_mask(y: np.ndarray, mask: np.ndarray) -> float:
    """Realised expected profit: saved churners among contacted minus cost of all contacts."""
    return float(y[mask].sum() * SAVE_RATE * VALUE_IF_SAVED - mask.sum() * OFFER_COST)


def profit_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y_sorted = y[order]
    k = np.arange(1, len(y) + 1)
    profit = np.cumsum(y_sorted) * SAVE_RATE * VALUE_IF_SAVED - k * OFFER_COST
    return k / len(y), profit


def precision_lift_at(y: np.ndarray, score: np.ndarray, frac: float) -> tuple[float, float]:
    k = max(1, int(len(y) * frac))
    top = np.argsort(-score)[:k]
    precision = y[top].mean()
    return float(precision), float(precision / y.mean())


def evaluate(name: str, y: np.ndarray, p: np.ndarray, rank_score: np.ndarray | None = None) -> dict:
    rank_score = p if rank_score is None else rank_score
    prec10, lift10 = precision_lift_at(y, rank_score, 0.10)
    prec20, lift20 = precision_lift_at(y, rank_score, 0.20)
    frac, profit = profit_curve(y, rank_score)
    be = break_even_probability()
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y, rank_score)),
        "pr_auc": float(average_precision_score(y, rank_score)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "precision@10%": prec10, "lift@10%": lift10,
        "precision@20%": prec20, "lift@20%": lift20,
        "profit_best_k": float(profit.max()),
        "best_k_fraction": float(frac[profit.argmax()]),
        "profit_at_break_even_threshold": profit_at_mask(y, p >= be),
    }


# ------------------------------------------------------------------- plots ---
def plot_profit_and_gain(y, scores: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, s in scores.items():
        frac, profit = profit_curve(y, s)
        ax[0].plot(frac, profit, label=name)
        order = np.argsort(-s)
        ax[1].plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[order]) / y.sum(), label=name)
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set(title="Expected profit vs. share of customers contacted", xlabel="Share contacted", ylabel="Profit (GBP)")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax[1].set(title="Cumulative gains (share of churners captured)", xlabel="Share contacted", ylabel="Share of churners")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "profit_and_gain.png", dpi=150); plt.close(fig)


def plot_calibration(y, p_raw, p_cal) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, p in (("raw LightGBM", p_raw), ("calibrated", p_cal)):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="Calibration on test set", xlabel="Predicted probability", ylabel="Observed churn rate")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS_DIR / "calibration.png", dpi=150); plt.close(fig)


def plot_shap(model, X: pd.DataFrame) -> None:
    try:
        import shap
        sample = X.sample(min(2000, len(X)), random_state=SEED)
        sv = shap.TreeExplainer(model).shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        plt.figure()
        shap.summary_plot(sv, sample, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150); plt.close()
    except Exception as exc:  # SHAP is a nice-to-have; never break training for it
        print(f"SHAP plot skipped: {exc}")


def segment_report(test: pd.DataFrame, p: np.ndarray) -> list[dict]:
    """Error analysis: does the model work equally well across customer segments?"""
    seg = pd.Series(np.where(test["frequency"] == 1, "1 order",
                    np.where(test["frequency"] <= 4, "2-4 orders", "5+ orders")), index=test.index)
    rows = []
    groups = {"UK": test["is_uk"] == 1, "non-UK": test["is_uk"] == 0,
              **{f"{s}": seg == s for s in ["1 order", "2-4 orders", "5+ orders"]}}
    for name, mask in groups.items():
        y = test.loc[mask, "churned"].to_numpy()
        if mask.sum() >= 30 and 0 < y.mean() < 1:
            rows.append({"segment": name, "n": int(mask.sum()), "churn_rate": float(y.mean()),
                         "roc_auc": float(roc_auc_score(y, p[mask.to_numpy()]))})
    return rows


# -------------------------------------------------------------------- main ---
def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(CLEAN_FILE)

    train = build_dataset(df, TRAIN_CUTOFFS)
    val = build_snapshot(df, VAL_CUTOFF)
    test = build_snapshot(df, TEST_CUTOFF)
    for name, d in (("train", train), ("val", val), ("test", test)):
        print(f"{name:5s}: {len(d):6,} rows | churn rate {d['churned'].mean():.1%}")

    Xtr, ytr = train[FEATURES], train["churned"].to_numpy()
    Xva, yva = val[FEATURES], val["churned"].to_numpy()
    Xte, yte = test[FEATURES], test["churned"].to_numpy()

    # 1) Benchmark: recency only
    rec = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    rec.fit(Xtr[["recency_days"]], ytr)
    p_rec = rec.predict_proba(Xte[["recency_days"]])[:, 1]

    # 2) Baseline: logistic regression on all features
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight=None))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    # 3) LightGBM with early stopping on the validation snapshot
    gbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )
    gbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"LightGBM best iteration: {gbm.best_iteration_}")

    # 4) Probability calibration (isotonic) fitted on the validation snapshot
    raw_va = gbm.predict_proba(Xva)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_va, yva)
    p_gbm_raw = gbm.predict_proba(Xte)[:, 1]
    p_gbm = calibrator.predict(p_gbm_raw)

    results = [
        evaluate("recency_only_logreg", yte, p_rec),
        evaluate("logistic_regression", yte, p_lr),
        evaluate("lightgbm_calibrated", yte, p_gbm, rank_score=p_gbm_raw),
    ]
    print("\n" + pd.DataFrame(results).set_index("model").round(3).T.to_string())

    # Business baselines
    all_contact = profit_at_mask(yte, np.ones(len(yte), dtype=bool))
    print(f"\nProfit if we contact EVERYONE: GBP {all_contact:,.0f} | contact nobody: GBP 0")
    print(f"Break-even probability: {break_even_probability():.2f}")

    plot_profit_and_gain(yte, {"recency only": p_rec, "logistic": p_lr, "lightgbm": p_gbm_raw})
    plot_calibration(yte, p_gbm_raw, p_gbm)
    plot_shap(gbm, Xte)
    segments = segment_report(test, p_gbm_raw)
    print("\nSegment AUCs:\n" + pd.DataFrame(segments).round(3).to_string(index=False))

    importance = dict(sorted(zip(FEATURES, gbm.booster_.feature_importance("gain")), key=lambda t: -t[1]))
    metrics = {
        "split": {"train_cutoffs": TRAIN_CUTOFFS, "val_cutoff": VAL_CUTOFF, "test_cutoff": TEST_CUTOFF,
                  "n_train": len(train), "n_val": len(val), "n_test": len(test),
                  "test_churn_rate": float(yte.mean())},
        "economics": {"offer_cost": OFFER_COST, "value_if_saved": VALUE_IF_SAVED, "save_rate": SAVE_RATE,
                      "break_even_probability": break_even_probability(), "profit_contact_everyone": all_contact},
        "results": results, "segments": segments,
        "feature_importance_gain": {k: float(v) for k, v in importance.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({"model": gbm, "calibrator": calibrator, "features": FEATURES,
                 "break_even": break_even_probability(), "test_metrics": results[-1]}, MODEL_FILE)
    print(f"\nSaved model -> {MODEL_FILE}")

    # Optional experiment tracking
    try:
        import mlflow
        from config import ROOT
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("churn")
        with mlflow.start_run():
            mlflow.log_params({"cutoff_test": TEST_CUTOFF, "n_estimators": gbm.best_iteration_,
                               "offer_cost": OFFER_COST, "value_if_saved": VALUE_IF_SAVED, "save_rate": SAVE_RATE})
            for k, v in results[-1].items():
                if isinstance(v, float):
                    mlflow.log_metric(k.replace("@", "_at_").replace("%", "pct"), v)
            mlflow.log_artifacts(str(REPORTS_DIR))
        print("Logged run to MLflow (run `mlflow ui` to browse).")
    except ImportError:
        print("MLflow not installed; skipping experiment tracking.")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


if __name__ == "__main__":
    main()
