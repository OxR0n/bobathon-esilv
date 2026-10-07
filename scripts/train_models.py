"""Parkinson's true-OFF prediction challenge - model pipeline (Day 1 + Day 2).

Runs, in order:
  1. Dummy mean baseline            -> submissions/01_dummy.csv
  2. Ridge (median-imputed numeric) -> submissions/02_ridge.csv
  3. HistGradientBoosting (NaNs kept as signal) -> submissions/03_hgbr.csv
  4. skrub tabular_pipeline (+ cohort/gene/rater_id categoricals) -> submissions/04_skrub.csv
  5. Best DataOp graph with patient-grouped CV baked in -> submissions/05_dataops.csv

Evaluation uses GroupKFold on ``patient_id`` (Kaggle holdout is by patient).

Usage (from the repo root, with the .venv active):

    python scripts/train_models.py

Pushing reports to Skore Hub requires a valid ``.skore`` file (SETUP.md step 5);
without it the script still trains models and writes submission CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
SKORE_FILE = REPO_ROOT / ".skore"

N_SPLITS = 5

# Numeric columns only (Ridge / hand-built pipelines).
FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train visits with target, test visits), plus engineered features."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train.merge(y_train, on="Index")

    for df in (visits, X_test):
        # Temporal progression: years since diagnosis at the visit.
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
        # Visit ordering within a patient (captures progression direction).
        df["visit_number"] = df.groupby("patient_id").cumcount()

    return visits, X_test


def grouped_cv(visits: pd.DataFrame, feature_frame: pd.DataFrame):
    """Pre-computed GroupKFold index pairs (skore's splitter wants no groups kwarg)."""
    y = visits["target"]
    return list(GroupKFold(n_splits=N_SPLITS).split(feature_frame, y, groups=visits["patient_id"]))


def skore_mean_rmse(report) -> float:
    """Extract the aggregate mean RMSE from a skore report (MultiIndex frame)."""
    frame = report.metrics.rmse(data_source="test")
    for col in frame.columns:
        if "mean" in str(col):
            return float(frame[col].iloc[0])
    raise KeyError("no mean column in skore RMSE frame")


def rmse_report(model, X, y, cv_splits, name: str) -> float:
    """Evaluate with skore if available; fall back to plain grouped CV RMSE.

    Returns the *pooled* RMSE (all out-of-fold predictions concatenated), which
    matches how Kaggle scores the whole test set.
    """
    from sklearn.model_selection import cross_val_predict

    oof = cross_val_predict(clone(model), X, y, cv=cv_splits)
    pooled = float(np.sqrt(np.mean((y.to_numpy() - oof) ** 2)))

    try:
        from skore import evaluate

        report = evaluate(model, X, y, splitter=cv_splits)
        mean_rmse = skore_mean_rmse(report)
        print(f"[{name}] skore CV-RMSE = {mean_rmse:.4f} | pooled OOF RMSE = {pooled:.4f}")
    except Exception as exc:  # skore missing or API mismatch
        print(f"[{name}] fallback pooled grouped-CV RMSE = {pooled:.4f} ({exc})")
    return pooled


def main():
    visits, X_test = load_data()
    y = visits["target"]

    # Feature frames -----------------------------------------------------------------
    num_cols = FEATURE_COLS + ["time_since_diagnosis", "visit_number"]
    X_num_train, X_num_test = visits[num_cols], X_test[num_cols]

    full_drop = ["Index", "patient_id", "target"]
    X_full_train = visits.drop(columns=full_drop)
    X_full_test = X_test.drop(columns=["Index", "patient_id"])

    # Same frames with string columns as pandas "category" dtype, so that
    # HistGradientBoostingRegressor (categorical_features="from_dtype") can use
    # them directly - this is what the DataOp graph in step 5 feeds the model.
    CAT_COLS = ["cohort", "gene", "rater_id"]
    X_cat_train = X_full_train.copy()
    X_cat_test = X_full_test.copy()
    for col in CAT_COLS:
        X_cat_train[col] = X_cat_train[col].astype("category")
        X_cat_test[col] = pd.Categorical(
            X_cat_test[col], categories=X_cat_train[col].cat.categories
        )

    cv_num = grouped_cv(visits, X_num_train)
    cv_full = grouped_cv(visits, X_full_train)

    results: dict[str, float] = {}

    # 1. Dummy mean -------------------------------------------------------------------
    dummy = DummyRegressor(strategy="mean")
    results["01_dummy"] = rmse_report(dummy, X_num_train, y, cv_num, "01_dummy")

    # 2. Ridge --------------------------------------------------------------------------
    ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
    results["02_ridge"] = rmse_report(ridge, X_num_train, y, cv_num, "02_ridge")

    # 3. HistGradientBoosting, NaNs kept ------------------------------------------------
    hgbr = HistGradientBoostingRegressor(random_state=0)
    results["03_hgbr"] = rmse_report(hgbr, X_num_train, y, cv_num, "03_hgbr")

    # 3b. HGBR + categoricals (cohort/gene/rater_id as pandas category dtype) ----------
    try:
        from skore import evaluate as skore_evaluate

        cv_cat = grouped_cv(visits, X_cat_train)
        report = skore_evaluate(hgbr, X_cat_train, y, splitter=cv_cat)
        results["03b_hgbr_cat"] = skore_mean_rmse(report)
        print(f"[03b_hgbr_cat] skore CV-RMSE = {results['03b_hgbr_cat']:.4f}")
    except Exception as exc:
        from sklearn.model_selection import cross_val_predict

        oof = cross_val_predict(clone(hgbr), X_cat_train, y, cv=grouped_cv(visits, X_cat_train))
        results["03b_hgbr_cat"] = float(np.sqrt(np.mean((y.to_numpy() - oof) ** 2)))
        print(f"[03b_hgbr_cat] fallback pooled RMSE = {results['03b_hgbr_cat']:.4f} ({exc})")


    # 4. skrub tabular_pipeline with categoricals ---------------------------------------
    skrub_model = None
    try:
        from skrub import tabular_pipeline

        skrub_model = tabular_pipeline("regressor")
        results["04_skrub"] = rmse_report(skrub_model, X_full_train, y, cv_full, "04_skrub")
    except ImportError:
        print("[04_skrub] skrub not installed, skipping")

    # 5. skrub DataOps: grouped CV baked into the graph ---------------------------------
    learner = None
    if skrub_model is not None:
        try:
            import skrub
            from sklearn.ensemble import HistGradientBoostingRegressor as _HGBR

            data = skrub.var("visits", visits)
            grp = data["patient_id"]
            # NOTE: pass the column list, not a bare string (skrub 0.11 forwards it
            # to DataFrame.drop, which treats a str as labels of the *index*).
            # Drop only "Index"/"patient_id": X_test has no "target", and skrub's
            # lazy graph re-executes this node at predict time, so dropping
            # "target" eagerly would crash on test. Instead we select the feature
            # columns explicitly - they exist in both train and test frames.
            feat_cols = [c for c in visits.columns if c not in ("Index", "patient_id", "target")]
            X_op = data[feat_cols].skb.mark_as_X(
                cv=GroupKFold(n_splits=N_SPLITS),
                split_kwargs={"groups": grp},
            )
            y_op = data["target"].skb.mark_as_y()
            pred_op = (
                X_op.skb.apply(skrub.TableVectorizer())
                .skb.apply(_HGBR(random_state=0), y=y_op)
            )
            from skore import evaluate as skore_evaluate

            # skore 0.26 needs the environment dict explicitly for named vars.
            report = skore_evaluate(pred_op, data={"visits": visits})
            mean_rmse = skore_mean_rmse(report)
            print(f"[05_dataops] skore CV-RMSE (cv/groups from DataOp) = {mean_rmse:.4f}")
            learner = pred_op.skb.make_learner()
            results["05_dataops"] = mean_rmse
        except Exception as exc:
            print(f"[05_dataops] skipped ({exc})")


    # Write submission CSVs -------------------------------------------------------------
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    def submit(key: str, model, X_fit, X_pred):
        final = clone(model).fit(X_fit, y)
        sub = X_test[["Index"]].copy()
        sub["target"] = final.predict(X_pred)
        sub.to_csv(SUBMISSIONS_DIR / f"{key}.csv", index=False)
        print(f"wrote submissions/{key}.csv")

    submit("01_dummy", dummy, X_num_train, X_num_test)
    submit("02_ridge", ridge, X_num_train, X_num_test)
    submit("03_hgbr", hgbr, X_num_train, X_num_test)
    if skrub_model is not None:
        submit("04_skrub", skrub_model, X_full_train, X_full_test)
    if learner is not None:
        learner.fit({"visits": visits})
        sub = X_test[["Index"]].copy()
        sub["target"] = learner.predict({"visits": X_test})
        sub.to_csv(SUBMISSIONS_DIR / "05_dataops.csv", index=False)
        print("wrote submissions/05_dataops.csv")


    best_key = min(results, key=results.get)
    print("\nGrouped-CV RMSE summary:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nBest model: {best_key} (dummy floor: {results['01_dummy']:.4f})")

    # Optional: push reports to Skore Hub ----------------------------------------------
    if SKORE_FILE.is_file():
        cfg = json.loads(SKORE_FILE.read_text())
        try:
            from skore import Project, login

            login(mode="hub")
            project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
            print("Hub configured - re-run each experiment with project.put(key, report) to publish.")
        except Exception as exc:
            print(f"Skore Hub sign-in failed ({exc}); submission CSVs are still ready.")
    else:
        print("No .skore file found: skipping Hub push (see SETUP.md step 5).")


if __name__ == "__main__":
    main()
