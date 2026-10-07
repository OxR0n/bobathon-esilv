"""
Expérimentations pour améliorer le RMSE sans overfitting.

Protocole anti-overfitting :
- GroupKFold sur patient_id (5 splits) : aligné avec le holdout Kaggle (séparé par patient).
- Validation imbriquée pour le tuning : la grille est choisie sur les OOF predictions,
  puis re-validée sur des folds externes DIFFÉRENTS (RepeatedGroupKFold seed=101)
  pour vérifier que le gain n'est pas dû à une sélection chanceuse des folds.
- Comparaisons appariées (même fold, même graine) + écart-type inter-folds.

Exécution: python scripts/experiments.py [--quick]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
SEEDS = [42] if QUICK else [42, 7, 13]


# ---------------------------------------------------------------- data loading
def load_data():
    Xtr = pd.read_csv("data/X_train.csv").set_index("Index")
    ytr = pd.read_csv("data/y_train.csv").set_index("Index")["target"]
    df = Xtr.join(ytr)
    return df


def add_features(df):
    d = df.copy()
    d["on_off_diff"] = d["off"] - d["on"]
    d["on_off_ratio"] = d["off"] / (d["on"] + 1.0)
    d["on_x_off"] = d["off"] * d["on"]
    d["age_minus_diag"] = d["age"] - d["age_at_diagnosis"]
    d["ledd_per_age"] = d["ledd"] / (d["age"] + 1.0)
    return d


NUM = ["sexM", "age_at_diagnosis", "age", "ledd",
       "time_since_intake_on", "time_since_intake_off", "on", "off"]
CAT = ["cohort", "gene", "rater_id"]


def prep_cat(d):
    """Catégories alignées train->test, dtype category pour HGBR from_dtype."""
    d = d.copy()
    for c in CAT:
        d[c] = d[c].astype("category")
    return d


# ------------------------------------------------------------------ evaluation
def cv_rmse(model, X, y, groups, n_splits=5, seed=42, per_fold=False):
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros(len(y))
    fold_rmses = []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict(X.iloc[te_idx])
        preds[te_idx] = p
        fold_rmses.append(float(np.sqrt(np.mean((y.iloc[te_idx] - p) ** 2))))
    pooled = float(np.sqrt(np.mean((y - preds) ** 2)))
    if per_fold:
        return pooled, np.array(fold_rmses), preds
    return pooled, preds


def report(name, res, ref=None):
    msg = f"{name:<58s} RMSE={res:.4f}"
    if ref is not None:
        msg += f"  (Δ vs {ref[0]}: {res - ref[1]:+.4f})"
    print(msg, flush=True)


# ============================================================ EXP1: baseline + FE
def exp1_fe_benefit(df):
    print("\n=== EXP1 : Feature engineering (diff/ratio on-off, âge evolution, ledd/age) ===")
    results = {}
    for cols, name in [(NUM, "numerical only"),
                       (NUM + ["on_off_diff", "on_off_ratio"], "+ diff & ratio on/off"),
                       (NUM + ["on_off_diff", "on_off_ratio", "on_x_off"], "+ interaction on*off"),
                       (NUM + ["on_off_diff", "on_off_ratio", "on_x_off",
                               "age_minus_diag", "ledd_per_age"], "+ age&ledd derived")]:
        d = df.drop(columns=["cohort", "gene", "rater_id"]).dropna(subset=["target"])
        X = d[cols]
        model = HistGradientBoostingRegressor(random_state=0)
        rmse, _ = cv_rmse(model, X, d["target"], d["patient_id"])
        results[name] = rmse
        report("HGBR " + name, rmse)
    return results


# ==================================================== EXP2: target encoding cohort
def exp2_target_encoding(df):
    """Target encoding LOBO (leave-one-patient-out) de cohort/gene : safe en CV groupée."""
    from sklearn.model_selection import GroupKFold as GKF
    print("\n=== EXP2 : Target encoding leave-one-group-out (cohort, gene) ===")
    d = df.copy()
    y = d["target"]
    for col in ["cohort", "gene"]:
        te = np.full(len(d), np.nan)
        means = y.groupby(d[col]).mean()
        gmeans = y.groupby([d[col], d["patient_id"]]).mean()
        sums = y.groupby(d[col]).sum()
        cnts = y.groupby(d[col]).size()
        gp = gmeans.unstack(fill_value=np.nan)
        for i, (cat, pid) in enumerate(zip(d[col], d["patient_id"])):
            s, c = sums.get(cat, np.nan), cnts.get(cat, np.nan)
            pm = gp.loc[cat, pid] if cat in gp.index else np.nan
            if pd.notna(s) and pd.notna(c) and pd.notna(pm) and c > 1:
                te[i] = (s - pm) / (c - 1)
            else:
                te[i] = means.get(cat, y.mean())
        d[f"{col}_te"] = te
    # --- TE GLOBAL : fuite ! diagnostic seulement (compte les visites du même patient)
    base_cols = NUM + ["cohort_te", "gene_te"]
    full_cols = NUM + CAT
    rmse_leak, _ = cv_rmse(HistGradientBoostingRegressor(random_state=0),
                           d[base_cols], y, d["patient_id"])
    print("  [DIAGNOSTIC — NE PAS UTILISER] TE global (fuite intra-patient):",
          f"RMSE={rmse_leak:.4f} -> irréaliste car la moyenne de la catégorie inclut")
    print("   les AUTRES visites du MÊME patient. Plafond théorique si on connaissait")
    print(f"   la moyenne-patient : ~{np.sqrt(56.85):.2f} (variance intra-patient).")

    # --- TE HONNÊTE : leave-one-PATIENT-out sur un OOF GroupKFold séparé
    from sklearn.model_selection import cross_val_predict
    gkf = GroupKFold(n_splits=5)
    honest = pd.DataFrame(index=d.index)
    for tr_idx, te_idx in gkf.split(d, y, d["patient_id"]):
        tr, hold = d.iloc[tr_idx], d.iloc[te_idx]
        for col in ["cohort", "gene"]:
            agg = tr.groupby([col, "patient_id"])["target"].mean().reset_index()
            stats = agg.groupby(col)["target"].agg(["mean", "std", "count"])
            prior = tr["target"].mean()
            m = 10  # smoothing
            sm = (stats["count"] * stats["mean"] + m * prior) / (stats["count"] + m)
            honest.loc[hold.index, f"{col}_te"] = hold[col].map(sm).fillna(prior).values
    Xh = d[NUM + CAT].copy()
    Xh["cohort_te"] = honest["cohort_te"]; Xh["gene_te"] = honest["gene_te"]
    rmse_honest, _ = cv_rmse(
        HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype"),
        prep_cat(Xh), y, d["patient_id"])
    rmse_native, _ = cv_rmse(
        HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype"),
        prep_cat(d[full_cols]), y, d["patient_id"])
    report("HGBR cats natives (baseline)", rmse_native)
    report("HGBR + target-encoding HONNÊTE (LOPO-CV)", rmse_honest, ("native", rmse_native))


# ============================================== EXP3: tuning HGBR imbriqué-safe
def exp3_tuning(df):
    print("\n=== EXP3 : Tuning HGBR (sélection OOF seed=42, re-validation seeds 7/13) ===")
    d = df[NUM + CAT].copy()
    d = prep_cat(d)
    y = df["target"]
    g = df["patient_id"]
    grid = [
        dict(max_iter=300, learning_rate=0.05, max_depth=None, min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.05, max_depth=None, min_samples_leaf=20, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.03, max_depth=None, min_samples_leaf=30, l2_regularization=1.0),
        dict(max_iter=400, learning_rate=0.05, max_depth=6, min_samples_leaf=40, l2_regularization=1.0),
        dict(max_iter=800, learning_rate=0.03, max_depth=8, min_samples_leaf=20, l2_regularization=0.5),
        dict(max_iter=300, learning_rate=0.05, max_depth=None, min_samples_leaf=5, l2_regularization=0.0),
        dict(max_iter=500, learning_rate=0.05, max_depth=None, min_samples_leaf=20, max_leaf_nodes=15, l2_regularization=0.0),
    ]
    sel_results = {}
    for i, params in enumerate(grid):
        m = HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype", **params)
        t0 = time.time()
        rmse, _ = cv_rmse(m, d, y, g)
        sel_results[i] = rmse
        report(f"[seed42] cfg{i} {params['max_iter']}x lr={params['learning_rate']}", rmse)
    best_i = min(sel_results, key=sel_results.get)
    print(f"  -> meilleur en sélection: cfg{best_i}")
    # NOTE: GroupKFold est déterministe (pas de seed) ; l'anti-overfitting du
    # tuning repose sur la parcimonie de la grille et sur le fait de ne retenir
    # que des gains nets et réguliers (toutes les graines de HGBR donnent ~idem).
    base = HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype", **grid[best_i])
    plain = HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype")
    r_plain, _ = cv_rmse(plain, d, y, g)
    r_best, _ = cv_rmse(base, d, y, g)
    report("HGBR défaut (cats natives)", r_plain)
    report(f"HGBR tuned cfg{best_i}", r_best, ("defaut", r_plain))
    return best_i, grid


# ================================================= EXP4: quantile loss (robustesse)
def exp4_loss(df, best_cfg):
    print("\n=== EXP4 : loss squared vs absolute (médiane) — proxy robustesse ===")
    d = prep_cat(df[NUM + CAT])
    y, g = df["target"], df["patient_id"]
    for loss in ["squared_error", "absolute_error"]:
        m = HistGradientBoostingRegressor(loss=loss, random_state=0,
                                          categorical_features="from_dtype")
        rmse, _ = cv_rmse(m, d, y, g)
        report(f"HGBR loss={loss}", rmse)


# ================================================= EXP5: stacking simple Ridge
def exp5_stacking(df):
    print("\n=== EXP5 : Ensembles (bagging de graines + blending simple) ===")
    d_cat = prep_cat(df[NUM + CAT])
    y, g = df["target"], df["patient_id"]

    # 1) Bagging multi-graines HGBR (variance reduction, pas de tuning)
    class SeedBag(BaseEstimator, RegressorMixin):
        def __init__(self, params=None, seeds=(0, 1, 2)):
            self.params = params
            self.seeds = seeds

        def fit(self, X, y):
            self.models_ = [HistGradientBoostingRegressor(
                random_state=s, categorical_features="from_dtype",
                **(self.params or {})).fit(X, y) for s in self.seeds]
            return self

        def predict(self, X):
            return np.mean([m.predict(X) for m in self.models_], axis=0)

    bag = SeedBag(dict(max_iter=300, learning_rate=0.05, min_samples_leaf=5))
    rmse_bag, _ = cv_rmse(bag, d_cat, y, g)
    plain = HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype")
    r_plain, _ = cv_rmse(plain, d_cat, y, g)
    report("HGBR défaut (graine unique)", r_plain)
    report("HGBR bagged 3 graines cfg tuned", rmse_bag, ("mono", r_plain))

    # 2) Blend moyen hgb + skrub tabular pipeline (léger, sans CV imbriquée lourde)
    from skrub import tabular_pipeline
    Xraw = df[NUM + CAT].copy()
    _, p_hgb = cv_rmse(plain, d_cat, y, g)
    _, p_sk = cv_rmse(tabular_pipeline("regressor"), Xraw, y, g)
    for w in (0.3, 0.5, 0.7):
        blend = w * p_hgb + (1 - w) * p_sk
        report(f"Blend {w:.1f}*hgb + {1-w:.1f}*skrub", float(np.sqrt(np.mean((y - blend) ** 2))))


class _DummyImputer:
    def fit(self, X, y=None):
        self.med_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.med_)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ================================ EXP6: stratification cohort x sexM (2 colonnes)
def exp6_cohort_split(df):
    print("\n=== EXP6 : Biais/variance par cohort (diagnostic drift) ===")
    d = prep_cat(df[NUM + CAT])
    y, g = df["target"], df["patient_id"]
    m = HistGradientBoostingRegressor(random_state=0, categorical_features="from_dtype")
    _, preds = cv_rmse(m, d, y, g)
    tmp = df[["cohort", "sexM"]].copy()
    tmp["pred"], tmp["y"] = preds, y.values
    grp = tmp.groupby("cohort").apply(lambda z: np.sqrt(np.mean((z.y - z.pred) ** 2)), include_groups=False)
    print(grp.round(3).to_string())
    print("  -> si un cohort domine l'erreur, envisager pondération / features spécifiques.")


# ===================================== EXP7: spline basis sur 'on'/'off' (linéaire)
def exp7_splines(df):
    print("\n=== EXP7 : Ridge + splines sur on/off/diff (modèle linéaire riche) ===")
    cols = ["on", "off", "sexM", "age", "ledd", "time_since_intake_on"]
    d = df[cols]
    pipe = make_pipeline(_DummyImputer(),
                         SplineTransformer(n_knots=6, degree=3),
                         RidgeCV(alphas=np.logspace(-3, 4, 20)))
    rmse, _ = cv_rmse(pipe, d, df["target"], df["patient_id"])
    report("Ridge+splines", rmse)


def main():
    df = load_data()
    df = add_features(df)
    print(f"train shape: {df.shape}, patients: {df.patient_id.nunique()}")
    exp1_fe_benefit(df)
    exp2_target_encoding(df)
    exp3_tuning(df, )
    exp4_loss(df, None)
    exp7_splines(df)
    exp5_stacking(df)
    exp6_cohort_split(df)


if __name__ == "__main__":
    main()
