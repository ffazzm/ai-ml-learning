"""
Proyek Data Science End-to-End (Production-Grade) — satu skrip.

Mendemonstrasikan alur kerja profesional: ingest+validasi -> split DULU ->
baseline -> Pipeline anti-leakage -> tuning via CV (metrik benar) -> pilih
threshold di validation -> evaluasi sekali di holdout -> simpan pipeline +
metadata + laporan.

Dataset dibuat sintetis (mixed numeric+categorical, imbalanced) agar
self-contained. Di dunia nyata, ganti `make_dataset()` dengan load dari
SQL/CSV (lihat Modul 12).

Jalankan:  python pipeline.py
Output  :  models/churn_pipeline_v1.joblib, .meta.json, reports/metrics.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Reproducibility — non-negotiable di produksi
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def make_dataset(n: int = 8000) -> pd.DataFrame:
    """Stand-in untuk ingest nyata (SQL/CSV). Data churn sintetis, imbalanced ~12%."""
    rng = np.random.default_rng(SEED)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(10, 150)
    n_complaints = rng.poisson(0.6, n)
    region = rng.choice(["jkt", "bdg", "sby", "mdn"], n, p=[0.4, 0.25, 0.2, 0.15])
    contract = rng.choice(["bulanan", "tahunan", "2-tahun"], n, p=[0.55, 0.3, 0.15])

    # Logit churn: kontrak bulanan, tenure pendek, komplain banyak -> churn naik
    logit = (-2.0
             + 0.9 * (contract == "bulanan")
             - 0.03 * tenure
             + 0.5 * n_complaints
             + 0.01 * (monthly - 70))
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(n),          # identifier -> WAJIB dibuang dari fitur
        "tenure": tenure,
        "monthly_charge": monthly,
        "n_complaints": n_complaints,
        "region": region,
        "contract": contract,
        "target": target,
    })
    # Sisipkan sedikit missing realistis
    df.loc[rng.choice(n, 200, replace=False), "monthly_charge"] = np.nan
    return df


def load_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Data contract — gagalkan lebih awal kalau asumsi tak terpenuhi."""
    assert df.shape[0] > 0, "Data kosong"
    assert {"customer_id", "target"}.issubset(df.columns), "Kolom wajib hilang"
    assert df["target"].isin([0, 1]).all(), "Target harus biner"
    assert df["customer_id"].is_unique, "customer_id duplikat"
    print(f"[validate] {len(df)} baris OK | churn rate = {df['target'].mean():.1%}")
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Semua transform di dalam Pipeline -> di-fit hanya pada fold train (anti-leakage)."""
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])


def pick_threshold(y_val, proba_val) -> float:
    """Pilih threshold yang memaksimalkan F1 di VALIDATION (bukan test)."""
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Ingest + validasi
    df = load_and_validate(make_dataset())

    # 2. SPLIT DULU — test set dikunci, disentuh sekali di akhir
    X = df.drop(columns=["target", "customer_id"])   # buang identifier (anti-leakage)
    y = df["target"]
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    # train vs validation (untuk pilih threshold)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.2, random_state=SEED, stratify=y_tv)

    # 3. Baseline dulu
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base_ap = average_precision_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print(f"[baseline] PR-AUC dummy = {base_ap:.3f}")

    # 4. Tuning via CV pada train (metrik PR-AUC cocok utk imbalance)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    param_dist = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_leaf": [1, 5, 20],
    }
    search = RandomizedSearchCV(
        build_pipeline(X), param_dist, n_iter=12, cv=cv,
        scoring="average_precision", random_state=SEED, n_jobs=-1, refit=True)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print(f"[tuning] best CV PR-AUC = {search.best_score_:.3f} | {search.best_params_}")

    # 5. Pilih threshold di validation
    proba_val = best.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, proba_val)
    print(f"[threshold] dipilih dari validation = {threshold:.3f}")

    # 6. EVALUASI SEKALI di holdout (jangan utak-atik model setelah ini)
    proba_test = best.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= threshold).astype(int)
    test_ap = average_precision_score(y_test, proba_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n[holdout] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"PR-AUC model = {test_ap:.3f}  vs  baseline = {base_ap:.3f}  "
          f"-> {'LULUS (mengalahkan baseline)' if test_ap > base_ap + 0.1 else 'TINJAU ULANG'}")

    # 7. Persist pipeline LENGKAP + metadata
    import joblib
    joblib.dump(best, MODELS_DIR / "churn_pipeline_v1.joblib")
    metadata = {
        "model_version": "1.0.0",
        "created_at": dt.datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "features": list(X.columns),
        "best_params": search.best_params_,
        "threshold": threshold,
        "metrics_holdout": {
            "pr_auc": test_ap,
            "recall_churn": report["1"]["recall"],
            "precision_churn": report["1"]["precision"],
        },
        "baseline_pr_auc": base_ap,
        "seed": SEED,
    }
    (MODELS_DIR / "churn_pipeline_v1.meta.json").write_text(json.dumps(metadata, indent=2))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\n[save] model + metadata tersimpan di {MODELS_DIR}/")


if __name__ == "__main__":
    main()
