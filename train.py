"""Entraîne le pipeline final et le sauvegarde dans model.pkl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    add_features,
)

DATA_PATH = "../data/reservations_azur.csv"
TARGET = "annule"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
META_PATH = Path(__file__).resolve().parent / "model_meta.json"
THRESHOLD = 0.18  # p > 0.18 => appeler (coût 6€, gain 100€, succès 1/3)


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Lignes identiques hors identifiant : une seule occurrence
    subset = [c for c in df.columns if c != "id_reservation"]
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    print(f"Doublons retirés : {before - len(df)} (reste {len(df)})")
    return df


def chronological_split(df: pd.DataFrame, test_ratio: float = 0.2):
    df = df.copy()
    df["date_reservation"] = pd.to_datetime(df["date_reservation"])
    df = df.sort_values("date_reservation").reset_index(drop=True)
    cut = int(len(df) * (1 - test_ratio))
    return df.iloc[:cut], df.iloc[cut:]


def build_pipeline():
    """Préparation + HistGradientBoosting dans un seul objet."""
    prep = FunctionTransformer(add_features, validate=False)
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("features", prep),
            ("columns", columns),
            ("model", model),
        ]
    )


def default_values(train_df: pd.DataFrame) -> dict:
    """Valeurs par défaut (mode / médiane) pour l'app Gradio."""
    feats = add_features(train_df[RAW_FEATURES])
    defaults = {}
    for col in RAW_FEATURES:
        if col in ("date_reservation", "date_arrivee"):
            continue
        if col in CATEGORICAL_FEATURES or train_df[col].dtype == object:
            mode = train_df[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if len(mode) else "INCONNU"
        else:
            defaults[col] = float(train_df[col].median(skipna=True))
    # dates par défaut : médiane du délai
    med_lead = float(feats["delai_jours"].median())
    defaults["delai_jours_defaut"] = med_lead
    return defaults


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    pred = (proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, pred)
    return {
        "roc_auc": float(auc),
        "auprc": float(auprc),
        "threshold": THRESHOLD,
        "confusion_matrix": cm.tolist(),
        "cancel_rate_test": float(y_test.mean()),
        "call_rate": float(pred.mean()),
    }


def main(data_path: str):
    df = load_clean(data_path)
    train_df, test_df = chronological_split(df)

    X_train = train_df[RAW_FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[RAW_FEATURES]
    y_test = test_df[TARGET].astype(int)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Taux annulation train={y_train.mean():.3f} test={y_test.mean():.3f}")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    metrics = evaluate(pipe, X_test, y_test)
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
        f"seuil={THRESHOLD}"
    )
    print("Matrice de confusion (seuil 0.18):\n", np.array(metrics["confusion_matrix"]))

    joblib.dump(pipe, MODEL_PATH)
    meta = {
        "raw_features": RAW_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "threshold": THRESHOLD,
        "metrics": metrics,
        "defaults": default_values(train_df),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "model": "HistGradientBoostingClassifier",
        "python": sys.version.split()[0],
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sauvegardé : {MODEL_PATH.name}, {META_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()
    main(args.data)
