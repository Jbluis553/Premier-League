"""Entrena y evalua modelos BTTS-Yes con particion temporal."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(ROOT / "config" / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    params = load_params()
    feature_path = ROOT / "data" / "processed" / "features.csv"
    if not feature_path.exists() or feature_path.stat().st_size == 0:
        raise SystemExit(f"No existe features.csv o esta vacio: {feature_path}")
    df = pd.read_csv(feature_path, parse_dates=["kickoff"])
    required = {"btts_yes", "kickoff"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Faltan columnas de entrenamiento: {', '.join(sorted(required - set(df.columns)))}")
    df["btts_yes"] = pd.to_numeric(df["btts_yes"], errors="coerce")
    df = df.dropna(subset=["btts_yes", "kickoff"]).sort_values("kickoff").reset_index(drop=True)
    if df.empty:
        raise SystemExit("No hay filas con objetivo y fecha validos.")
    df["btts_yes"] = df["btts_yes"].astype(int)
    if not set(df["btts_yes"].unique()).issubset({0, 1}):
        raise SystemExit("btts_yes debe contener unicamente 0 y 1.")

    if params.get("use_odds_feature", True) and "btts_yes_odd" in df:
        coverage = float(df["btts_yes_odd"].notna().mean())
        if coverage < float(params.get("min_odds_coverage", 0.5)):
            df = df.drop(columns=["btts_yes_odd"])
            print(f"Cuota BTTS con cobertura {coverage:.0%} < minima: se omite como feature")

    excluded = {"btts_yes", "kickoff", "match_id", "season_id"}
    feature_cols = [col for col in df.columns if col not in excluded]
    if not feature_cols:
        raise SystemExit("No hay columnas predictoras en features.csv.")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = df["btts_yes"]
    fraction = float(params.get("test_fraction", 0.2))
    if not 0 < fraction < 1:
        raise SystemExit("test_fraction debe estar entre 0 y 1.")
    cut = int(len(df) * (1 - fraction))
    if cut < 2 or cut >= len(df) - 1:
        raise SystemExit(f"Datos insuficientes para split temporal: {len(df)} filas.")
    X_train, X_test = X.iloc[:cut].copy(), X.iloc[cut:].copy()
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    if y_train.nunique() != 2:
        raise SystemExit("El segmento de entrenamiento debe contener ejemplos de ambas clases BTTS.")
    class_counts = y_train.value_counts()
    cv = min(3, int(class_counts.min()))
    if cv < 2:
        raise SystemExit("Se necesitan al menos dos ejemplos de cada clase en train para calibrar el modelo.")

    # Ajusta los valores faltantes solo con train; conserva y aplica los mismos valores al predecir.
    fill_values = X_train.median().fillna(0.0)
    X_train = X_train.fillna(fill_values).astype(float)
    X_test = X_test.fillna(fill_values).astype(float)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=int(params.get("random_state", 42))),
    }

    results, best_name, best_model, best_loss = {}, None, None, float("inf")
    for name, estimator in candidates.items():
        calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        calibrated.fit(X_train, y_train)
        probabilities = calibrated.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, probabilities, labels=[0, 1])
        auc = (float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None)
        results[name] = {
            "log_loss": round(float(loss), 4),
            "brier": round(float(brier_score_loss(y_test, probabilities)), 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
        }
        print(f"{name}: {results[name]}")
        if loss < best_loss:
            best_loss, best_name, best_model = loss, name, calibrated

    if best_model is None:
        raise SystemExit("No se pudo entrenar ningun modelo.")
    bundle = {
        "model": best_model,
        "features": feature_cols,
        "fill_values": fill_values.to_dict(),
        "trained_until": str(df["kickoff"].iloc[cut - 1]),
        "params": params,
    }
    joblib.dump(bundle, MODELS / "btts_yes_model.joblib")
    best_probabilities = best_model.predict_proba(X_test)[:, 1]
    if y_test.nunique() == 2:
        prob_true, prob_pred = calibration_curve(y_test, best_probabilities, n_bins=10, strategy="uniform")
        calibration = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    else:
        calibration = {"prob_true": [], "prob_pred": []}
    metrics = {
        "best_model": best_name,
        "results": results,
        "test_period": {"from": str(df["kickoff"].iloc[cut]), "to": str(df["kickoff"].iloc[-1])},
        "test_size": len(X_test),
        "base_rate_btts_yes": round(float(y_test.mean()), 4),
        "calibration": calibration,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# Reporte de entrenamiento", "", f"- Mejor modelo: **{best_name}**", 
             f"- Periodo de test: {metrics['test_period']['from'][:10]} a {metrics['test_period']['to'][:10]} ({len(X_test)} partidos)",
             f"- Tasa base BTTS-Yes en test: {metrics['base_rate_btts_yes']:.1%}", "",
             "| Modelo | Log loss | Brier | ROC-AUC |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        auc = result["roc_auc"] if result["roc_auc"] is not None else "N/A"
        lines.append(f"| {name} | {result['log_loss']} | {result['brier']} | {auc} |")
    (REPORTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Guardado: models/btts_yes_model.joblib, reports/metrics.json, reports/report.md")


if __name__ == "__main__":
    main()
