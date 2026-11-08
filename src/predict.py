"""Genera predicciones BTTS-Yes para los proximos partidos programados de la Premier League."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_features import build_features, load_params
from src.thestatsapi_client import TheStatsAPIClient

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
PRED.mkdir(exist_ok=True)


def main() -> None:
    model_path = ROOT / "models" / "btts_yes_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"No existe el modelo entrenado: {model_path}")
    bundle = joblib.load(model_path)
    params = bundle.get("params") or load_params()
    client = TheStatsAPIClient()

    competition = client.find_premier_league()
    competition_id = competition.get("id")
    if not competition_id:
        raise SystemExit("La respuesta de la API no contiene id de Premier League.")
    current_season = competition.get("current_season_id")
    if not current_season:
        seasons = client.list_seasons(competition_id)
        if not seasons:
            raise SystemExit("La API no devolvio temporadas para la Premier League.")
        current_season = next((s.get("id") for s in seasons if s.get("is_current") and s.get("id")), None)
        if not current_season:
            current_season = next((s.get("id") for s in reversed(seasons) if s.get("id")), None)
    if not current_season:
        raise SystemExit("No se pudo determinar la temporada actual de Premier League.")

    upcoming = client.list_matches(competition_id, current_season, status="scheduled")
    if not upcoming:
        print("No hay partidos programados proximamente.")
        return
    valid_upcoming, new_rows = [], []
    for match in upcoming:
        home = match.get("home_team") or {}
        away = match.get("away_team") or {}
        match_id = match.get("id")
        kickoff = match.get("utc_date")
        season_id = match.get("season_id") or current_season
        if not match_id or not kickoff or not home.get("id") or not away.get("id"):
            print(f"Se omite partido con datos incompletos: {match_id or '(sin id)'}")
            continue
        valid_upcoming.append(match)
        new_rows.append({
            "match_id": match_id, "season_id": season_id, "kickoff": kickoff,
            "home_team_id": home["id"], "home_team": home.get("name"),
            "away_team_id": away["id"], "away_team": away.get("name"),
            # NaN evita que un partido futuro se trate como resultado jugado al generar features.
            "home_goals": np.nan, "away_goals": np.nan,
            "btts_yes_odd": None, "home_xg": None, "away_xg": None,
        })
    if not new_rows:
        raise SystemExit("La API devolvio partidos programados, pero ninguno tenia datos suficientes.")

    history_path = ROOT / "data" / "raw" / "premier_league_dataset.jsonl"
    if not history_path.exists() or history_path.stat().st_size == 0:
        raise SystemExit("Falta el historial data/raw/premier_league_dataset.jsonl; ejecuta primero fetch_data.")
    try:
        history = pd.read_json(history_path, lines=True)
    except ValueError as exc:
        raise SystemExit(f"No se pudo leer el historial JSONL: {exc}") from exc
    if history.empty:
        raise SystemExit("El historial de partidos esta vacio.")
    features = build_features(pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True), params)
    target_ids = {match["id"] for match in valid_upcoming}
    features = features[features["match_id"].isin(target_ids)].copy()
    if features.empty:
        raise SystemExit("No se pudieron generar features para los partidos proximos.")

    columns = bundle.get("features")
    if not isinstance(columns, list) or not columns:
        raise SystemExit("El bundle del modelo no contiene una lista de features valida.")
    missing = set(columns) - set(features.columns)
    if missing:
        raise SystemExit(f"Faltan features requeridas por el modelo: {', '.join(sorted(missing))}")
    X = features[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill_values = pd.Series(bundle.get("fill_values", {})).reindex(columns).fillna(0.0)
    X = X.fillna(fill_values).astype(float)
    probabilities = bundle["model"].predict_proba(X)[:, 1]

    meta = {match["id"]: ((match.get("home_team") or {}).get("name"),
                           (match.get("away_team") or {}).get("name")) for match in valid_upcoming}
    output = features[["match_id", "kickoff"]].copy()
    output["home_team"] = output["match_id"].map(lambda match_id: meta[match_id][0])
    output["away_team"] = output["match_id"].map(lambda match_id: meta[match_id][1])
    output["p_btts_yes"] = probabilities.round(4)
    output = output.sort_values("kickoff")
    output.to_csv(PRED / "latest.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
