"""Construye features temporales sin usar resultados futuros ni partidos pendientes."""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
REQUIRED_COLUMNS = {"match_id", "season_id", "kickoff", "home_team_id", "away_team_id",
                    "home_goals", "away_goals"}


def load_params() -> dict:
    with open(ROOT / "config" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    for key, default in (("lookback_home_away", 5), ("lookback_any", 10), ("xg_lookback", 5)):
        params[key] = int(params.get(key, default))
        if params[key] < 1:
            raise ValueError(f"{key} debe ser un entero positivo")
    return params


def _team_history(df: pd.DataFrame) -> dict:
    """team_id -> historial ordenado de partidos finalizados y puntuables."""
    played = df[df["home_goals"].notna() & df["away_goals"].notna()]
    history = {}
    for side, team_col, gf_col, ga_col, xg_col in (
        ("home", "home_team_id", "home_goals", "away_goals", "home_xg"),
        ("away", "away_team_id", "away_goals", "home_goals", "away_xg"),
    ):
        cols = ["kickoff", team_col, gf_col, ga_col]
        if xg_col in played.columns:
            cols.append(xg_col)
        part = played[cols].copy()
        part.columns = ["kickoff", "team_id", "gf", "ga"] + (["xg"] if len(cols) == 5 else [])
        part["side"] = side
        if "xg" not in part:
            part["xg"] = float("nan")
        part["btts"] = ((part["gf"] > 0) & (part["ga"] > 0)).astype(int)
        for team_id, team_games in part.dropna(subset=["team_id"]).groupby("team_id", sort=False):
            history.setdefault(team_id, []).append(team_games)
    return {team_id: pd.concat(parts, ignore_index=True).sort_values("kickoff").reset_index(drop=True)
            for team_id, parts in history.items()}


def build_features(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para features: {', '.join(sorted(missing))}")
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
    for col in ("home_goals", "away_goals", "home_xg", "away_xg", "btts_yes_odd"):
        if col not in df:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["kickoff", "match_id", "season_id", "home_team_id", "away_team_id"])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    history = _team_history(df)
    n_home_away = int(p.get("lookback_home_away", 5))
    n_any = int(p.get("lookback_any", 10))
    n_xg = int(p.get("xg_lookback", 5))

    rows = []
    for _, match in df.iterrows():
        features = {}
        for side in ("home", "away"):
            team_id = match[f"{side}_team_id"]
            team_games = history.get(team_id)
            prev = (team_games[team_games["kickoff"] < match["kickoff"]]
                    if team_games is not None else pd.DataFrame())
            same_side = prev[prev["side"] == side].tail(n_home_away) if not prev.empty else prev
            recent = prev.tail(n_any)
            prefix = side
            features[f"{prefix}_gf_same"] = same_side["gf"].mean() if not same_side.empty else float("nan")
            features[f"{prefix}_ga_same"] = same_side["ga"].mean() if not same_side.empty else float("nan")
            features[f"{prefix}_btts_rate"] = recent["btts"].mean() if not recent.empty else float("nan")
            features[f"{prefix}_gf_avg"] = recent["gf"].mean() if not recent.empty else float("nan")
            features[f"{prefix}_ga_avg"] = recent["ga"].mean() if not recent.empty else float("nan")
            features[f"{prefix}_rest_days"] = ((match["kickoff"] - prev["kickoff"].iloc[-1]).total_seconds() / 86400
                                                   if not prev.empty else float("nan"))
            # Media xG histórica: se usa xG del equipo en sus partidos previos, de cualquier sede.
            features[f"{prefix}_xg_avg"] = (recent["xg"].dropna().tail(n_xg).mean()
                                              if not recent.empty else float("nan"))

        has_score = pd.notna(match["home_goals"]) and pd.notna(match["away_goals"])
        features["btts_yes"] = (int(match["home_goals"] > 0 and match["away_goals"] > 0)
                                 if has_score else float("nan"))
        features["kickoff"] = match["kickoff"]
        features["match_id"] = match["match_id"]
        features["season_id"] = match["season_id"]
        if p.get("use_odds_feature", True):
            features["btts_yes_odd"] = match["btts_yes_odd"]
        rows.append(features)
    return pd.DataFrame(rows)


def main() -> None:
    params = load_params()
    raw = ROOT / "data" / "raw" / "premier_league_dataset.jsonl"
    if not raw.exists() or raw.stat().st_size == 0:
        raise SystemExit(f"No existe el dataset de partidos o esta vacio: {raw}")
    try:
        df = pd.read_json(raw, lines=True)
    except ValueError as exc:
        raise SystemExit(f"No se pudo leer el dataset JSONL: {exc}") from exc
    if df.empty:
        raise SystemExit("El dataset no contiene partidos.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"El dataset no contiene columnas requeridas: {', '.join(sorted(missing))}")
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
    df = df.dropna(subset=["kickoff", "season_id"])
    if params.get("train_seasons", 0):
        count = int(params["train_seasons"])
        if count < 1:
            raise SystemExit("train_seasons debe ser 0 (todas) o un entero positivo")
        season_order = (df.sort_values("kickoff").drop_duplicates("season_id", keep="last")["season_id"])
        selected = set(season_order.tail(count))
        df = df[df["season_id"].isin(selected)]
    features = build_features(df, params)
    if features.empty:
        raise SystemExit("No se pudieron construir features: no hay filas de entrada validas.")
    before = len(features)
    # Los primeros partidos de cada equipo pueden carecer de historial; el imputador
    # de train.py los maneja sin descartar datos ni eliminar filas por odds/xG ausentes.
    features = features.dropna(subset=["btts_yes"])
    if features.empty:
        raise SystemExit("No hay partidos finalizados con marcador utilizable para entrenar.")
    features.to_csv(PROCESSED / "features.csv", index=False)
    print(f"Filas: {before} -> {len(features)} partidos finalizados guardados en {PROCESSED / 'features.csv'}")
    print(f"Tasa BTTS-Yes global: {features['btts_yes'].mean():.3f}")


if __name__ == "__main__":
    main()
