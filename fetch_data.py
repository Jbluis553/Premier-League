"""Descarga incremental de datos reales de la Premier League desde TheStatsAPI."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.thestatsapi_client import (TheStatsAPIClient, extract_btts_yes_odd,
                                    extract_xg)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATASET = RAW / "premier_league_dataset.jsonl"


def load_existing() -> dict:
    """match_id -> fila, conservando el cache previo válido."""
    rows = {}
    if not DATASET.exists():
        return rows
    for line_number, line in enumerate(DATASET.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSON invalido en {DATASET}, linea {line_number}: {exc}") from exc
        if isinstance(row, dict) and row.get("match_id"):
            rows[row["match_id"]] = row
    return rows


def main() -> None:
    client = TheStatsAPIClient()
    competition = client.find_premier_league()
    competition_id = competition.get("id")
    if not competition_id:
        raise SystemExit("La respuesta de la API no contiene id de Premier League.")
    print(f"Premier League: {competition.get('name', 'Premier League')} ({competition_id})")

    seasons = client.list_seasons(competition_id)
    if not seasons:
        raise SystemExit("La API no devolvio temporadas para la Premier League")
    cache = load_existing()
    new_rows = 0

    for season in seasons:
        season_id = season.get("id")
        if not season_id:
            continue
        matches = client.list_matches(competition_id, season_id, status="finished")
        for match in matches:
            if match.get("status") != "finished":
                continue
            match_id = match.get("id")
            if not match_id or match_id in cache or not match.get("utc_date"):
                continue
            score = match.get("score") or {}
            if not isinstance(score, dict):
                continue
            score = score.get("regulation") or {}
            if not isinstance(score, dict) or score.get("home") is None or score.get("away") is None:
                continue
            home_team = match.get("home_team") or {}
            away_team = match.get("away_team") or {}
            if not isinstance(home_team, dict) or not isinstance(away_team, dict):
                continue
            if not home_team.get("id") or not away_team.get("id"):
                continue
            try:
                home_goals, away_goals = int(score["home"]), int(score["away"])
            except (TypeError, ValueError):
                continue

            odds = extract_btts_yes_odd(client.match_odds(match_id)) if match.get("odds_available") else None
            home_xg = away_xg = None
            if match.get("xg_available"):
                home_xg, away_xg = extract_xg(client.match_stats(match_id))
            row = {
                "match_id": match_id,
                "season_id": season_id,
                "season_name": season.get("name"),
                "kickoff": match["utc_date"],
                "matchday": match.get("matchday"),
                "home_team_id": home_team["id"],
                "home_team": home_team.get("name"),
                "away_team_id": away_team["id"],
                "away_team": away_team.get("name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "btts_yes_odd": odds,
                "home_xg": home_xg,
                "away_xg": away_xg,
            }
            cache[match_id] = row
            new_rows += 1

    temporary = DATASET.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in sorted(cache.values(), key=lambda item: item.get("kickoff") or ""):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(DATASET)
    print(f"Partidos en dataset: {len(cache)}  |  nuevos: {new_rows}")


if __name__ == "__main__":
    main()
