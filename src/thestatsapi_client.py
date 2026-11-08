"""Cliente minimo para TheStatsAPI con manejo de rate limits (429 / Retry-After)."""
import math
import os
import time

import requests

BASE_URL = "https://api.thestatsapi.com/api"

# Competición objetivo: Premier League (England), según contrato de TheStatsAPI.
PREMIER_LEAGUE_ID = "comp_3039"
PREMIER_LEAGUE_NAME = "Premier League"
PREMIER_LEAGUE_COUNTRY_CODE = "GB"
PREMIER_LEAGUE_SUBDIVISION_CODE = "GB-ENG"


class TheStatsAPIError(RuntimeError):
    pass


class TheStatsAPIClient:
    """Cliente REST. Toda llamada lleva `Authorization: Bearer <key>`."""

    def __init__(self, api_key: str | None = None, max_retries: int = 6, timeout: int = 30):
        self.api_key = api_key or os.environ.get("THESTATSAPI_KEY")
        if not self.api_key:
            raise TheStatsAPIError("Falta configurar la variable THESTATSAPI_KEY.")
        if max_retries < 1:
            raise ValueError("max_retries debe ser al menos 1")
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get(self, path: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt + 1 == self.max_retries:
                    raise TheStatsAPIError(f"Error de red consultando {path}: {exc}") from exc
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 429:
                try:
                    wait = max(0, int(resp.headers.get("Retry-After", "60")))
                except (TypeError, ValueError):
                    wait = min(2 ** attempt, 60)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            try:
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                raise TheStatsAPIError(f"Respuesta invalida consultando {path}: {exc}") from exc
        raise TheStatsAPIError(f"La API sigue limitando solicitudes tras {self.max_retries} intentos: {path}")

    @staticmethod
    def _data_list(response, endpoint: str) -> list:
        if response is None:
            return []
        if isinstance(response, list):
            return response
        if not isinstance(response, dict):
            raise TheStatsAPIError(f"Formato inesperado de respuesta en {endpoint}: se esperaba objeto/lista")
        data = response.get("data", [])
        if data is None:
            return []
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise TheStatsAPIError(f"Formato inesperado de data en {endpoint}")

    def get_competition(self, competition_id: str = PREMIER_LEAGUE_ID) -> dict:
        """Carga el detalle de una competicion por id."""
        response = self._get(f"/football/competitions/{competition_id}")
        data = response.get("data", response) if isinstance(response, dict) else None
        if not isinstance(data, dict) or not data.get("id"):
            raise TheStatsAPIError(f"No se pudo cargar la competicion {competition_id}")
        return data

    @staticmethod
    def _is_premier_league(competition: dict) -> bool:
        """Valida que el objeto sea la Premier League inglesa y no otra liga homonima."""
        if not isinstance(competition, dict):
            return False
        name_ok = str(competition.get("name", "")).strip().casefold() == PREMIER_LEAGUE_NAME.casefold()
        type_ok = str(competition.get("type", "league")).strip().casefold() == "league"
        if competition.get("id") == PREMIER_LEAGUE_ID and name_ok and type_ok:
            return True
        return (
            name_ok
            and type_ok
            and competition.get("country_code") == PREMIER_LEAGUE_COUNTRY_CODE
            and competition.get("subdivision_code") == PREMIER_LEAGUE_SUBDIVISION_CODE
        )

    def find_premier_league(self) -> dict:
        """Devuelve la competicion Premier League inglesa (id comp_3039)."""
        try:
            competition = self.get_competition(PREMIER_LEAGUE_ID)
            if self._is_premier_league(competition):
                return competition
        except TheStatsAPIError:
            pass

        data = self._data_list(
            self._get("/football/competitions", search=PREMIER_LEAGUE_NAME, per_page=50),
            "/football/competitions",
        )
        candidates = [c for c in data if self._is_premier_league(c)]
        for competition in candidates:
            if competition.get("id") == PREMIER_LEAGUE_ID:
                return competition
        if len(candidates) == 1:
            return candidates[0]
        raise TheStatsAPIError(
            "Premier League (England) no encontrada en /football/competitions; "
            f"se esperaba id {PREMIER_LEAGUE_ID}"
        )

    def list_seasons(self, competition_id: str) -> list[dict]:
        return [s for s in self._data_list(
            self._get(f"/football/competitions/{competition_id}/seasons"), "seasons")
                if isinstance(s, dict)]

    def list_matches(self, competition_id: str, season_id: str,
                     status: str | None = None) -> list[dict]:
        """Partidos de una temporada, paginando con per_page=100."""
        out, page = [], 1
        while True:
            response = self._get("/football/matches", competition_id=competition_id,
                                 season_id=season_id, status=status, per_page=100, page=page)
            if response is None:
                break
            if not isinstance(response, dict):
                raise TheStatsAPIError("Formato inesperado de respuesta al consultar partidos")
            batch = self._data_list(response, "/football/matches")
            out.extend(m for m in batch if isinstance(m, dict))
            meta = response.get("meta") or {}
            if not isinstance(meta, dict) or page >= int(meta.get("total_pages") or 1) or not batch:
                break
            page += 1
        return out

    def match_odds(self, match_id: str) -> dict | None:
        response = self._get(f"/football/matches/{match_id}/odds")
        if isinstance(response, dict):
            data = response.get("data", response)
            return data if isinstance(data, dict) else None
        return None

    def match_stats(self, match_id: str) -> dict | None:
        response = self._get(f"/football/matches/{match_id}/stats")
        if isinstance(response, dict):
            data = response.get("data", response)
            return data if isinstance(data, dict) else None
        return None


def _as_dict_list(value) -> list[dict]:
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, (list, tuple)) and not hasattr(value, "__iter__"):
        return []
    return [item for item in value if isinstance(item, dict)]


def extract_btts_yes_odd(odds_data: dict | None) -> float | None:
    """Promedia cuotas 'yes' de BTTS; ignora campos con tipos o valores invalidos."""
    if not isinstance(odds_data, dict):
        return None
    yes_odds = []
    for bookmaker in _as_dict_list(odds_data.get("bookmakers")):
        for market in _as_dict_list(bookmaker.get("markets")):
            name = str(market.get("name", "")).strip().casefold()
            if "both_teams_to_score" not in name and name != "btts":
                continue
            entries = market.get("entries") or market.get("selections") or []
            for entry in _as_dict_list(entries):
                selection = str(entry.get("selection", entry.get("name", ""))).strip().casefold()
                if selection != "yes":
                    continue
                try:
                    odd = float(entry.get("odd", entry.get("odds")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(odd) and odd > 0:
                    yes_odds.append(odd)
    return round(sum(yes_odds) / len(yes_odds), 4) if yes_odds else None


def _as_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _find_xg(node, side: str) -> float | None:
    """Busca xG del equipo local/visitante dentro del JSON de stats."""
    if not isinstance(node, dict):
        return None
    team = node.get(side) or node.get(f"{side}_team") or {}
    if isinstance(team, dict):
        for key in ("expected_goals", "xg", "xG"):
            if team.get(key) is not None:
                value = _as_float(team[key])
                if value is not None:
                    return value
    for key in (f"{side}_xg", f"{side}_expected_goals", f"xg_{side}"):
        if node.get(key) is not None:
            value = _as_float(node[key])
            if value is not None:
                return value
    return None


def extract_xg(stats_data: dict | None) -> tuple[float | None, float | None]:
    if not isinstance(stats_data, dict):
        return None, None
    data = stats_data.get("data", stats_data)
    return _find_xg(data, "home"), _find_xg(data, "away")
