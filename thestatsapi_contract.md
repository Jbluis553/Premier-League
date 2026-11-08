# Contrato TheStatsAPI usado por este repo

Fuente oficial: https://api.thestatsapi.com/llms.txt (v1.0.0)

- Base URL: `https://api.thestatsapi.com/api`
- Auth: `Authorization: Bearer THESTATSAPI_KEY` en cada request.
- Rate limits: headers `X-RateLimit-Remaining` (por minuto) y `X-Monthly-Quota-Remaining`;
  `429` + `Retry-After` al excederse (el cliente espera y reintenta).
- IDs con prefijo: `comp_` competicion, `sn_` temporada, `tm_` equipo, `mt_` partido.

## Competicion objetivo

```json
{
  "id": "comp_3039",
  "name": "Premier League",
  "country": "England",
  "country_code": "GB",
  "subdivision_code": "GB-ENG",
  "type": "league",
  "has_team_stats": true,
  "has_player_stats": true,
  "odds_available": true,
  "live_odds_available": false,
  "xg_available": true
}
```

## Endpoints utilizados

| Endpoint | Uso en el pipeline |
|---|---|
| `GET /football/competitions/comp_3039` | Detalle de la Premier League: `current_season_id`, `odds_available`, `xg_available` |
| `GET /football/competitions?search=Premier League` | Respaldo para localizar `comp_3039` validando `name`, `country_code = GB` y `subdivision_code = GB-ENG` |
| `GET /football/competitions/{competition_id}/seasons` | Lista de temporadas con `is_current` |
| `GET /football/matches?competition_id&season_id&status=finished&per_page=100&page=N` | Resultados reales (`score.regulation`) |
| `GET /football/matches/{match_id}/odds` | Mercado `both_teams_to_score` (yes/no) |
| `GET /football/matches/{match_id}/stats` | xG por equipo donde `xg_available` |

## Target

`btts_yes = (score.regulation.home > 0) & (score.regulation.away > 0)`

## Regla anti-datos-ficticios

`src/fetch_data.py` SOLO escribe filas cuyo marcador viene del campo `score.regulation`
del endpoint real. Si un partido no tiene marcador, se omite. No existe ningun
generador de datos sinteticos en el repo.
