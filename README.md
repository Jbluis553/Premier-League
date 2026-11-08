# Premier League BTTS-Yes ML Pipeline

Pipeline automatizado (GitHub Actions) para entrenar modelos de Machine Learning que
predicen el mercado **Both Teams To Score - Yes (BTTS Yes)** de la **Premier League inglesa**.

**Competicion configurada:** `Premier League`, `id = comp_3039`, England (`GB-ENG`), con
`odds_available = true` y `xg_available = true`.

**Datos 100% reales:** todo proviene de [TheStatsAPI](https://www.thestatsapi.com)
(base `https://api.thestatsapi.com/api`, auth `Authorization: Bearer <key>`).
No se usa ningun dato ficticio ni simulacion.

## Arquitectura

```
src/thestatsapi_client.py   Cliente REST con reintentos ante 429 (Retry-After)
src/fetch_data.py           Descarga incremental: partidos finalizados + cuotas BTTS + stats/xG
src/build_features.py       Features con rolling windows SIN leakage (solo partidos previos)
src/train.py                Entrena LR + Gradient Boosting, calibra, evalua, guarda modelo
src/predict.py              Genera predicciones para proximos partidos programados
config/params.yaml          Hiperparametros (ventanas, split, cobertura minima de cuotas)
data/raw/                   Dataset cacheado (premier_league_dataset.jsonl, versionado en git)
models/                     Modelo entrenado + bundle de features
reports/                    Metricas (metrics.json) y reporte.md
docs/thestatsapi_contract.md Resumen del contrato de API usado
```

## Setup

1. Crea el repo en GitHub y sube este codigo.
2. Obtén una API key en https://www.thestatsapi.com/api-keys (trial 7 dias).
3. En GitHub: **Settings > Secrets and variables > Actions > New repository secret**
   - Nombre: `THESTATSAPI_KEY` — Valor: tu key.
4. Ejecuta manualmente el workflow **Train** (workflow_dispatch) o espera al cron semanal.

## Workflows

| Workflow | Trigger | Que hace |
|---|---|---|
| `.github/workflows/train.yml` | Lunes 06:00 UTC + manual + push a main | fetch incremental de datos -> features -> entrenamiento -> sube artefactos -> commitea `data/`, `models/`, `reports/` |
| `.github/workflows/predict.yml` | Diario 12:00 UTC + manual | fetch incremental -> genera `predictions/latest.csv` con probabilidad BTTS-Yes |

## Endpoints TheStatsAPI usados

- `GET /football/competitions/comp_3039` -> detalle de la Premier League (`current_season_id`, flags)
- `GET /football/competitions?search=Premier League` -> busqueda de respaldo validando England (`GB-ENG`)
- `GET /football/competitions/{id}/seasons` -> temporadas disponibles
- `GET /football/matches?competition_id&season_id&status=finished&per_page=100` -> resultados (paginado)
- `GET /football/matches/{match_id}/odds` -> cuotas incluido mercado `both_teams_to_score` (yes/no)
- `GET /football/matches/{match_id}/stats` -> stats del partido (xG donde este disponible)

## Target

`btts_yes = 1` si `home_goals > 0 AND away_goals > 0` (marcador a 90 minutos, `score.regulation`).

## Features (sin leakage, solo informacion disponible ANTES del partido)

- Promedio de goles a favor/en contra del local en sus ultimos N partidos como local
- Promedio de goles a favor/en contra del visitante en sus ultimos N partidos como visitante
- Tasa de BTTS del local y del visitante (ultimos M partidos, cualquier lado)
- xG promedio a favor/en contra (donde `xg_available`)
- Descanso en dias desde el ultimo partido de cada equipo
- Cuota de cierre (closing) del mercado BTTS-Yes como feature de mercado (opcional, `USE_ODDS_FEATURE`)

## Modelos

- Regresion Logistica (baseline calibrada)
- Histogram Gradient Boosting (sklearn)

Se selecciona por **log loss** en test temporal (ultimos 20% cronologico),
se reporta ROC-AUC, Brier score, log loss y curva de calibracion.

## Aviso

Proyecto con fines educativos/analiticos. Las predicciones deportivas no garantizan
resultados ni rentabilidad; apostar conlleva riesgo. Cumple los terminos de uso de
TheStatsAPI (la key solo vive en GitHub Secrets, nunca en el codigo).
