#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_two_stage_n.sh --n-sim N [options]

Basic examples:
  bash run_two_stage_n.sh --n-sim 100000
  bash run_two_stage_n.sh -n 100000 --out-dir outputs_100000
  bash run_two_stage_n.sh --group-n-sim 100000 --knockout-n-sim 50000

Recent-Elo examples:
  # Use a local match-results CSV and replay the last 365 days match-by-match.
  bash run_two_stage_n.sh -n 10000 \
    --recent-elo \
    --elo-results-csv data/results.csv \
    --elo-end-date 2026-06-09

  # Add a conservative recent-form momentum term after match-by-match updates.
  bash run_two_stage_n.sh -n 10000 \
    --recent-elo \
    --elo-results-csv data/results.csv \
    --elo-trend-weight 0.20

Project-location examples:
  # run from inside the project directory
  bash run_two_stage_n.sh -n 100000

  # run from parent directory
  bash run_two_stage_n.sh -n 100000 --project-dir worldcup2026_two_stage_pipeline_latest

Main options:
  -n, --n-sim N                  Set both group-n-sim and knockout-n-sim to N.
  --group-n-sim N                Number of group-stage simulations.
  --knockout-n-sim N             Number of conditional knockout simulations.
  --out-dir DIR                  Output directory. Default: outputs_two_stage_consensus_<n>.
  --seed SEED                    Base random seed. Default: 20260609.
  --progress-every N             Progress interval. Default: 1000.
  --project-dir DIR              Project directory. Auto-detected if omitted.
  --zip-file FILE                Project ZIP fallback. Default: worldcup2026_two_stage_pipeline_revised.zip.
  --teams-file FILE              Teams CSV. Default: teams.csv; relative paths are resolved under data/ when present.

Recent-Elo preprocessing:
  --recent-elo                   Build <out-dir>/elo_recent/teams_elo_updated.csv before simulation.
  --elo-results-csv PATH_OR_URL  Results CSV path or URL. Required when --recent-elo is used unless the default URL is acceptable.
                                 Default URL: martj42 international_results results.csv.
  --elo-start-date YYYY-MM-DD    Start date for recent match window. Default: end-date minus --elo-lookback-days.
  --elo-end-date YYYY-MM-DD      End date for recent match window. Default: latest date in results CSV.
  --elo-lookback-days N          Lookback length when start-date is omitted. Default: 365.
  --elo-trend-weight X           Momentum term: forecast_elo=end_elo+X*(end_elo-start_elo). Default: 0.
  --elo-home-advantage X         Home boost used only in Elo expected result. Default: 100.
  --elo-unknown-initial X        Initial Elo for non-tournament opponents absent from teams.csv. Default: 1500.
  --elo-aliases-file FILE        Alias CSV under data/. Default: team_name_aliases.csv.
  --elo-out-dir DIR              Output audit directory. Default: <out-dir>/elo_recent.

Dynamic Elo inside W杯 simulation:
  --dynamic-elo                  Update Elo after each simulated W杯 match. Default: on in this wrapper.
  --no-dynamic-elo               Use fixed Elo throughout simulated W杯 matches.
  --dynamic-elo-group-k X         K-factor for group matches. Default: 40.
  --dynamic-elo-r32-k X           K-factor for Round of 32. Default: 60.
  --dynamic-elo-r16-k X           K-factor for Round of 16. Default: 60.
  --dynamic-elo-qf-k X            K-factor for quarter-finals. Default: 60.
  --dynamic-elo-sf-k X            K-factor for semi-finals. Default: 60.
  --dynamic-elo-final-k X         K-factor for final. Default: 60.
  --dynamic-elo-third-place-k X   K-factor for third-place match. Default: 40.
  --dynamic-elo-scale X           Logistic scale for expected result. Default: 400.
  --dynamic-elo-home-advantage X  Home boost in Elo update expectation. Default: reuse --host-advantage.
  --dynamic-elo-knockout-draw-value X
                                 Elo result assigned to the team advancing after drawn knockout score. Default: 0.5.
  --dynamic-elo-no-margin-multiplier
                                 Disable goal-difference multiplier in Elo updates.
  --dynamic-elo-round-delta       Round each Elo update to nearest integer.
  --dynamic-elo-audit-limit N     Rows written to dynamic Elo audit sample CSVs. Default: 5000.

Scenario-selection options:
  --scenario-selection MODE      consensus or topk_exact. Default: consensus.
  --scenario-topk K              Used with topk_exact. Default: 10.
  --write-scenario-limit K       Number of group scenarios to write. Default: 200.

Uncertainty-profile:
  --uncertainty-profile PROFILE  none, mild, moderate, high. Default: moderate.
  --profile PROFILE              Alias for --uncertainty-profile.

Core match-model parameters:
  --base-mu X
  --goal-elo-scale X
  --ko-elo-scale X
  --host-advantage X             Host boost in Elo points. Default: 75 in this wrapper.
  --host-scope MODE              tournament, venue, hybrid, none. Default: venue.
  --fallback-ranking MODE        elo, fifa_rank, name.
  --ci-level X

Uncertainty overrides:
  --team-rating-sd X             Tournament-level team strength SD in Elo points.
  --match-rating-sd X            Match-level shock SD in Elo points.
  --goal-overdispersion X        Extra-Poisson goal volatility.
  --penalty-randomness X         Blend knockout draw resolution toward 50/50.
  --elo-shrink X                 Shrink input Elo gap; lower means more upset-prone.
  --match-shock-dist DIST        normal or student_t.
  --match-shock-df X             Degrees of freedom for student_t shock.
  --upset-prob X                 Probability of explicit underdog shock.
  --upset-underdog-bonus X       Elo bonus for underdog in explicit upset shock.
  --upset-shock-sd X             SD around upset underdog bonus.
  --upset-min-abs-delta X        Minimum abs Elo gap for explicit underdog shock.

Context/factor overrides:
  --contextual-effects           Apply venue-specific heat/humidity/altitude/travel effects. Default: on.
  --no-contextual-effects        Disable venue-specific context effects.
  --match-context-file FILE      CSV under data/ with venue context by match id. Default: match_context.csv.
  --include-static-environment-factors
                                 Also apply environmental factors as static team-wide bonuses. Default: off.
  --confederation-bonus SPEC     Example: CONMEBOL=25,CONCACAF=15,UEFA=-5. Can repeat.
  --team-bonus SPEC              Example: Japan=20. Can repeat.
  --depth-weight X
  --heat-weight X
  --humidity-weight X
  --altitude-weight X
  --travel-weight X
  --upset-potential-weight X
  --upset-resilience-weight X

Visualization options:
  --language LANG                en, ja, both. Default: both.
  --scenario-limit K             Number of selected scenarios to visualize. Default: 1.
  --no-visualize                 Skip compact HTML/SVG visualization.
  --no-compact-visualize         Alias: skip compact HTML/SVG visualization.
  --no-png                       Generate HTML/SVG only.

EOF
}

# ----------------------------
# Defaults
# ----------------------------
PROJECT_DIR=""
ZIP_FILE="worldcup2026_two_stage_pipeline_revised.zip"
TEAMS_FILE="teams.csv"

RECENT_ELO="0"
ELO_RESULTS_CSV="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ELO_START_DATE=""
ELO_END_DATE=""
ELO_LOOKBACK_DAYS="365"
ELO_TREND_WEIGHT="0"
ELO_HOME_ADVANTAGE="100"
ELO_UNKNOWN_INITIAL="1500"
ELO_ALIASES_FILE="team_name_aliases.csv"
ELO_OUT_DIR=""

DYNAMIC_ELO="1"
DYNAMIC_ELO_GROUP_K="40"
DYNAMIC_ELO_R32_K="60"
DYNAMIC_ELO_R16_K="60"
DYNAMIC_ELO_QF_K="60"
DYNAMIC_ELO_SF_K="60"
DYNAMIC_ELO_FINAL_K="60"
DYNAMIC_ELO_THIRD_PLACE_K="40"
DYNAMIC_ELO_SCALE="400"
DYNAMIC_ELO_HOME_ADVANTAGE=""
DYNAMIC_ELO_KNOCKOUT_DRAW_VALUE="0.5"
DYNAMIC_ELO_NO_MARGIN_MULTIPLIER="0"
DYNAMIC_ELO_ROUND_DELTA="0"
DYNAMIC_ELO_AUDIT_LIMIT="5000"

N_SIM=""
GROUP_N_SIM=""
KNOCKOUT_N_SIM=""
OUT_DIR=""
SEED="20260609"
PROGRESS_EVERY="1000"

SCENARIO_SELECTION="consensus"
SCENARIO_TOPK="10"
WRITE_SCENARIO_LIMIT="200"

UNCERTAINTY_PROFILE="moderate"
BASE_MU="1.35"
GOAL_ELO_SCALE="1600"
KO_ELO_SCALE="400"
HOST_ADVANTAGE="75"
HOST_SCOPE="venue"
FALLBACK_RANKING="elo"
CI_LEVEL="0.95"

# Current scenario defaults used in the previous run.
TEAM_RATING_SD=""
MATCH_RATING_SD="150"
GOAL_OVERDISPERSION="0.25"
PENALTY_RANDOMNESS="0.50"
ELO_SHRINK="0.82"
MATCH_SHOCK_DIST="student_t"
MATCH_SHOCK_DF="5"
UPSET_PROB="0.10"
UPSET_UNDERDOG_BONUS="180"
UPSET_SHOCK_SD="80"
UPSET_MIN_ABS_DELTA="50"

CONFEDERATION_BONUS=()
TEAM_BONUS=()
DEPTH_WEIGHT="35"
HEAT_WEIGHT="30"
HUMIDITY_WEIGHT="15"
ALTITUDE_WEIGHT="10"
TRAVEL_WEIGHT="10"
UPSET_POTENTIAL_WEIGHT="0.30"
UPSET_RESILIENCE_WEIGHT="0.20"
CONTEXTUAL_EFFECTS="1"
MATCH_CONTEXT_FILE="match_context.csv"
INCLUDE_STATIC_ENVIRONMENT_FACTORS="0"

LANGUAGE="both"
SCENARIO_LIMIT="1"
DO_VISUALIZE="1"
DO_COMPACT_VISUALIZE="1"
NO_PNG="0"

CALLER_PWD="$(pwd)"

# ----------------------------
# Argument parsing
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    -n|--n-sim)
      N_SIM="$2"; shift 2 ;;
    --group-n-sim)
      GROUP_N_SIM="$2"; shift 2 ;;
    --knockout-n-sim)
      KNOCKOUT_N_SIM="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --progress-every)
      PROGRESS_EVERY="$2"; shift 2 ;;
    --project-dir)
      PROJECT_DIR="$2"; shift 2 ;;
    --zip-file)
      ZIP_FILE="$2"; shift 2 ;;
    --teams-file)
      TEAMS_FILE="$2"; shift 2 ;;

    --recent-elo)
      RECENT_ELO="1"; shift ;;
    --elo-results-csv|--elo-results-file)
      ELO_RESULTS_CSV="$2"; shift 2 ;;
    --elo-start-date)
      ELO_START_DATE="$2"; shift 2 ;;
    --elo-end-date)
      ELO_END_DATE="$2"; shift 2 ;;
    --elo-lookback-days)
      ELO_LOOKBACK_DAYS="$2"; shift 2 ;;
    --elo-trend-weight)
      ELO_TREND_WEIGHT="$2"; shift 2 ;;
    --elo-home-advantage)
      ELO_HOME_ADVANTAGE="$2"; shift 2 ;;
    --elo-unknown-initial|--elo-unknown-initial-elo)
      ELO_UNKNOWN_INITIAL="$2"; shift 2 ;;
    --elo-aliases-file)
      ELO_ALIASES_FILE="$2"; shift 2 ;;
    --elo-out-dir)
      ELO_OUT_DIR="$2"; shift 2 ;;

    --dynamic-elo)
      DYNAMIC_ELO="1"; shift ;;
    --no-dynamic-elo)
      DYNAMIC_ELO="0"; shift ;;
    --dynamic-elo-group-k)
      DYNAMIC_ELO_GROUP_K="$2"; shift 2 ;;
    --dynamic-elo-r32-k)
      DYNAMIC_ELO_R32_K="$2"; shift 2 ;;
    --dynamic-elo-r16-k)
      DYNAMIC_ELO_R16_K="$2"; shift 2 ;;
    --dynamic-elo-qf-k)
      DYNAMIC_ELO_QF_K="$2"; shift 2 ;;
    --dynamic-elo-sf-k)
      DYNAMIC_ELO_SF_K="$2"; shift 2 ;;
    --dynamic-elo-final-k)
      DYNAMIC_ELO_FINAL_K="$2"; shift 2 ;;
    --dynamic-elo-third-place-k)
      DYNAMIC_ELO_THIRD_PLACE_K="$2"; shift 2 ;;
    --dynamic-elo-scale)
      DYNAMIC_ELO_SCALE="$2"; shift 2 ;;
    --dynamic-elo-home-advantage)
      DYNAMIC_ELO_HOME_ADVANTAGE="$2"; shift 2 ;;
    --dynamic-elo-knockout-draw-value)
      DYNAMIC_ELO_KNOCKOUT_DRAW_VALUE="$2"; shift 2 ;;
    --dynamic-elo-no-margin-multiplier)
      DYNAMIC_ELO_NO_MARGIN_MULTIPLIER="1"; shift ;;
    --dynamic-elo-round-delta)
      DYNAMIC_ELO_ROUND_DELTA="1"; shift ;;
    --dynamic-elo-audit-limit)
      DYNAMIC_ELO_AUDIT_LIMIT="$2"; shift 2 ;;

    --scenario-selection)
      SCENARIO_SELECTION="$2"; shift 2 ;;
    --scenario-topk)
      SCENARIO_TOPK="$2"; shift 2 ;;
    --write-scenario-limit)
      WRITE_SCENARIO_LIMIT="$2"; shift 2 ;;

    --uncertainty-profile|--profile)
      UNCERTAINTY_PROFILE="$2"; shift 2 ;;
    --base-mu)
      BASE_MU="$2"; shift 2 ;;
    --goal-elo-scale)
      GOAL_ELO_SCALE="$2"; shift 2 ;;
    --ko-elo-scale)
      KO_ELO_SCALE="$2"; shift 2 ;;
    --host-advantage)
      HOST_ADVANTAGE="$2"; shift 2 ;;
    --host-scope)
      HOST_SCOPE="$2"; shift 2 ;;
    --fallback-ranking)
      FALLBACK_RANKING="$2"; shift 2 ;;
    --ci-level)
      CI_LEVEL="$2"; shift 2 ;;

    --team-rating-sd)
      TEAM_RATING_SD="$2"; shift 2 ;;
    --match-rating-sd)
      MATCH_RATING_SD="$2"; shift 2 ;;
    --goal-overdispersion)
      GOAL_OVERDISPERSION="$2"; shift 2 ;;
    --penalty-randomness)
      PENALTY_RANDOMNESS="$2"; shift 2 ;;
    --elo-shrink)
      ELO_SHRINK="$2"; shift 2 ;;
    --match-shock-dist)
      MATCH_SHOCK_DIST="$2"; shift 2 ;;
    --match-shock-df)
      MATCH_SHOCK_DF="$2"; shift 2 ;;
    --upset-prob)
      UPSET_PROB="$2"; shift 2 ;;
    --upset-underdog-bonus)
      UPSET_UNDERDOG_BONUS="$2"; shift 2 ;;
    --upset-shock-sd)
      UPSET_SHOCK_SD="$2"; shift 2 ;;
    --upset-min-abs-delta)
      UPSET_MIN_ABS_DELTA="$2"; shift 2 ;;

    --contextual-effects)
      CONTEXTUAL_EFFECTS="1"; shift ;;
    --no-contextual-effects)
      CONTEXTUAL_EFFECTS="0"; shift ;;
    --match-context-file)
      MATCH_CONTEXT_FILE="$2"; shift 2 ;;
    --include-static-environment-factors)
      INCLUDE_STATIC_ENVIRONMENT_FACTORS="1"; shift ;;

    --confederation-bonus)
      CONFEDERATION_BONUS+=("$2"); shift 2 ;;
    --team-bonus)
      TEAM_BONUS+=("$2"); shift 2 ;;
    --depth-weight)
      DEPTH_WEIGHT="$2"; shift 2 ;;
    --heat-weight)
      HEAT_WEIGHT="$2"; shift 2 ;;
    --humidity-weight)
      HUMIDITY_WEIGHT="$2"; shift 2 ;;
    --altitude-weight)
      ALTITUDE_WEIGHT="$2"; shift 2 ;;
    --travel-weight)
      TRAVEL_WEIGHT="$2"; shift 2 ;;
    --upset-potential-weight)
      UPSET_POTENTIAL_WEIGHT="$2"; shift 2 ;;
    --upset-resilience-weight)
      UPSET_RESILIENCE_WEIGHT="$2"; shift 2 ;;

    --language)
      LANGUAGE="$2"; shift 2 ;;
    --scenario-limit)
      SCENARIO_LIMIT="$2"; shift 2 ;;
    --no-visualize)
      DO_VISUALIZE="0"; shift ;;
    --no-compact-visualize)
      DO_COMPACT_VISUALIZE="0"; shift ;;
    --no-png)
      NO_PNG="1"; shift ;;

    *)
      echo "[error] Unknown option: $1"
      echo
      usage
      exit 1 ;;
  esac
done

if [[ -n "${N_SIM}" ]]; then
  GROUP_N_SIM="${GROUP_N_SIM:-$N_SIM}"
  KNOCKOUT_N_SIM="${KNOCKOUT_N_SIM:-$N_SIM}"
fi

GROUP_N_SIM="${GROUP_N_SIM:-1000}"
KNOCKOUT_N_SIM="${KNOCKOUT_N_SIM:-1000}"

if [[ -z "${OUT_DIR}" ]]; then
  if [[ "${GROUP_N_SIM}" == "${KNOCKOUT_N_SIM}" ]]; then
    OUT_DIR="outputs_two_stage_consensus_${GROUP_N_SIM}"
  else
    OUT_DIR="outputs_two_stage_consensus_g${GROUP_N_SIM}_k${KNOCKOUT_N_SIM}"
  fi
fi

# If no confederation-bonus option was supplied, use the current scenario default.
if [[ ${#CONFEDERATION_BONUS[@]} -eq 0 ]]; then
  CONFEDERATION_BONUS=("CONMEBOL=25,CONCACAF=15,UEFA=-5")
fi

# ----------------------------
# Auto-detect project directory
# ----------------------------
if [[ -z "${PROJECT_DIR}" ]]; then
  if [[ -f "scripts/two_stage_worldcup_2026.py" && -d "data" ]]; then
    PROJECT_DIR="."
  elif [[ -d "worldcup2026_two_stage_pipeline_latest" ]]; then
    PROJECT_DIR="worldcup2026_two_stage_pipeline_latest"
  elif [[ -d "worldcup2026_two_stage_pipeline_revised" ]]; then
    PROJECT_DIR="worldcup2026_two_stage_pipeline_revised"
  elif [[ -d "worldcup2026_two_stage_pipeline" ]]; then
    PROJECT_DIR="worldcup2026_two_stage_pipeline"
  else
    PROJECT_DIR="worldcup2026_two_stage_pipeline_latest"
  fi
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
  if [[ -f "${ZIP_FILE}" ]]; then
    echo "[setup] ${PROJECT_DIR} not found. Unzipping ${ZIP_FILE}..."
    unzip -q "${ZIP_FILE}"
    if [[ -d "worldcup2026_two_stage_pipeline_latest" ]]; then
      PROJECT_DIR="worldcup2026_two_stage_pipeline_latest"
    elif [[ -d "worldcup2026_two_stage_pipeline_revised" ]]; then
      PROJECT_DIR="worldcup2026_two_stage_pipeline_revised"
    elif [[ -d "worldcup2026_two_stage_pipeline" ]]; then
      PROJECT_DIR="worldcup2026_two_stage_pipeline"
    fi
  else
    echo "[error] Project directory ${PROJECT_DIR} not found, and ZIP ${ZIP_FILE} not found."
    echo "        Run from inside the project directory, or from its parent directory."
    exit 1
  fi
fi

cd "${PROJECT_DIR}"

if [[ ! -f "scripts/two_stage_worldcup_2026.py" ]]; then
  echo "[error] scripts/two_stage_worldcup_2026.py not found in $(pwd)"
  exit 1
fi

if [[ ! -f "scripts/update_elo_from_results.py" && "${RECENT_ELO}" == "1" ]]; then
  echo "[error] scripts/update_elo_from_results.py not found, but --recent-elo was requested."
  exit 1
fi

echo "[project] $(pwd)"
echo "[settings]"
echo "  group-n-sim          = ${GROUP_N_SIM}"
echo "  knockout-n-sim       = ${KNOCKOUT_N_SIM}"
echo "  out-dir              = ${OUT_DIR}"
echo "  scenario-selection   = ${SCENARIO_SELECTION}"
echo "  teams-file           = ${TEAMS_FILE}"
echo "  recent-elo           = ${RECENT_ELO}"
echo "  dynamic-elo          = ${DYNAMIC_ELO}"
echo "  contextual-effects   = ${CONTEXTUAL_EFFECTS}"
echo "  match-context-file   = ${MATCH_CONTEXT_FILE}"
echo "  host-advantage       = ${HOST_ADVANTAGE}"
echo "  host-scope           = ${HOST_SCOPE}"
echo "  uncertainty-profile  = ${UNCERTAINTY_PROFILE}"
echo "  seed                 = ${SEED}"

if [[ "${RECENT_ELO}" == "1" ]]; then
  if [[ -z "${ELO_OUT_DIR}" ]]; then
    ELO_OUT_DIR="${OUT_DIR}/elo_recent"
  fi
  mkdir -p "${ELO_OUT_DIR}"

  ELO_RESULTS_LOCAL="${ELO_RESULTS_CSV}"
  if [[ "${ELO_RESULTS_CSV}" =~ ^https?:// ]]; then
    ELO_RESULTS_LOCAL="${ELO_OUT_DIR}/results.csv"
    echo "[recent elo] downloading ${ELO_RESULTS_CSV}"
    ELO_RESULTS_URL="${ELO_RESULTS_CSV}" ELO_RESULTS_OUT="${ELO_RESULTS_LOCAL}" python3 - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import os
url = os.environ["ELO_RESULTS_URL"]
out = Path(os.environ["ELO_RESULTS_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url, timeout=60) as r:
    out.write_bytes(r.read())
print(f"[recent elo] wrote {out}")
PY
  elif [[ "${ELO_RESULTS_CSV}" != /* && ! -f "${ELO_RESULTS_CSV}" ]]; then
    # Allow paths relative to the caller's original working directory.
    if [[ -f "${CALLER_PWD}/${ELO_RESULTS_CSV}" ]]; then
      ELO_RESULTS_LOCAL="${CALLER_PWD}/${ELO_RESULTS_CSV}"
    fi
  fi

  echo "[recent elo] replay match-by-match Elo updates"
  ELO_ARGS=(
    scripts/update_elo_from_results.py
    --data-dir data
    --teams-file "${TEAMS_FILE}"
    --results-file "${ELO_RESULTS_LOCAL}"
    --aliases-file "${ELO_ALIASES_FILE}"
    --out-dir "${ELO_OUT_DIR}"
    --lookback-days "${ELO_LOOKBACK_DAYS}"
    --trend-weight "${ELO_TREND_WEIGHT}"
    --home-advantage-elo "${ELO_HOME_ADVANTAGE}"
    --unknown-initial-elo "${ELO_UNKNOWN_INITIAL}"
  )
  if [[ -n "${ELO_START_DATE}" ]]; then
    ELO_ARGS+=(--start-date "${ELO_START_DATE}")
  fi
  if [[ -n "${ELO_END_DATE}" ]]; then
    ELO_ARGS+=(--end-date "${ELO_END_DATE}")
  fi
  python3 "${ELO_ARGS[@]}"

  if [[ "${ELO_OUT_DIR}" = /* ]]; then
    TEAMS_FILE="${ELO_OUT_DIR}/teams_elo_updated.csv"
  else
    TEAMS_FILE="$(pwd)/${ELO_OUT_DIR}/teams_elo_updated.csv"
  fi
  echo "[recent elo] simulation teams-file = ${TEAMS_FILE}"
fi

echo "[validate] Annex C"
python3 scripts/validate_annex_c.py --data-dir data

PY_ARGS=(
  scripts/two_stage_worldcup_2026.py
  --teams-file "${TEAMS_FILE}"
  --group-n-sim "${GROUP_N_SIM}"
  --knockout-n-sim "${KNOCKOUT_N_SIM}"
  --scenario-selection "${SCENARIO_SELECTION}"
  --scenario-topk "${SCENARIO_TOPK}"
  --write-scenario-limit "${WRITE_SCENARIO_LIMIT}"
  --seed "${SEED}"
  --uncertainty-profile "${UNCERTAINTY_PROFILE}"
  --base-mu "${BASE_MU}"
  --goal-elo-scale "${GOAL_ELO_SCALE}"
  --ko-elo-scale "${KO_ELO_SCALE}"
  --host-advantage "${HOST_ADVANTAGE}"
  --host-scope "${HOST_SCOPE}"
  --match-context-file "${MATCH_CONTEXT_FILE}"
  --fallback-ranking "${FALLBACK_RANKING}"
  --ci-level "${CI_LEVEL}"
  --goal-overdispersion "${GOAL_OVERDISPERSION}"
  --penalty-randomness "${PENALTY_RANDOMNESS}"
  --elo-shrink "${ELO_SHRINK}"
  --match-rating-sd "${MATCH_RATING_SD}"
  --match-shock-dist "${MATCH_SHOCK_DIST}"
  --match-shock-df "${MATCH_SHOCK_DF}"
  --upset-prob "${UPSET_PROB}"
  --upset-underdog-bonus "${UPSET_UNDERDOG_BONUS}"
  --upset-shock-sd "${UPSET_SHOCK_SD}"
  --upset-min-abs-delta "${UPSET_MIN_ABS_DELTA}"
  --depth-weight "${DEPTH_WEIGHT}"
  --heat-weight "${HEAT_WEIGHT}"
  --humidity-weight "${HUMIDITY_WEIGHT}"
  --altitude-weight "${ALTITUDE_WEIGHT}"
  --travel-weight "${TRAVEL_WEIGHT}"
  --upset-potential-weight "${UPSET_POTENTIAL_WEIGHT}"
  --upset-resilience-weight "${UPSET_RESILIENCE_WEIGHT}"
  --progress-every "${PROGRESS_EVERY}"
  --out-dir "${OUT_DIR}"
)


if [[ "${CONTEXTUAL_EFFECTS}" == "1" ]]; then
  PY_ARGS+=(--contextual-effects)
fi
if [[ "${INCLUDE_STATIC_ENVIRONMENT_FACTORS}" == "1" ]]; then
  PY_ARGS+=(--include-static-environment-factors)
fi

if [[ "${DYNAMIC_ELO}" == "1" ]]; then
  PY_ARGS+=(
    --dynamic-elo
    --dynamic-elo-group-k "${DYNAMIC_ELO_GROUP_K}"
    --dynamic-elo-r32-k "${DYNAMIC_ELO_R32_K}"
    --dynamic-elo-r16-k "${DYNAMIC_ELO_R16_K}"
    --dynamic-elo-qf-k "${DYNAMIC_ELO_QF_K}"
    --dynamic-elo-sf-k "${DYNAMIC_ELO_SF_K}"
    --dynamic-elo-final-k "${DYNAMIC_ELO_FINAL_K}"
    --dynamic-elo-third-place-k "${DYNAMIC_ELO_THIRD_PLACE_K}"
    --dynamic-elo-scale "${DYNAMIC_ELO_SCALE}"
    --dynamic-elo-knockout-draw-value "${DYNAMIC_ELO_KNOCKOUT_DRAW_VALUE}"
    --dynamic-elo-audit-limit "${DYNAMIC_ELO_AUDIT_LIMIT}"
  )
  if [[ -n "${DYNAMIC_ELO_HOME_ADVANTAGE}" ]]; then
    PY_ARGS+=(--dynamic-elo-home-advantage "${DYNAMIC_ELO_HOME_ADVANTAGE}")
  fi
  if [[ "${DYNAMIC_ELO_NO_MARGIN_MULTIPLIER}" == "1" ]]; then
    PY_ARGS+=(--dynamic-elo-no-margin-multiplier)
  fi
  if [[ "${DYNAMIC_ELO_ROUND_DELTA}" == "1" ]]; then
    PY_ARGS+=(--dynamic-elo-round-delta)
  fi
fi

# Only pass team-rating-sd if explicitly supplied; otherwise the profile default is used.
if [[ -n "${TEAM_RATING_SD}" ]]; then
  PY_ARGS+=(--team-rating-sd "${TEAM_RATING_SD}")
fi

# macOS bash 3.2 can raise "unbound variable" for empty arrays under set -u.
# Temporarily disable nounset while expanding optional arrays.
set +u
for spec in "${CONFEDERATION_BONUS[@]}"; do
  if [[ -n "${spec}" ]]; then
    PY_ARGS+=(--confederation-bonus "${spec}")
  fi
done

for spec in "${TEAM_BONUS[@]}"; do
  if [[ -n "${spec}" ]]; then
    PY_ARGS+=(--team-bonus "${spec}")
  fi
done
set -u

echo "[run] two-stage simulation"
python3 "${PY_ARGS[@]}"

if [[ "${DO_VISUALIZE}" == "1" && "${DO_COMPACT_VISUALIZE}" == "1" ]]; then
  if [[ ! -f "scripts/visualize_two_stage_worldcup_2026_compact.py" ]]; then
    echo "[error] scripts/visualize_two_stage_worldcup_2026_compact.py not found."
    exit 1
  fi
  echo "[visualize compact]"
  COMPACT_VIS_ARGS=(
    scripts/visualize_two_stage_worldcup_2026_compact.py
    --input-dir "${OUT_DIR}"
    --output-dir "${OUT_DIR}/visuals_compact"
    --scenario-limit "${SCENARIO_LIMIT}"
    --language "${LANGUAGE}"
  )
  if [[ "${NO_PNG}" == "1" ]]; then
    COMPACT_VIS_ARGS+=(--no-png)
  fi
  python3 "${COMPACT_VIS_ARGS[@]}"
fi

if [[ "${OUT_DIR}" = /* ]]; then
  OUT_DIR_ABS="${OUT_DIR}"
else
  OUT_DIR_ABS="$(pwd)/${OUT_DIR}"
fi

echo "[done]"
echo "Summary:"
echo "  ${OUT_DIR_ABS}/two_stage_summary.md"
echo "Compact HTML/SVG:"
if [[ "${DO_VISUALIZE}" == "1" && "${DO_COMPACT_VISUALIZE}" == "1" ]]; then
  echo "  ${OUT_DIR_ABS}/visuals_compact/index.html"
fi
if [[ "${RECENT_ELO}" == "1" ]]; then
  echo "Recent Elo audit:"
  echo "  $(pwd)/${ELO_OUT_DIR}/team_elo_summary.csv"
  echo "  $(pwd)/${ELO_OUT_DIR}/teams_elo_updated.csv"
fi
