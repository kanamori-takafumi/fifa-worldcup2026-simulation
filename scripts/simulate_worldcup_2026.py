#!/usr/bin/env python3
"""Elo-based Monte Carlo simulator for the FIFA World Cup 2026.

Features
--------
- 48 teams, 12 groups, Round of 32 through Final.
- Group-stage scores are sampled from independent Poisson models whose means
  are functions of the Elo difference.
- Optional uncertainty layers perturb team ratings at tournament level, add
  heavy-tailed match-day shocks, add goal overdispersion, shrink Elo gaps,
  inject explicit underdog shocks, and make drawn knockout matches less
  Elo-deterministic.
- Scenario factors can add confederation/team bonuses and editable team-level
  depth, heat/humidity/altitude adaptation, travel resilience, upset potential,
  and upset resilience.
- Knockout matches use the same 90-minute score model; if the score is level,
  the advancing team is sampled from an Elo-logistic tie-break model blended
  with a 50/50 penalty proxy.
- The eight best third-placed teams are assigned to Round-of-32 slots by a
  direct lookup in FIFA Regulations Annexe C: all 495 combinations.
- Output probabilities include Monte Carlo confidence intervals using Wilson
  intervals. These intervals quantify simulation error, not model error.

Data files expected under --data-dir
------------------------------------
- teams.csv
- team_factors.csv optional
- group_stage.csv
- round32_slots.csv
- knockout_bracket.csv
- annex_c_third_place_assignments.csv

The implementation has no third-party Python dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Callable, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

GROUPS = tuple("ABCDEFGHIJKL")
ANNEX_C_COLUMNS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")

# Annexe C columns identify the group winner facing a third-placed team.
# These are the corresponding Round-of-32 match numbers in data/round32_slots.csv.
ANNEX_C_SLOT_TO_MATCH = {
    "1A": 79,
    "1B": 85,
    "1D": 81,
    "1E": 74,
    "1G": 82,
    "1I": 77,
    "1K": 87,
    "1L": 80,
}

STAGES = (
    "reach_r32",
    "reach_r16",
    "reach_qf",
    "reach_sf",
    "reach_final",
    "runner_up",
    "third_place",
    "champion",
)

ROUND_BY_MATCH: Dict[int, str] = {**{i: "R32" for i in range(73, 89)}}
ROUND_BY_MATCH.update({**{i: "R16" for i in range(89, 97)}})
ROUND_BY_MATCH.update({**{i: "QF" for i in range(97, 101)}})
ROUND_BY_MATCH.update({101: "SF", 102: "SF", 103: "Third place", 104: "Final"})

UNCERTAINTY_PRESETS = {
    # All values are in Elo-rating units except goal_overdispersion,
    # penalty_randomness, elo_shrink and probabilities.
    # team_rating_sd is a persistent tournament-level strength uncertainty for each team.
    # match_rating_sd is an independent match-day shock on the rating difference.
    # match_shock_dist="student_t" gives fatter tails than a Gaussian shock.
    # goal_overdispersion is the variance of a multiplicative Gamma shock on each Poisson mean.
    # penalty_randomness blends the Elo tie-break probability with 0.5 in drawn knockout matches.
    # elo_shrink discounts the input Elo gap itself; values below 1 make favourites less dominant.
    # upset_* injects one-off underdog-favouring events, representing red cards, tactical mismatch,
    # injury/game-state chaos, finishing swings, refereeing variance, or matchup-specific variance.
    "none": {
        "team_rating_sd": 0.0,
        "match_rating_sd": 0.0,
        "match_shock_dist": "normal",
        "match_shock_df": 5.0,
        "goal_overdispersion": 0.0,
        "penalty_randomness": 0.0,
        "elo_shrink": 1.0,
        "upset_prob": 0.0,
        "upset_underdog_bonus": 0.0,
        "upset_shock_sd": 0.0,
    },
    "mild": {
        "team_rating_sd": 50.0,
        "match_rating_sd": 70.0,
        "match_shock_dist": "normal",
        "match_shock_df": 5.0,
        "goal_overdispersion": 0.06,
        "penalty_randomness": 0.15,
        "elo_shrink": 0.95,
        "upset_prob": 0.02,
        "upset_underdog_bonus": 100.0,
        "upset_shock_sd": 50.0,
    },
    "moderate": {
        "team_rating_sd": 85.0,
        "match_rating_sd": 120.0,
        "match_shock_dist": "student_t",
        "match_shock_df": 5.0,
        "goal_overdispersion": 0.18,
        "penalty_randomness": 0.35,
        "elo_shrink": 0.88,
        "upset_prob": 0.06,
        "upset_underdog_bonus": 140.0,
        "upset_shock_sd": 60.0,
    },
    "high": {
        "team_rating_sd": 140.0,
        "match_rating_sd": 180.0,
        "match_shock_dist": "student_t",
        "match_shock_df": 4.0,
        "goal_overdispersion": 0.30,
        "penalty_randomness": 0.55,
        "elo_shrink": 0.75,
        "upset_prob": 0.12,
        "upset_underdog_bonus": 190.0,
        "upset_shock_sd": 90.0,
    },
}

ROUND_DEPTH_MULTIPLIER = {
    "Group": 0.15,
    "R32": 0.30,
    "R16": 0.45,
    "QF": 0.65,
    "SF": 0.85,
    "Third place": 0.80,
    "Final": 1.00,
}

ELO_GAP_BINS = (
    (0.0, 50.0, "000-050"),
    (50.0, 100.0, "050-100"),
    (100.0, 200.0, "100-200"),
    (200.0, 300.0, "200-300"),
    (300.0, 400.0, "300-400"),
    (400.0, math.inf, "400+"),
)

@dataclass(frozen=True)
class Team:
    name: str
    group: str
    position: str
    confederation: str
    elo: float
    host: bool
    fifa_rank: Optional[int] = None


def team_with_elo(team: Team, elo: float) -> Team:
    """Return a copy of ``team`` with a replaced Elo rating.

    The simulator normally treats ``Team.elo`` as a fixed pre-tournament input.
    Dynamic-Elo simulation keeps mutable ratings in a separate dictionary and
    uses this helper to pass the current rating into the existing match model
    without changing group/host/federation metadata.
    """
    return Team(
        name=team.name,
        group=team.group,
        position=team.position,
        confederation=team.confederation,
        elo=float(elo),
        host=team.host,
        fifa_rank=team.fifa_rank,
    )


def teams_with_elos(teams: Mapping[str, Team], ratings: Mapping[str, float]) -> Dict[str, Team]:
    return {name: team_with_elo(team, ratings.get(name, team.elo)) for name, team in teams.items()}


def result_points(score_for: int, score_against: int) -> float:
    if score_for > score_against:
        return 1.0
    if score_for == score_against:
        return 0.5
    return 0.0


def goal_difference_multiplier(abs_goal_diff: int) -> float:
    """World-football-Elo-style goal-difference multiplier."""
    if abs_goal_diff <= 1:
        return 1.0
    if abs_goal_diff == 2:
        return 1.5
    return (11.0 + abs_goal_diff) / 8.0


def elo_update_score_team1(
    goals1: int,
    goals2: int,
    advancer: Optional[int] = None,
    knockout_draw_value: float = 0.5,
) -> float:
    """Observed score W for team1 in the Elo update.

    For ordinary draws W=0.5.  In knockout matches whose simulated 90-minute
    score is level, ``knockout_draw_value`` controls how much credit is assigned
    to the team that advances: 0.5 means no rating reward for a penalty-proxy
    advancement; 1.0 treats it as a full win; intermediate values are allowed.
    """
    if goals1 > goals2:
        return 1.0
    if goals2 > goals1:
        return 0.0
    if advancer == 1:
        return knockout_draw_value
    if advancer == 2:
        return 1.0 - knockout_draw_value
    return 0.5


def update_dynamic_elos(
    ratings: Dict[str, float],
    team1: Team,
    team2: Team,
    goals1: int,
    goals2: int,
    *,
    k_factor: float,
    elo_scale: float = 400.0,
    host_advantage: float = 0.0,
    use_goal_multiplier: bool = True,
    advancer: Optional[int] = None,
    knockout_draw_value: float = 0.5,
    round_delta: bool = False,
    host_delta_override: Optional[float] = None,
    context_delta: float = 0.0,
) -> Dict[str, float]:
    """Update mutable Elo ratings after one simulated match.

    Returns a compact audit dictionary.  ``ratings`` is modified in-place.
    The expected score uses current dynamic ratings plus a host adjustment only
    for computing expectation; the stored rating itself is not permanently
    shifted by host status.
    """
    if k_factor <= 0.0:
        old1 = float(ratings.get(team1.name, team1.elo))
        old2 = float(ratings.get(team2.name, team2.elo))
        return {
            "dynamic_elo_enabled": 0,
            "team1_elo_pre": old1,
            "team2_elo_pre": old2,
            "team1_elo_post": old1,
            "team2_elo_post": old2,
            "team1_elo_change": 0.0,
            "team2_elo_change": 0.0,
            "team1_expected_result": math.nan,
            "team2_expected_result": math.nan,
            "team1_result_for_elo": math.nan,
            "team2_result_for_elo": math.nan,
            "elo_update_k": k_factor,
            "elo_update_g": math.nan,
            "elo_update_host_delta": math.nan,
            "elo_update_context_delta": math.nan,
        }

    old1 = float(ratings.get(team1.name, team1.elo))
    old2 = float(ratings.get(team2.name, team2.elo))
    host_delta = (host_advantage if team1.host else 0.0) - (host_advantage if team2.host else 0.0)
    if host_delta_override is not None:
        host_delta = float(host_delta_override)
    context_delta = float(context_delta)
    expected1 = elo_logistic((old1 - old2) + host_delta + context_delta, elo_scale)
    score1 = elo_update_score_team1(goals1, goals2, advancer=advancer, knockout_draw_value=knockout_draw_value)
    g = goal_difference_multiplier(abs(int(goals1) - int(goals2))) if use_goal_multiplier else 1.0
    delta1 = float(k_factor) * g * (score1 - expected1)
    if round_delta:
        delta1 = math.floor(delta1 + 0.5) if delta1 >= 0.0 else -math.floor(abs(delta1) + 0.5)
    ratings[team1.name] = old1 + delta1
    ratings[team2.name] = old2 - delta1
    return {
        "dynamic_elo_enabled": 1,
        "team1_elo_pre": old1,
        "team2_elo_pre": old2,
        "team1_elo_post": ratings[team1.name],
        "team2_elo_post": ratings[team2.name],
        "team1_elo_change": delta1,
        "team2_elo_change": -delta1,
        "team1_expected_result": expected1,
        "team2_expected_result": 1.0 - expected1,
        "team1_result_for_elo": score1,
        "team2_result_for_elo": 1.0 - score1,
        "elo_update_k": float(k_factor),
        "elo_update_g": g,
        "elo_update_host_delta": host_delta,
        "elo_update_context_delta": context_delta,
    }


@dataclass(frozen=True)
class TeamFactors:
    """Editable scenario factors on roughly [-1, 1] scale.

    They are not measured truth by default. Treat them as prior/scenario inputs.
    Positive values help the team when the corresponding CLI weight is positive.
    """
    depth: float = 0.0
    heat_adapt: float = 0.0
    humidity_adapt: float = 0.0
    altitude_adapt: float = 0.0
    travel_resilience: float = 0.0
    upset_potential: float = 0.0
    upset_resilience: float = 0.0


@dataclass(frozen=True)
class MatchContext:
    """Editable venue/match context on roughly [0, 1] scale.

    The bundled data are intentionally heuristic scenario inputs, not measured
    meteorological forecasts.  heat_index, humidity_index and altitude_index
    represent the stress level of the venue/date slot; travel_index represents
    generic travel/time-zone/logistics stress.
    """
    key: str = ""
    venue: str = ""
    city: str = ""
    country: str = ""
    region: str = ""
    heat_index: float = 0.0
    humidity_index: float = 0.0
    altitude_index: float = 0.0
    travel_index: float = 0.0
    note: str = ""


@dataclass
class Record:
    team: str
    pts: int = 0
    gf: int = 0
    ga: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    conduct_score: int = 0  # not simulated; kept for compatibility with FIFA tie-breaker structure

    @property
    def gd(self) -> int:
        return self.gf - self.ga


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def resolve_data_path(data_dir: Path, value: str | Path) -> Path:
    """Resolve an input path relative to data_dir when that file exists.

    This lets the simulator use the default data/teams.csv while also accepting
    generated files such as outputs_x/elo_recent/teams_elo_updated.csv.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = data_dir / path
    if candidate.exists():
        return candidate
    return path


def load_teams(path: Path) -> Dict[str, Team]:
    rows = read_csv(path)
    out: Dict[str, Team] = {}
    for r in rows:
        fifa_rank_raw = r.get("fifa_rank", "")
        fifa_rank = int(fifa_rank_raw) if str(fifa_rank_raw).strip() else None
        team = Team(
            name=r["team"],
            group=r["group"],
            position=r["position"],
            confederation=r.get("confederation", ""),
            elo=float(r["elo"]),
            host=bool(int(r.get("host", "0"))),
            fifa_rank=fifa_rank,
        )
        if team.group not in GROUPS:
            raise ValueError(f"Unknown group for {team.name}: {team.group}")
        out[team.name] = team
    if len(out) != 48:
        raise ValueError(f"Expected 48 teams, found {len(out)} in {path}")
    return out


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def normal_z(ci_level: float) -> float:
    if not (0.0 < ci_level < 1.0):
        raise ValueError("--ci-level must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + ci_level / 2.0)


def wilson_interval(k: int, n: int, z: float) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def add_prob(row: Dict[str, object], prefix: str, k: int, n: int, z: float) -> None:
    lo, hi = wilson_interval(k, n, z)
    row[prefix] = k / n if n else math.nan
    row[f"{prefix}_ci_low"] = lo
    row[f"{prefix}_ci_high"] = hi
    row[f"{prefix}_count"] = k


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def parse_bonus_entries(entries: Optional[Sequence[str]]) -> Dict[str, float]:
    """Parse repeated CLI entries such as ["CONMEBOL=30,UEFA=-10"]."""
    out: Dict[str, float] = {}
    for entry in entries or []:
        for part in str(entry).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Bonus entry must have NAME=VALUE format: {part!r}")
            key, value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Empty bonus key in entry: {part!r}")
            out[key] = float(value)
    return out


def float_cell(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def load_team_factors(path: Path, teams: Mapping[str, Team]) -> Dict[str, TeamFactors]:
    """Load optional team_factors.csv; missing file or missing teams default to neutral."""
    out: Dict[str, TeamFactors] = {name: TeamFactors() for name in teams}
    if not path.exists():
        return out
    for r in read_csv(path):
        name = r.get("team", "").strip()
        if not name:
            continue
        if name not in teams:
            raise ValueError(f"Unknown team in team factors file {path}: {name}")
        out[name] = TeamFactors(
            depth=float_cell(r, "depth"),
            heat_adapt=float_cell(r, "heat_adapt"),
            humidity_adapt=float_cell(r, "humidity_adapt"),
            altitude_adapt=float_cell(r, "altitude_adapt"),
            travel_resilience=float_cell(r, "travel_resilience"),
            upset_potential=float_cell(r, "upset_potential"),
            upset_resilience=float_cell(r, "upset_resilience"),
        )
    return out


def load_match_contexts(path: Path) -> Dict[str, MatchContext]:
    """Load optional match_context.csv keyed by group match_id or knockout match_no."""
    if not path.exists():
        return {}
    out: Dict[str, MatchContext] = {}
    for r in read_csv(path):
        key = str(r.get("match_key") or r.get("match_id") or r.get("match_no") or "").strip()
        if not key:
            continue
        out[key] = MatchContext(
            key=key,
            venue=str(r.get("venue", "")).strip(),
            city=str(r.get("city", "")).strip(),
            country=str(r.get("country", "")).strip(),
            region=str(r.get("region", "")).strip(),
            heat_index=float_cell(r, "heat_index"),
            humidity_index=float_cell(r, "humidity_index"),
            altitude_index=float_cell(r, "altitude_index"),
            travel_index=float_cell(r, "travel_index"),
            note=str(r.get("note", "")).strip(),
        )
    return out


def match_context_for_key(contexts: Optional[Mapping[str, MatchContext]], key: object) -> MatchContext:
    if not contexts:
        return MatchContext(key=str(key))
    return contexts.get(str(key), MatchContext(key=str(key)))


def team_is_local_host(team: Team, context: Optional[MatchContext]) -> bool:
    if context is None or not context.country:
        return bool(team.host)
    # Current host-team names coincide with host-country names in teams.csv.
    return bool(team.host and team.name == context.country)


def host_indicator(team: Team, context: Optional[MatchContext], host_scope: str) -> float:
    if host_scope == "none":
        return 0.0
    if host_scope == "tournament":
        return 1.0 if team.host else 0.0
    if host_scope == "venue":
        return 1.0 if team_is_local_host(team, context) else 0.0
    if host_scope == "hybrid":
        # Half credit for being a co-host anywhere in the tournament, full credit
        # when playing in the team's own host country.
        if team_is_local_host(team, context):
            return 1.0
        return 0.5 if team.host else 0.0
    raise ValueError(f"Unknown host scope: {host_scope}")


def compute_context_delta(
    team1: Team,
    team2: Team,
    team_factors: Optional[Mapping[str, TeamFactors]],
    context: Optional[MatchContext],
    heat_weight: float,
    humidity_weight: float,
    altitude_weight: float,
    travel_weight: float,
) -> Tuple[float, float, float, float, float]:
    """Venue-conditioned environmental/logistics adjustment in Elo points."""
    if team_factors is None or context is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    f1 = team_factors.get(team1.name, TeamFactors())
    f2 = team_factors.get(team2.name, TeamFactors())
    heat_delta = heat_weight * context.heat_index * (f1.heat_adapt - f2.heat_adapt)
    humidity_delta = humidity_weight * context.humidity_index * (f1.humidity_adapt - f2.humidity_adapt)
    altitude_delta = altitude_weight * context.altitude_index * (f1.altitude_adapt - f2.altitude_adapt)
    travel_delta = travel_weight * context.travel_index * (f1.travel_resilience - f2.travel_resilience)
    total = heat_delta + humidity_delta + altitude_delta + travel_delta
    return total, heat_delta, humidity_delta, altitude_delta, travel_delta


def compute_static_factor_adjustments(
    teams: Mapping[str, Team],
    factors: Mapping[str, TeamFactors],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Static Elo-point adjustments from scenario inputs, excluding stage-dependent depth.

    When contextual venue effects are enabled, heat/humidity/altitude/travel
    weights are applied match-by-match from match_context.csv and are therefore
    omitted here unless --include-static-environment-factors is explicitly used.
    """
    confed_bonus = parse_bonus_entries(args.confederation_bonus)
    team_bonus = parse_bonus_entries(args.team_bonus)
    use_static_env = (not getattr(args, "contextual_effects", False)) or getattr(args, "include_static_environment_factors", False)
    out: Dict[str, float] = {}
    for name, team in teams.items():
        f = factors.get(name, TeamFactors())
        env = 0.0
        if use_static_env:
            env = (
                args.heat_weight * f.heat_adapt
                + args.humidity_weight * f.humidity_adapt
                + args.altitude_weight * f.altitude_adapt
                + args.travel_weight * f.travel_resilience
            )
        out[name] = confed_bonus.get(team.confederation, 0.0) + team_bonus.get(name, 0.0) + env
    return out


def sample_student_t(rng: random.Random, df: float) -> float:
    if df <= 0.0:
        raise ValueError("Student-t degrees of freedom must be positive")
    z = rng.gauss(0.0, 1.0)
    chi2 = rng.gammavariate(df / 2.0, 2.0)
    return z / math.sqrt(chi2 / df)


def sample_match_shock(rng: random.Random, sd: float, dist: str, df: float) -> float:
    if sd <= 0.0:
        return 0.0
    if dist == "normal":
        return rng.gauss(0.0, sd)
    if dist == "student_t":
        # Standard t has variance df/(df-2). Rescale to have approximately the requested SD.
        raw = sample_student_t(rng, df)
        if df > 2.0:
            raw *= math.sqrt((df - 2.0) / df)
        return sd * raw
    raise ValueError(f"Unknown match shock distribution: {dist}")


def elo_gap_bin(gap: float) -> str:
    for lo, hi, label in ELO_GAP_BINS:
        if lo <= gap < hi:
            return label
    return "unknown"


def upset_diagnostics(
    team1: Team,
    team2: Team,
    winner: str,
    pre_upset_delta: float,
    upset_applied: bool,
    upset_adjustment: float,
) -> Dict[str, object]:
    base_delta = team1.elo - team2.elo
    gap = abs(base_delta)
    if abs(base_delta) < 1e-12:
        favourite = ""
        underdog = ""
    elif base_delta > 0:
        favourite = team1.name
        underdog = team2.name
    else:
        favourite = team2.name
        underdog = team1.name
    if not favourite:
        underdog_win = 0
        favourite_nonwin = 0
    elif not winner:
        underdog_win = 0
        favourite_nonwin = 1
    elif winner == underdog:
        underdog_win = 1
        favourite_nonwin = 1
    else:
        underdog_win = 0
        favourite_nonwin = 0
    return {
        "base_elo_delta": round(base_delta, 3),
        "base_elo_gap": round(gap, 3),
        "elo_gap_bin": elo_gap_bin(gap),
        "elo_favorite": favourite,
        "elo_underdog": underdog,
        "underdog_win_by_elo": underdog_win,
        "favorite_nonwin_by_elo": favourite_nonwin,
        "pre_upset_delta": round(pre_upset_delta, 3),
        "upset_applied": int(upset_applied),
        "upset_adjustment": round(upset_adjustment, 3),
    }


def accumulate_upset_metrics(metric_counts: DefaultDict[Tuple[str, str], Counter], row: Mapping[str, object]) -> None:
    """Aggregate model-upset diagnostics by round and Elo-gap bin."""
    if not row.get("elo_favorite"):
        return
    for key in (("ALL", "ALL"), (str(row.get("round", "")), "ALL"), ("ALL", str(row.get("elo_gap_bin", "")))):
        c = metric_counts[key]
        c["total"] += 1
        c["underdog_win"] += int(row.get("underdog_win_by_elo", 0))
        c["favorite_nonwin"] += int(row.get("favorite_nonwin_by_elo", 0))
        c["upset_applied"] += int(row.get("upset_applied", 0))


# ---------------------------------------------------------------------------
# Match model
# ---------------------------------------------------------------------------

def elo_logistic(delta: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-delta / scale))


def init_dynamic_elos(
    teams: Mapping[str, Team],
    initial_elos: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Create the mutable Elo state used inside one simulated tournament path."""
    if initial_elos is None:
        return {name: float(t.elo) for name, t in teams.items()}
    return {name: float(initial_elos.get(name, t.elo)) for name, t in teams.items()}


def team_current_elo(team: Team, dynamic_elos: Optional[Mapping[str, float]] = None) -> float:
    if dynamic_elos is None:
        return float(team.elo)
    return float(dynamic_elos.get(team.name, team.elo))


def dynamic_elo_margin_multiplier(goal_diff: int) -> float:
    """Football-Elo-style margin multiplier for simulated rating updates."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def dynamic_elo_update_row(
    *,
    dynamic_elos: Optional[MutableMapping[str, float]],
    team1: Team,
    team2: Team,
    goals1: int,
    goals2: int,
    round_name: str,
    advanced_team: str,
    host_advantage: float,
    k_factor: float,
    scale: float,
    use_margin: bool,
    shootout_win_value: float,
) -> Dict[str, object]:
    """Update the mutable Elo state after one simulated match and return diagnostics.

    The predictive model first uses the current Elo state to simulate the match.
    This function is called only after the simulated result is known, so the new
    Elo values can affect later matches in the same simulated tournament path.
    """
    if dynamic_elos is None:
        return {}

    pre1 = float(dynamic_elos.get(team1.name, team1.elo))
    pre2 = float(dynamic_elos.get(team2.name, team2.elo))
    host_delta = (host_advantage if team1.host else 0.0) - (host_advantage if team2.host else 0.0)
    expected1 = elo_logistic(pre1 - pre2 + host_delta, scale)

    if goals1 > goals2:
        actual1 = 1.0
        result_type = "win_loss"
    elif goals1 < goals2:
        actual1 = 0.0
        result_type = "loss_win"
    else:
        if round_name == "Group" or not advanced_team:
            actual1 = 0.5
            result_type = "draw"
        else:
            actual1 = shootout_win_value if advanced_team == team1.name else (1.0 - shootout_win_value)
            result_type = "draw_advancement"

    margin = dynamic_elo_margin_multiplier(goals1 - goals2) if use_margin else 1.0
    update1 = float(k_factor) * margin * (actual1 - expected1)
    post1 = pre1 + update1
    post2 = pre2 - update1
    dynamic_elos[team1.name] = post1
    dynamic_elos[team2.name] = post2

    return {
        "dynamic_elo_enabled": 1,
        "elo1_pre": round(pre1, 3),
        "elo2_pre": round(pre2, 3),
        "elo1_post": round(post1, 3),
        "elo2_post": round(post2, 3),
        "elo1_update": round(update1, 3),
        "elo2_update": round(-update1, 3),
        "elo_expected1": round(expected1, 5),
        "elo_actual1": round(actual1, 3),
        "elo_update_k": round(float(k_factor), 3),
        "elo_update_margin_multiplier": round(margin, 3),
        "elo_update_result_type": result_type,
    }


def poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth Poisson sampler; adequate for the small football goal means here."""
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def expected_goals_from_delta(delta: float, base_mu: float, goal_elo_scale: float) -> Tuple[float, float]:
    """Convert an Elo-rating difference into two independent Poisson means.

    If the teams have equal effective rating, both expected goals are base_mu.
    If team1 has a +400 rating advantage and goal_elo_scale=1600, the
    multiplier is 10**0.25.
    """
    multiplier = 10.0 ** (delta / goal_elo_scale)
    return base_mu * multiplier, base_mu / multiplier


def expected_goals(elo1: float, elo2: float, base_mu: float, goal_elo_scale: float) -> Tuple[float, float]:
    return expected_goals_from_delta(elo1 - elo2, base_mu, goal_elo_scale)


def sample_tournament_rating_adjustments(
    teams: Mapping[str, Team],
    rng: random.Random,
    team_rating_sd: float,
) -> Dict[str, float]:
    """Draw persistent team-level rating errors for one simulated tournament.

    This is the key layer that prevents the simulation from treating the input
    Elo table as ground truth.  The same draw is used for a team throughout the
    tournament, so it represents tournament-form / rating-estimation uncertainty
    rather than independent match noise.
    """
    if team_rating_sd <= 0.0:
        return {name: 0.0 for name in teams}
    return {name: rng.gauss(0.0, team_rating_sd) for name in teams}


def gamma_perturb_mean(lam: float, rng: random.Random, overdispersion: float) -> float:
    """Apply a mean-one Gamma shock to a Poisson mean.

    overdispersion is Var(U) where U has E[U]=1 and lambda' = lambda * U.
    The marginal goal distribution is then a Gamma-Poisson mixture.
    """
    if overdispersion <= 0.0:
        return lam
    shape = 1.0 / overdispersion
    scale = overdispersion
    return lam * rng.gammavariate(shape, scale)


def effective_elo(
    team: Team,
    host_advantage: float,
    rating_adjustments: Optional[Mapping[str, float]] = None,
    dynamic_elos: Optional[Mapping[str, float]] = None,
) -> float:
    rating_noise = 0.0 if rating_adjustments is None else rating_adjustments.get(team.name, 0.0)
    return team_current_elo(team, dynamic_elos) + rating_noise + (host_advantage if team.host else 0.0)


def compute_effective_delta(
    team1: Team,
    team2: Team,
    host_advantage: float,
    elo_shrink: float,
    rating_adjustments: Optional[Mapping[str, float]],
    static_factor_adjustments: Optional[Mapping[str, float]],
    team_factors: Optional[Mapping[str, TeamFactors]],
    depth_weight: float,
    depth_multiplier: float,
    dynamic_elos: Optional[Mapping[str, float]] = None,
    match_context: Optional[MatchContext] = None,
    contextual_effects: bool = False,
    host_scope: str = "tournament",
    heat_weight: float = 0.0,
    humidity_weight: float = 0.0,
    altitude_weight: float = 0.0,
    travel_weight: float = 0.0,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Return effective rating difference and its main components."""
    adj1 = 0.0 if rating_adjustments is None else rating_adjustments.get(team1.name, 0.0)
    adj2 = 0.0 if rating_adjustments is None else rating_adjustments.get(team2.name, 0.0)
    fadj1 = 0.0 if static_factor_adjustments is None else static_factor_adjustments.get(team1.name, 0.0)
    fadj2 = 0.0 if static_factor_adjustments is None else static_factor_adjustments.get(team2.name, 0.0)
    host_delta = host_advantage * (host_indicator(team1, match_context, host_scope) - host_indicator(team2, match_context, host_scope))
    depth1 = 0.0
    depth2 = 0.0
    if team_factors is not None and depth_weight != 0.0 and depth_multiplier != 0.0:
        depth1 = team_factors.get(team1.name, TeamFactors()).depth
        depth2 = team_factors.get(team2.name, TeamFactors()).depth
    base_delta = elo_shrink * (team_current_elo(team1, dynamic_elos) - team_current_elo(team2, dynamic_elos))
    rating_delta = adj1 - adj2
    factor_delta = fadj1 - fadj2
    depth_delta = depth_weight * depth_multiplier * (depth1 - depth2)
    context_delta = heat_delta = humidity_delta = altitude_delta = travel_delta = 0.0
    if contextual_effects:
        context_delta, heat_delta, humidity_delta, altitude_delta, travel_delta = compute_context_delta(
            team1, team2, team_factors, match_context, heat_weight, humidity_weight, altitude_weight, travel_weight
        )
    delta = base_delta + host_delta + rating_delta + factor_delta + depth_delta + context_delta
    return (
        delta, base_delta, rating_delta, factor_delta, depth_delta,
        context_delta, heat_delta, humidity_delta, altitude_delta, travel_delta, host_delta
    )


def apply_explicit_upset_shock(
    delta: float,
    team1: Team,
    team2: Team,
    rng: random.Random,
    upset_prob: float,
    upset_underdog_bonus: float,
    upset_shock_sd: float,
    upset_min_abs_delta: float,
    team_factors: Optional[Mapping[str, TeamFactors]],
    upset_potential_weight: float,
    upset_resilience_weight: float,
) -> Tuple[float, bool, float]:
    """Occasionally give the current effective underdog a one-off Elo bonus."""
    if upset_prob <= 0.0 or upset_underdog_bonus <= 0.0:
        return delta, False, 0.0
    if abs(delta) < upset_min_abs_delta:
        return delta, False, 0.0
    if rng.random() >= upset_prob:
        return delta, False, 0.0

    if delta > 0.0:
        underdog, favourite = team2, team1
        sign = -1.0
    else:
        underdog, favourite = team1, team2
        sign = 1.0

    scale = 1.0
    if team_factors is not None:
        uf = team_factors.get(underdog.name, TeamFactors())
        ff = team_factors.get(favourite.name, TeamFactors())
        scale += upset_potential_weight * uf.upset_potential
        scale -= upset_resilience_weight * ff.upset_resilience
        scale = max(0.0, scale)
    bonus = max(0.0, rng.gauss(upset_underdog_bonus * scale, upset_shock_sd))
    return delta + sign * bonus, True, sign * bonus


def simulate_score(
    team1: Team,
    team2: Team,
    rng: random.Random,
    base_mu: float,
    goal_elo_scale: float,
    host_advantage: float,
    rating_adjustments: Optional[Mapping[str, float]] = None,
    static_factor_adjustments: Optional[Mapping[str, float]] = None,
    team_factors: Optional[Mapping[str, TeamFactors]] = None,
    depth_weight: float = 0.0,
    depth_multiplier: float = 0.0,
    elo_shrink: float = 1.0,
    match_rating_sd: float = 0.0,
    match_shock_dist: str = "normal",
    match_shock_df: float = 5.0,
    goal_overdispersion: float = 0.0,
    upset_prob: float = 0.0,
    upset_underdog_bonus: float = 0.0,
    upset_shock_sd: float = 0.0,
    upset_min_abs_delta: float = 50.0,
    upset_potential_weight: float = 0.0,
    upset_resilience_weight: float = 0.0,
    dynamic_elos: Optional[Mapping[str, float]] = None,
    match_context: Optional[MatchContext] = None,
    contextual_effects: bool = False,
    host_scope: str = "tournament",
    heat_weight: float = 0.0,
    humidity_weight: float = 0.0,
    altitude_weight: float = 0.0,
    travel_weight: float = 0.0,
) -> Tuple[int, int, float, float, float, float, bool, float, float, float, float, float, float, float, float, float, float]:
    (
        delta, base_delta, rating_delta, factor_delta, depth_delta,
        context_delta, heat_delta, humidity_delta, altitude_delta, travel_delta, host_delta
    ) = compute_effective_delta(
        team1,
        team2,
        host_advantage,
        elo_shrink,
        rating_adjustments,
        static_factor_adjustments,
        team_factors,
        depth_weight,
        depth_multiplier,
        dynamic_elos=dynamic_elos,
        match_context=match_context,
        contextual_effects=contextual_effects,
        host_scope=host_scope,
        heat_weight=heat_weight,
        humidity_weight=humidity_weight,
        altitude_weight=altitude_weight,
        travel_weight=travel_weight,
    )
    match_shock = sample_match_shock(rng, match_rating_sd, match_shock_dist, match_shock_df)
    delta += match_shock
    pre_upset_delta = delta
    delta, upset_applied, upset_adjustment = apply_explicit_upset_shock(
        delta,
        team1,
        team2,
        rng,
        upset_prob,
        upset_underdog_bonus,
        upset_shock_sd,
        upset_min_abs_delta,
        team_factors,
        upset_potential_weight,
        upset_resilience_weight,
    )
    mu1, mu2 = expected_goals_from_delta(delta, base_mu, goal_elo_scale)
    mu1 = gamma_perturb_mean(mu1, rng, goal_overdispersion)
    mu2 = gamma_perturb_mean(mu2, rng, goal_overdispersion)
    return (
        poisson_sample(rng, mu1), poisson_sample(rng, mu2), delta, mu1, mu2,
        pre_upset_delta, upset_applied, upset_adjustment, factor_delta, depth_delta,
        match_shock, context_delta, heat_delta, humidity_delta, altitude_delta,
        travel_delta, host_delta,
    )

def settle_knockout(
    g1: int,
    g2: int,
    delta: float,
    rng: random.Random,
    ko_elo_scale: float,
    penalty_randomness: float = 0.0,
) -> int:
    """Return 1 if team1 advances, 2 if team2 advances.

    If the 90-minute score is level, the Elo-logistic tie-break probability is
    blended with 0.5.  penalty_randomness=0 gives the pure Elo tie-breaker;
    penalty_randomness=1 makes drawn knockout matches pure coin flips.
    """
    if g1 > g2:
        return 1
    if g2 > g1:
        return 2
    p_elo = elo_logistic(delta, ko_elo_scale)
    p_advances = (1.0 - penalty_randomness) * p_elo + penalty_randomness * 0.5
    return 1 if rng.random() < p_advances else 2


def update_record(table: Dict[str, Record], t1: str, t2: str, g1: int, g2: int) -> None:
    r1, r2 = table[t1], table[t2]
    r1.gf += g1
    r1.ga += g2
    r2.gf += g2
    r2.ga += g1
    if g1 > g2:
        r1.pts += 3
        r1.wins += 1
        r2.losses += 1
    elif g2 > g1:
        r2.pts += 3
        r2.wins += 1
        r1.losses += 1
    else:
        r1.pts += 1
        r2.pts += 1
        r1.draws += 1
        r2.draws += 1


# ---------------------------------------------------------------------------
# FIFA-style ranking, with explicit approximation for unsimulated fair play/FIFA ranking
# ---------------------------------------------------------------------------

def partition_desc(items: Sequence[str], key: Callable[[str], object]) -> List[List[str]]:
    buckets: DefaultDict[object, List[str]] = defaultdict(list)
    for item in items:
        buckets[key(item)].append(item)
    return [buckets[k] for k in sorted(buckets.keys(), reverse=True)]


def mini_table_metrics(names: Sequence[str], matches: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, int]]:
    name_set = set(names)
    metrics = {name: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for name in names}
    for m in matches:
        t1 = str(m["team1"])
        t2 = str(m["team2"])
        if t1 not in name_set or t2 not in name_set:
            continue
        g1 = int(m["goals1"])
        g2 = int(m["goals2"])
        metrics[t1]["gf"] += g1
        metrics[t1]["ga"] += g2
        metrics[t2]["gf"] += g2
        metrics[t2]["ga"] += g1
        if g1 > g2:
            metrics[t1]["pts"] += 3
        elif g2 > g1:
            metrics[t2]["pts"] += 3
        else:
            metrics[t1]["pts"] += 1
            metrics[t2]["pts"] += 1
    for name in names:
        metrics[name]["gd"] = metrics[name]["gf"] - metrics[name]["ga"]
    return metrics


def fallback_value(
    team_name: str,
    teams: Mapping[str, Team],
    fallback_ranking: str,
    rating_overrides: Optional[Mapping[str, float]] = None,
) -> float:
    team = teams[team_name]
    if fallback_ranking == "fifa_rank" and team.fifa_rank is not None:
        # Lower FIFA rank is better; convert to higher-is-better for partition_desc.
        return -float(team.fifa_rank)
    if fallback_ranking in {"fifa_rank", "elo"}:
        return float(rating_overrides.get(team_name, team.elo)) if rating_overrides is not None else float(team.elo)
    if fallback_ranking == "name":
        # Final deterministic fallback.  Reversed lexicographic order is arbitrary;
        # it is used only after football criteria and configured ranking fail.
        return 0.0
    raise ValueError(f"Unknown fallback ranking: {fallback_ranking}")


def break_tie_by_overall(
    names: Sequence[str],
    rec_by_team: Mapping[str, Record],
    teams: Mapping[str, Team],
    fallback_ranking: str,
    rating_overrides: Optional[Mapping[str, float]] = None,
) -> List[str]:
    blocks: List[List[str]] = [list(names)]
    criteria: List[Tuple[str, Callable[[str], object]]] = [
        ("overall_gd", lambda n: rec_by_team[n].gd),
        ("overall_gf", lambda n: rec_by_team[n].gf),
        ("conduct_score", lambda n: rec_by_team[n].conduct_score),
        ("fallback", lambda n: fallback_value(n, teams, fallback_ranking, rating_overrides)),
    ]
    for _, key in criteria:
        new_blocks: List[List[str]] = []
        for block in blocks:
            if len(block) == 1:
                new_blocks.append(block)
            else:
                new_blocks.extend(partition_desc(block, key))
        blocks = new_blocks
    # Fully deterministic final fallback.
    out: List[str] = []
    for block in blocks:
        out.extend(sorted(block, reverse=True))
    return out


def break_equal_points_tie(
    names: Sequence[str],
    rec_by_team: Mapping[str, Record],
    matches: Sequence[Mapping[str, object]],
    teams: Mapping[str, Team],
    fallback_ranking: str,
    rating_overrides: Optional[Mapping[str, float]] = None,
) -> List[str]:
    """Rank teams tied on points.

    FIFA Regulations Article 13 first applies head-to-head criteria among teams
    equal on points.  If teams remain equal, overall goal difference, goals scored,
    team conduct and FIFA ranking are applied.  Team conduct is not simulated here;
    FIFA ranking can be supplied as an optional teams.csv column.  Otherwise Elo is
    used as the deterministic final sporting fallback.
    """
    blocks: List[List[str]] = [list(names)]
    h2h_criteria = ("pts", "gd", "gf")
    for crit in h2h_criteria:
        new_blocks: List[List[str]] = []
        for block in blocks:
            if len(block) == 1:
                new_blocks.append(block)
                continue
            # Recompute mini-table on the remaining tied teams only.
            metrics = mini_table_metrics(block, matches)
            new_blocks.extend(partition_desc(block, lambda n, c=crit, mm=metrics: mm[n][c]))
        blocks = new_blocks
    out: List[str] = []
    for block in blocks:
        if len(block) == 1:
            out.extend(block)
        else:
            out.extend(break_tie_by_overall(block, rec_by_team, teams, fallback_ranking, rating_overrides))
    return out


def rank_group_records(
    records: Iterable[Record],
    group_matches: Sequence[Mapping[str, object]],
    teams: Mapping[str, Team],
    fallback_ranking: str,
    rating_overrides: Optional[Mapping[str, float]] = None,
) -> List[Record]:
    rec_by_team = {r.team: r for r in records}
    names = list(rec_by_team.keys())
    blocks = partition_desc(names, lambda n: rec_by_team[n].pts)
    ranked_names: List[str] = []
    for block in blocks:
        if len(block) == 1:
            ranked_names.extend(block)
        else:
            ranked_names.extend(break_equal_points_tie(block, rec_by_team, group_matches, teams, fallback_ranking, rating_overrides))
    return [rec_by_team[n] for n in ranked_names]


def rank_third_place_records(
    records: Iterable[Record],
    teams: Mapping[str, Team],
    fallback_ranking: str,
    rating_overrides: Optional[Mapping[str, float]] = None,
) -> List[Record]:
    def key(r: Record) -> Tuple[object, ...]:
        return (
            r.pts,
            r.gd,
            r.gf,
            r.conduct_score,
            fallback_value(r.team, teams, fallback_ranking, rating_overrides),
            r.team,
        )

    return sorted(records, key=key, reverse=True)


# ---------------------------------------------------------------------------
# Annexe C lookup
# ---------------------------------------------------------------------------

def load_annex_c_assignments(
    annex_c_path: Path,
    r32_rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[Tuple[str, ...], Dict[int, str]], Dict[Tuple[str, ...], int]]:
    rows = read_csv(annex_c_path)

    allowed_by_match: Dict[int, set[str]] = {}
    for r in r32_rows:
        if r["side2"] == "3rd":
            match_no = int(r["match_no"])
            allowed_by_match[match_no] = set(x for x in r["side2_allowed_third_groups"].split("/") if x)

    expected_matches = set(ANNEX_C_SLOT_TO_MATCH.values())
    if set(allowed_by_match) != expected_matches:
        raise ValueError(
            "Round-of-32 third-place slots do not match Annexe C columns: "
            f"got {sorted(allowed_by_match)}, expected {sorted(expected_matches)}"
        )

    assignments: Dict[Tuple[str, ...], Dict[int, str]] = {}
    option_by_key: Dict[Tuple[str, ...], int] = {}
    options_seen: set[int] = set()
    for r in rows:
        option = int(r["option"])
        if option in options_seen:
            raise ValueError(f"Duplicate Annexe C option: {option}")
        options_seen.add(option)

        third_groups = tuple(sorted(g.strip() for g in r["third_groups"].split("/") if g.strip()))
        if len(third_groups) != 8:
            raise ValueError(f"Annexe C option {option} does not list eight groups: {third_groups}")

        values: List[str] = []
        assignment: Dict[int, str] = {}
        for col in ANNEX_C_COLUMNS:
            cell = r[col].strip().upper()
            if len(cell) != 2 or not cell.startswith("3") or cell[1] not in GROUPS:
                raise ValueError(f"Invalid Annexe C cell at option {option}, {col}: {cell!r}")
            group = cell[1]
            values.append(group)
            match_no = ANNEX_C_SLOT_TO_MATCH[col]
            if group not in allowed_by_match[match_no]:
                raise ValueError(
                    f"Annexe C option {option}, {col} assigns group {group} "
                    f"to match {match_no}, but allowed groups are {sorted(allowed_by_match[match_no])}"
                )
            assignment[match_no] = group

        if tuple(sorted(values)) != third_groups:
            raise ValueError(
                f"Annexe C option {option} has inconsistent third_groups={third_groups} "
                f"and slot values={values}"
            )
        if len(set(values)) != 8:
            raise ValueError(f"Annexe C option {option} repeats a third-place group: {values}")
        if third_groups in assignments:
            raise ValueError(f"Duplicate Annexe C third-place combination: {third_groups}")
        assignments[third_groups] = assignment
        option_by_key[third_groups] = option

    if options_seen != set(range(1, 496)):
        missing_options = sorted(set(range(1, 496)) - options_seen)
        extra_options = sorted(options_seen - set(range(1, 496)))
        raise ValueError(
            "Annexe C options must be exactly 1..495; "
            f"missing={missing_options[:5]}, extra={extra_options[:5]}"
        )

    expected_sets = {tuple(c) for c in __import__("itertools").combinations(GROUPS, 8)}
    if set(assignments) != expected_sets:
        missing = sorted(expected_sets - set(assignments))
        extra = sorted(set(assignments) - expected_sets)
        raise ValueError(f"Annexe C must contain all 495 combinations; missing={missing[:5]}, extra={extra[:5]}")

    return assignments, option_by_key


def assign_thirds_to_slots(
    advanced_third_groups: Sequence[str],
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]],
) -> Dict[int, str]:
    key = tuple(sorted(advanced_third_groups))
    try:
        return dict(annex_c_assignments[key])
    except KeyError as exc:
        raise RuntimeError(f"No FIFA Annexe C assignment for third-place groups {key}") from exc


def resolve_slot(slot: str, qualifiers: Mapping[str, str], third_assignment: Mapping[int, str], match_no: int) -> str:
    if slot == "3rd":
        group = third_assignment[match_no]
        return qualifiers[f"{group}3q"]
    return qualifiers[slot]


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def simulate_group_stage(
    teams: Mapping[str, Team],
    fixtures: Sequence[Mapping[str, str]],
    rng: random.Random,
    base_mu: float,
    goal_elo_scale: float,
    host_advantage: float,
    fallback_ranking: str,
    rating_adjustments: Optional[Mapping[str, float]] = None,
    static_factor_adjustments: Optional[Mapping[str, float]] = None,
    team_factors: Optional[Mapping[str, TeamFactors]] = None,
    depth_weight: float = 0.0,
    elo_shrink: float = 1.0,
    match_rating_sd: float = 0.0,
    match_shock_dist: str = "normal",
    match_shock_df: float = 5.0,
    goal_overdispersion: float = 0.0,
    upset_prob: float = 0.0,
    upset_underdog_bonus: float = 0.0,
    upset_shock_sd: float = 0.0,
    upset_min_abs_delta: float = 50.0,
    upset_potential_weight: float = 0.0,
    upset_resilience_weight: float = 0.0,
    dynamic_elo: bool = False,
    dynamic_elo_k: float = 40.0,
    dynamic_elo_scale: float = 400.0,
    dynamic_elo_margin: bool = True,
    dynamic_elo_round_delta: bool = False,
    dynamic_elo_home_advantage: Optional[float] = None,
    dynamic_elo_initial: Optional[Mapping[str, float]] = None,
    dynamic_elo_sink: Optional[MutableMapping[str, float]] = None,
    match_contexts: Optional[Mapping[str, MatchContext]] = None,
    contextual_effects: bool = False,
    host_scope: str = "tournament",
    heat_weight: float = 0.0,
    humidity_weight: float = 0.0,
    altitude_weight: float = 0.0,
    travel_weight: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, List[str]], List[Record], List[dict], List[str]]:
    group_tables: Dict[str, Dict[str, Record]] = {}
    for t in teams.values():
        group_tables.setdefault(t.group, {})[t.name] = Record(team=t.name)

    match_log: List[dict] = []
    matches_by_group: DefaultDict[str, List[dict]] = defaultdict(list)
    dynamic_ratings: Dict[str, float] = {
        name: float(dynamic_elo_initial[name]) if dynamic_elo_initial and name in dynamic_elo_initial else float(team.elo)
        for name, team in teams.items()
    }
    elo_home_adv = host_advantage if dynamic_elo_home_advantage is None else dynamic_elo_home_advantage

    for m in fixtures:
        context_key = str(m.get("match_id") or m.get("match_no") or "")
        mcontext = match_context_for_key(match_contexts, context_key)
        base_t1, base_t2 = teams[m["team1"]], teams[m["team2"]]
        t1 = team_with_elo(base_t1, dynamic_ratings[base_t1.name]) if dynamic_elo else base_t1
        t2 = team_with_elo(base_t2, dynamic_ratings[base_t2.name]) if dynamic_elo else base_t2
        (
            g1, g2, delta, mu1, mu2, pre_upset_delta, upset_applied,
            upset_adjustment, factor_delta, depth_delta, match_shock,
            context_delta, heat_delta, humidity_delta, altitude_delta, travel_delta, host_delta
        ) = simulate_score(
            t1,
            t2,
            rng,
            base_mu,
            goal_elo_scale,
            host_advantage,
            rating_adjustments=rating_adjustments,
            static_factor_adjustments=static_factor_adjustments,
            team_factors=team_factors,
            depth_weight=depth_weight,
            depth_multiplier=ROUND_DEPTH_MULTIPLIER["Group"],
            elo_shrink=elo_shrink,
            match_rating_sd=match_rating_sd,
            match_shock_dist=match_shock_dist,
            match_shock_df=match_shock_df,
            goal_overdispersion=goal_overdispersion,
            upset_prob=upset_prob,
            upset_underdog_bonus=upset_underdog_bonus,
            upset_shock_sd=upset_shock_sd,
            upset_min_abs_delta=upset_min_abs_delta,
            upset_potential_weight=upset_potential_weight,
            upset_resilience_weight=upset_resilience_weight,
            match_context=mcontext,
            contextual_effects=contextual_effects,
            host_scope=host_scope,
            heat_weight=heat_weight,
            humidity_weight=humidity_weight,
            altitude_weight=altitude_weight,
            travel_weight=travel_weight,
        )
        update_record(group_tables[m["group"]], t1.name, t2.name, g1, g2)
        outcome = "draw" if g1 == g2 else ("team1" if g1 > g2 else "team2")
        row = {
            "match_no": m["match_id"],
            "round": "Group",
            "group": m["group"],
            "team1": t1.name,
            "team2": t2.name,
            "goals1": g1,
            "goals2": g2,
            "winner": "" if outcome == "draw" else (t1.name if outcome == "team1" else t2.name),
            "loser": "" if outcome == "draw" else (t2.name if outcome == "team1" else t1.name),
            "outcome": outcome,
            "elo_delta": round(delta, 3),
            "xg1_model_mean": round(mu1, 4),
            "xg2_model_mean": round(mu2, 4),
            "factor_delta": round(factor_delta, 3),
            "depth_delta": round(depth_delta, 3),
            "match_shock": round(match_shock, 3),
            "venue": mcontext.venue,
            "city": mcontext.city,
            "country": mcontext.country,
            "host_delta": round(host_delta, 3),
            "context_delta": round(context_delta, 3),
            "heat_delta": round(heat_delta, 3),
            "humidity_delta": round(humidity_delta, 3),
            "altitude_delta": round(altitude_delta, 3),
            "travel_delta": round(travel_delta, 3),
        }
        row.update(upset_diagnostics(t1, t2, str(row["winner"]), pre_upset_delta, upset_applied, upset_adjustment))
        if dynamic_elo:
            elo_audit = update_dynamic_elos(
                dynamic_ratings,
                t1,
                t2,
                g1,
                g2,
                k_factor=dynamic_elo_k,
                elo_scale=dynamic_elo_scale,
                host_advantage=elo_home_adv,
                use_goal_multiplier=dynamic_elo_margin,
                round_delta=dynamic_elo_round_delta,
                host_delta_override=host_delta if host_scope != "tournament" else None,
                context_delta=context_delta if contextual_effects else 0.0,
            )
            row.update({k: (round(v, 6) if isinstance(v, float) and not math.isnan(v) else v) for k, v in elo_audit.items()})
        match_log.append(row)
        matches_by_group[m["group"]].append(row)

    qualifiers: Dict[str, str] = {}
    group_rankings: Dict[str, List[str]] = {}
    all_ranked_records: List[Record] = []
    third_records: List[Record] = []

    ranking_teams = teams_with_elos(teams, dynamic_ratings) if dynamic_elo else teams
    for g in GROUPS:
        table = group_tables[g]
        ranked = rank_group_records(table.values(), matches_by_group[g], ranking_teams, fallback_ranking)
        group_rankings[g] = [r.team for r in ranked]
        qualifiers[f"{g}1"] = ranked[0].team
        qualifiers[f"{g}2"] = ranked[1].team
        qualifiers[f"{g}3"] = ranked[2].team
        third_records.append(ranked[2])
        all_ranked_records.extend(ranked)

    best_thirds = rank_third_place_records(third_records, ranking_teams, fallback_ranking)[:8]
    advanced_third_groups = sorted(teams[r.team].group for r in best_thirds)
    for r in best_thirds:
        qualifiers[f"{teams[r.team].group}3q"] = r.team

    if dynamic_elo and dynamic_elo_sink is not None:
        dynamic_elo_sink.clear()
        dynamic_elo_sink.update(dynamic_ratings)

    return qualifiers, group_rankings, all_ranked_records, match_log, advanced_third_groups


def simulate_knockout(
    teams: Mapping[str, Team],
    qualifiers: Mapping[str, str],
    r32_rows: Sequence[Mapping[str, str]],
    bracket_rows: Sequence[Mapping[str, str]],
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]],
    rng: random.Random,
    base_mu: float,
    goal_elo_scale: float,
    ko_elo_scale: float,
    host_advantage: float,
    rating_adjustments: Optional[Mapping[str, float]] = None,
    static_factor_adjustments: Optional[Mapping[str, float]] = None,
    team_factors: Optional[Mapping[str, TeamFactors]] = None,
    depth_weight: float = 0.0,
    elo_shrink: float = 1.0,
    match_rating_sd: float = 0.0,
    match_shock_dist: str = "normal",
    match_shock_df: float = 5.0,
    goal_overdispersion: float = 0.0,
    penalty_randomness: float = 0.0,
    upset_prob: float = 0.0,
    upset_underdog_bonus: float = 0.0,
    upset_shock_sd: float = 0.0,
    upset_min_abs_delta: float = 50.0,
    upset_potential_weight: float = 0.0,
    upset_resilience_weight: float = 0.0,
    dynamic_elo: bool = False,
    dynamic_elo_k: float = 60.0,
    dynamic_elo_k_by_round: Optional[Mapping[str, float]] = None,
    dynamic_elo_scale: float = 400.0,
    dynamic_elo_margin: bool = True,
    dynamic_elo_round_delta: bool = False,
    dynamic_elo_home_advantage: Optional[float] = None,
    dynamic_elo_knockout_draw_value: float = 0.5,
    dynamic_elo_initial: Optional[Mapping[str, float]] = None,
    match_contexts: Optional[Mapping[str, MatchContext]] = None,
    contextual_effects: bool = False,
    host_scope: str = "tournament",
    heat_weight: float = 0.0,
    humidity_weight: float = 0.0,
    altitude_weight: float = 0.0,
    travel_weight: float = 0.0,
) -> Tuple[Dict[int, str], Dict[int, str], List[dict], Dict[int, str]]:
    winners: Dict[int, str] = {}
    losers: Dict[int, str] = {}
    match_log: List[dict] = []
    dynamic_ratings: Dict[str, float] = {
        name: float(dynamic_elo_initial[name]) if dynamic_elo_initial and name in dynamic_elo_initial else float(team.elo)
        for name, team in teams.items()
    }
    elo_home_adv = host_advantage if dynamic_elo_home_advantage is None else dynamic_elo_home_advantage

    advanced_third_groups = sorted(k[0] for k in qualifiers if k.endswith("3q"))
    third_assignment = assign_thirds_to_slots(advanced_third_groups, annex_c_assignments)

    def play(match_no: int, round_name: str, name1: str, name2: str, source1: str, source2: str) -> None:
        mcontext = match_context_for_key(match_contexts, match_no)
        base_t1, base_t2 = teams[name1], teams[name2]
        t1 = team_with_elo(base_t1, dynamic_ratings[base_t1.name]) if dynamic_elo else base_t1
        t2 = team_with_elo(base_t2, dynamic_ratings[base_t2.name]) if dynamic_elo else base_t2
        (
            g1, g2, delta, mu1, mu2, pre_upset_delta, upset_applied,
            upset_adjustment, factor_delta, depth_delta, match_shock,
            context_delta, heat_delta, humidity_delta, altitude_delta, travel_delta, host_delta
        ) = simulate_score(
            t1,
            t2,
            rng,
            base_mu,
            goal_elo_scale,
            host_advantage,
            rating_adjustments=rating_adjustments,
            static_factor_adjustments=static_factor_adjustments,
            team_factors=team_factors,
            depth_weight=depth_weight,
            depth_multiplier=ROUND_DEPTH_MULTIPLIER.get(round_name, 0.5),
            elo_shrink=elo_shrink,
            match_rating_sd=match_rating_sd,
            match_shock_dist=match_shock_dist,
            match_shock_df=match_shock_df,
            goal_overdispersion=goal_overdispersion,
            upset_prob=upset_prob,
            upset_underdog_bonus=upset_underdog_bonus,
            upset_shock_sd=upset_shock_sd,
            upset_min_abs_delta=upset_min_abs_delta,
            upset_potential_weight=upset_potential_weight,
            upset_resilience_weight=upset_resilience_weight,
            match_context=mcontext,
            contextual_effects=contextual_effects,
            host_scope=host_scope,
            heat_weight=heat_weight,
            humidity_weight=humidity_weight,
            altitude_weight=altitude_weight,
            travel_weight=travel_weight,
        )
        adv = settle_knockout(g1, g2, delta, rng, ko_elo_scale, penalty_randomness)
        win, lose = (t1.name, t2.name) if adv == 1 else (t2.name, t1.name)
        winners[match_no] = win
        losers[match_no] = lose
        row = {
            "match_no": match_no,
            "round": round_name,
            "group": "",
            "source1": source1,
            "source2": source2,
            "team1": t1.name,
            "team2": t2.name,
            "goals1": g1,
            "goals2": g2,
            "winner": win,
            "loser": lose,
            "outcome": "team1" if adv == 1 else "team2",
            "decided_by": "90min" if g1 != g2 else "extra_time_or_penalties_proxy",
            "elo_delta": round(delta, 3),
            "xg1_model_mean": round(mu1, 4),
            "xg2_model_mean": round(mu2, 4),
            "factor_delta": round(factor_delta, 3),
            "depth_delta": round(depth_delta, 3),
            "match_shock": round(match_shock, 3),
            "venue": mcontext.venue,
            "city": mcontext.city,
            "country": mcontext.country,
            "host_delta": round(host_delta, 3),
            "context_delta": round(context_delta, 3),
            "heat_delta": round(heat_delta, 3),
            "humidity_delta": round(humidity_delta, 3),
            "altitude_delta": round(altitude_delta, 3),
            "travel_delta": round(travel_delta, 3),
        }
        row.update(upset_diagnostics(t1, t2, win, pre_upset_delta, upset_applied, upset_adjustment))
        if dynamic_elo:
            k = dynamic_elo_k
            if dynamic_elo_k_by_round is not None:
                k = float(dynamic_elo_k_by_round.get(round_name, dynamic_elo_k))
            elo_audit = update_dynamic_elos(
                dynamic_ratings,
                t1,
                t2,
                g1,
                g2,
                k_factor=k,
                elo_scale=dynamic_elo_scale,
                host_advantage=elo_home_adv,
                use_goal_multiplier=dynamic_elo_margin,
                advancer=adv,
                knockout_draw_value=dynamic_elo_knockout_draw_value,
                round_delta=dynamic_elo_round_delta,
                host_delta_override=host_delta if host_scope != "tournament" else None,
                context_delta=context_delta if contextual_effects else 0.0,
            )
            row.update({k: (round(v, 6) if isinstance(v, float) and not math.isnan(v) else v) for k, v in elo_audit.items()})
        match_log.append(row)

    for r in r32_rows:
        mn = int(r["match_no"])
        team1 = resolve_slot(r["side1"], qualifiers, third_assignment, mn)
        team2 = resolve_slot(r["side2"], qualifiers, third_assignment, mn)
        source2 = f"3{third_assignment[mn]}" if r["side2"] == "3rd" else r["side2"]
        play(mn, "R32", team1, team2, r["side1"], source2)

    for r in bracket_rows:
        mn = int(r["match_no"])

        def deref(x: str) -> str:
            if x.startswith("W"):
                return winners[int(x[1:])]
            if x.startswith("L"):
                return losers[int(x[1:])]
            raise ValueError(f"Unexpected bracket reference: {x}")

        play(mn, r["round"], deref(r["side1"]), deref(r["side2"]), r["side1"], r["side2"])

    return winners, losers, match_log, third_assignment


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def most_common(counter: Counter) -> Tuple[object, int]:
    if not counter:
        return "", 0
    return counter.most_common(1)[0]


def collect_stage_counts(stage_counts: Dict[str, Counter], winners: Mapping[int, str], losers: Mapping[int, str]) -> None:
    for mn in range(73, 89):
        stage_counts["reach_r16"][winners[mn]] += 1
    for mn in range(89, 97):
        stage_counts["reach_qf"][winners[mn]] += 1
    for mn in range(97, 101):
        stage_counts["reach_sf"][winners[mn]] += 1
    for mn in (101, 102):
        stage_counts["reach_final"][winners[mn]] += 1
    stage_counts["third_place"][winners[103]] += 1
    stage_counts["champion"][winners[104]] += 1
    stage_counts["runner_up"][losers[104]] += 1


def scenario_key(group_rankings: Mapping[str, Sequence[str]], knockout_log: Sequence[Mapping[str, object]]) -> str:
    parts = [f"G{g}:{'>'.join(group_rankings[g])}" for g in GROUPS]
    for m in sorted(knockout_log, key=lambda x: int(x["match_no"])):
        parts.append(f"M{m['match_no']}:{m['team1']} v {m['team2']} -> {m['winner']}")
    return "|".join(parts)


def knockout_dynamic_elo_k_by_round_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "R32": args.dynamic_elo_r32_k,
        "R16": args.dynamic_elo_r16_k,
        "QF": args.dynamic_elo_qf_k,
        "SF": args.dynamic_elo_sf_k,
        "Final": args.dynamic_elo_final_k,
        "Third place": args.dynamic_elo_third_place_k,
    }


def make_scenario_repr(
    group_rankings: Mapping[str, Sequence[str]],
    advanced_third_groups: Sequence[str],
    knockout_log: Sequence[Mapping[str, object]],
    annex_option: int,
) -> dict:
    return {
        "group_rankings": {g: list(group_rankings[g]) for g in GROUPS},
        "advanced_third_groups": list(advanced_third_groups),
        "annex_c_option": annex_option,
        "knockout_matches": [dict(m) for m in sorted(knockout_log, key=lambda x: int(x["match_no"]))],
        "champion": next(str(m["winner"]) for m in knockout_log if int(m["match_no"]) == 104),
    }


def run_simulation(args: argparse.Namespace) -> dict:
    teams_path = resolve_data_path(args.data_dir, args.teams_file)
    teams = load_teams(teams_path)
    team_factors = load_team_factors(args.data_dir / args.team_factors_file, teams)
    match_contexts = load_match_contexts(args.data_dir / args.match_context_file) if getattr(args, "contextual_effects", False) else {}
    static_factor_adjustments = compute_static_factor_adjustments(teams, team_factors, args)
    group_fixtures = read_csv(args.data_dir / "group_stage.csv")
    r32_rows = read_csv(args.data_dir / "round32_slots.csv")
    bracket_rows = read_csv(args.data_dir / "knockout_bracket.csv")
    annex_c_assignments, annex_option_by_key = load_annex_c_assignments(
        args.data_dir / "annex_c_third_place_assignments.csv", r32_rows
    )

    rng = random.Random(args.seed)
    z = normal_z(args.ci_level)

    stage_counts: Dict[str, Counter] = {stage: Counter() for stage in STAGES}
    group_pos_counts: DefaultDict[Tuple[str, int], Counter] = defaultdict(Counter)
    group_order_counts: Dict[str, Counter] = {g: Counter() for g in GROUPS}

    group_outcome_counts: DefaultDict[Tuple[int, str, str, str], Counter] = defaultdict(Counter)
    group_score_counts: DefaultDict[Tuple[int, str, str, str], Counter] = defaultdict(Counter)
    group_goal_sums: DefaultDict[Tuple[int, str, str, str], List[int]] = defaultdict(lambda: [0, 0, 0])

    third_group_counts: Counter = Counter()
    third_combination_counts: Counter = Counter()
    annex_option_counts: Counter = Counter()

    ko_side_counts: DefaultDict[Tuple[int, str], Counter] = defaultdict(Counter)
    ko_pair_counts: DefaultDict[int, Counter] = defaultdict(Counter)
    ko_pair_winner_counts: DefaultDict[Tuple[int, Tuple[str, str]], Counter] = defaultdict(Counter)
    ko_winner_counts: DefaultDict[int, Counter] = defaultdict(Counter)

    scenario_counts: Counter = Counter()
    scenario_examples: Dict[str, dict] = {}
    full_match_log: List[dict] = []
    upset_metric_counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)

    progress_every = int(getattr(args, "progress_every", 0) or 0)
    start_time = time.time()
    if progress_every > 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for sim_id in range(1, args.n_sim + 1):
        rating_adjustments = sample_tournament_rating_adjustments(teams, rng, args.team_rating_sd)
        dynamic_elo_state: Dict[str, float] = {}

        qualifiers, group_rankings, group_records, g_log, advanced_third_groups = simulate_group_stage(
            teams=teams,
            fixtures=group_fixtures,
            rng=rng,
            base_mu=args.base_mu,
            goal_elo_scale=args.goal_elo_scale,
            host_advantage=args.host_advantage,
            fallback_ranking=args.fallback_ranking,
            rating_adjustments=rating_adjustments,
            static_factor_adjustments=static_factor_adjustments,
            team_factors=team_factors,
            depth_weight=args.depth_weight,
            elo_shrink=args.elo_shrink,
            match_rating_sd=args.match_rating_sd,
            match_shock_dist=args.match_shock_dist,
            match_shock_df=args.match_shock_df,
            goal_overdispersion=args.goal_overdispersion,
            upset_prob=args.upset_prob,
            upset_underdog_bonus=args.upset_underdog_bonus,
            upset_shock_sd=args.upset_shock_sd,
            upset_min_abs_delta=args.upset_min_abs_delta,
            upset_potential_weight=args.upset_potential_weight,
            upset_resilience_weight=args.upset_resilience_weight,
            dynamic_elo=args.dynamic_elo,
            dynamic_elo_k=args.dynamic_elo_group_k,
            dynamic_elo_scale=args.dynamic_elo_scale,
            dynamic_elo_margin=not args.dynamic_elo_no_margin_multiplier,
            dynamic_elo_round_delta=args.dynamic_elo_round_delta,
            dynamic_elo_home_advantage=args.dynamic_elo_home_advantage,
            dynamic_elo_sink=dynamic_elo_state,
            match_contexts=match_contexts,
            contextual_effects=args.contextual_effects,
            host_scope=args.host_scope,
            heat_weight=args.heat_weight,
            humidity_weight=args.humidity_weight,
            altitude_weight=args.altitude_weight,
            travel_weight=args.travel_weight,
        )

        # Group position distributions and exact order modes.
        for g in GROUPS:
            order = tuple(group_rankings[g])
            group_order_counts[g][order] += 1
            for pos, team_name in enumerate(order, start=1):
                group_pos_counts[(g, pos)][team_name] += 1

        # R32 participants.
        r32_participants = set()
        for g in GROUPS:
            r32_participants.add(qualifiers[f"{g}1"])
            r32_participants.add(qualifiers[f"{g}2"])
        for g in advanced_third_groups:
            r32_participants.add(qualifiers[f"{g}3q"])
            third_group_counts[g] += 1
        for t in r32_participants:
            stage_counts["reach_r32"][t] += 1

        # Group match aggregation.
        for m in g_log:
            accumulate_upset_metrics(upset_metric_counts, m)
            key = (str(m["match_no"]), str(m["group"]), str(m["team1"]), str(m["team2"]))
            group_outcome_counts[key][str(m["outcome"])] += 1
            group_score_counts[key][(int(m["goals1"]), int(m["goals2"]))] += 1
            group_goal_sums[key][0] += int(m["goals1"])
            group_goal_sums[key][1] += int(m["goals2"])
            group_goal_sums[key][2] += 1

        third_key = tuple(sorted(advanced_third_groups))
        third_combination_counts[third_key] += 1
        annex_option = annex_option_by_key[third_key]
        annex_option_counts[annex_option] += 1

        winners, losers, k_log, _third_assignment = simulate_knockout(
            teams=teams,
            qualifiers=qualifiers,
            r32_rows=r32_rows,
            bracket_rows=bracket_rows,
            annex_c_assignments=annex_c_assignments,
            rng=rng,
            base_mu=args.base_mu,
            goal_elo_scale=args.goal_elo_scale,
            ko_elo_scale=args.ko_elo_scale,
            host_advantage=args.host_advantage,
            rating_adjustments=rating_adjustments,
            static_factor_adjustments=static_factor_adjustments,
            team_factors=team_factors,
            depth_weight=args.depth_weight,
            elo_shrink=args.elo_shrink,
            match_rating_sd=args.match_rating_sd,
            match_shock_dist=args.match_shock_dist,
            match_shock_df=args.match_shock_df,
            goal_overdispersion=args.goal_overdispersion,
            penalty_randomness=args.penalty_randomness,
            upset_prob=args.upset_prob,
            upset_underdog_bonus=args.upset_underdog_bonus,
            upset_shock_sd=args.upset_shock_sd,
            upset_min_abs_delta=args.upset_min_abs_delta,
            upset_potential_weight=args.upset_potential_weight,
            upset_resilience_weight=args.upset_resilience_weight,
            dynamic_elo=args.dynamic_elo,
            dynamic_elo_k=args.dynamic_elo_final_k,
            dynamic_elo_k_by_round=knockout_dynamic_elo_k_by_round_from_args(args),
            dynamic_elo_scale=args.dynamic_elo_scale,
            dynamic_elo_margin=not args.dynamic_elo_no_margin_multiplier,
            dynamic_elo_round_delta=args.dynamic_elo_round_delta,
            dynamic_elo_home_advantage=args.dynamic_elo_home_advantage,
            dynamic_elo_knockout_draw_value=args.dynamic_elo_knockout_draw_value,
            dynamic_elo_initial=dynamic_elo_state if args.dynamic_elo else None,
            match_contexts=match_contexts,
            contextual_effects=args.contextual_effects,
            host_scope=args.host_scope,
            heat_weight=args.heat_weight,
            humidity_weight=args.humidity_weight,
            altitude_weight=args.altitude_weight,
            travel_weight=args.travel_weight,
        )
        collect_stage_counts(stage_counts, winners, losers)

        # Knockout match aggregation.
        for m in k_log:
            accumulate_upset_metrics(upset_metric_counts, m)
            mn = int(m["match_no"])
            team1 = str(m["team1"])
            team2 = str(m["team2"])
            winner = str(m["winner"])
            pair = (team1, team2)
            ko_side_counts[(mn, "side1")][team1] += 1
            ko_side_counts[(mn, "side2")][team2] += 1
            ko_pair_counts[mn][pair] += 1
            ko_pair_winner_counts[(mn, pair)][winner] += 1
            ko_winner_counts[mn][winner] += 1

        if args.track_scenarios:
            skey = scenario_key(group_rankings, k_log)
            scenario_counts[skey] += 1
            if skey not in scenario_examples:
                scenario_examples[skey] = make_scenario_repr(group_rankings, advanced_third_groups, k_log, annex_option)

        if args.save_match_log:
            for m in g_log + k_log:
                row = {"sim_id": sim_id}
                row.update(m)
                full_match_log.append(row)

        if progress_every > 0 and (sim_id % progress_every == 0 or sim_id == args.n_sim):
            elapsed = time.time() - start_time
            rate = sim_id / elapsed if elapsed > 0 else float("nan")
            remaining = (args.n_sim - sim_id) / rate if rate and rate > 0 else float("nan")
            progress = {
                "completed": sim_id,
                "total": args.n_sim,
                "percent": round(100.0 * sim_id / args.n_sim, 2),
                "elapsed_sec": round(elapsed, 1),
                "sim_per_sec": round(rate, 3),
                "eta_sec": round(remaining, 1),
            }
            (args.out_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(
                f"progress: {sim_id}/{args.n_sim} "
                f"({100.0 * sim_id / args.n_sim:.1f}%), "
                f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s",
                flush=True,
            )

    return {
        "teams": teams,
        "group_fixtures": group_fixtures,
        "r32_rows": r32_rows,
        "bracket_rows": bracket_rows,
        "match_contexts": match_contexts,
        "z": z,
        "stage_counts": stage_counts,
        "group_pos_counts": group_pos_counts,
        "group_order_counts": group_order_counts,
        "group_outcome_counts": group_outcome_counts,
        "group_score_counts": group_score_counts,
        "group_goal_sums": group_goal_sums,
        "third_group_counts": third_group_counts,
        "third_combination_counts": third_combination_counts,
        "annex_option_counts": annex_option_counts,
        "ko_side_counts": ko_side_counts,
        "ko_pair_counts": ko_pair_counts,
        "ko_pair_winner_counts": ko_pair_winner_counts,
        "ko_winner_counts": ko_winner_counts,
        "scenario_counts": scenario_counts,
        "scenario_examples": scenario_examples,
        "team_factors": team_factors,
        "static_factor_adjustments": static_factor_adjustments,
        "upset_metric_counts": upset_metric_counts,
        "full_match_log": full_match_log,
    }


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def write_team_stage_probabilities(out_dir: Path, state: Mapping[str, object], n: int, z: float) -> List[dict]:
    teams: Mapping[str, Team] = state["teams"]  # type: ignore[assignment]
    stage_counts: Mapping[str, Counter] = state["stage_counts"]  # type: ignore[assignment]
    rows: List[dict] = []
    for name in sorted(teams.keys(), key=lambda x: (-stage_counts["champion"][x], -teams[x].elo, x)):
        row: Dict[str, object] = {
            "team": name,
            "group": teams[name].group,
            "position": teams[name].position,
            "elo": teams[name].elo,
        }
        for stage in STAGES:
            add_prob(row, stage, stage_counts[stage][name], n, z)
        rows.append(row)

    fieldnames = ["team", "group", "position", "elo"]
    for stage in STAGES:
        fieldnames.extend([stage, f"{stage}_ci_low", f"{stage}_ci_high", f"{stage}_count"])
    write_csv(out_dir / "team_stage_probabilities.csv", rows, fieldnames)
    return rows


def write_group_position_outputs(out_dir: Path, state: Mapping[str, object], n: int, z: float) -> None:
    group_pos_counts: Mapping[Tuple[str, int], Counter] = state["group_pos_counts"]  # type: ignore[assignment]
    group_order_counts: Mapping[str, Counter] = state["group_order_counts"]  # type: ignore[assignment]

    pos_rows: List[dict] = []
    for g in GROUPS:
        for pos in range(1, 5):
            c = group_pos_counts[(g, pos)]
            for team, k in c.most_common():
                row: Dict[str, object] = {"group": g, "position": pos, "team": team}
                add_prob(row, "probability", k, n, z)
                pos_rows.append(row)
    write_csv(out_dir / "group_position_probabilities.csv", pos_rows)

    exact_rows: List[dict] = []
    ml_rows: List[dict] = []
    for g in GROUPS:
        c = group_order_counts[g]
        for rank, (order, k) in enumerate(c.most_common(), start=1):
            if rank > 10:
                break
            row: Dict[str, object] = {
                "group": g,
                "rank_among_orders": rank,
                "order": " > ".join(order),
            }
            add_prob(row, "probability", k, n, z)
            exact_rows.append(row)
        if c:
            order, k = c.most_common(1)[0]
            lo, hi = wilson_interval(k, n, z)
            for pos, team in enumerate(order, start=1):
                pos_k = group_pos_counts[(g, pos)][team]
                pos_lo, pos_hi = wilson_interval(pos_k, n, z)
                ml_rows.append(
                    {
                        "group": g,
                        "predicted_position": pos,
                        "team": team,
                        "exact_order": " > ".join(order),
                        "exact_order_probability": k / n,
                        "exact_order_ci_low": lo,
                        "exact_order_ci_high": hi,
                        "team_position_probability": pos_k / n,
                        "team_position_ci_low": pos_lo,
                        "team_position_ci_high": pos_hi,
                    }
                )
    write_csv(out_dir / "group_exact_order_forecast.csv", exact_rows)
    write_csv(out_dir / "most_likely_group_standings.csv", ml_rows)


def write_group_match_forecast(out_dir: Path, state: Mapping[str, object], n: int, z: float) -> None:
    outcomes: Mapping[Tuple[int, str, str, str], Counter] = state["group_outcome_counts"]  # type: ignore[assignment]
    scores: Mapping[Tuple[int, str, str, str], Counter] = state["group_score_counts"]  # type: ignore[assignment]
    goal_sums: Mapping[Tuple[int, str, str, str], List[int]] = state["group_goal_sums"]  # type: ignore[assignment]
    rows: List[dict] = []
    for key in sorted(outcomes.keys()):
        mn, group, team1, team2 = key
        c = outcomes[key]
        score_counter = scores[key]
        modal_outcome, modal_k = most_common(c)
        modal_score, modal_score_k = most_common(score_counter)
        row: Dict[str, object] = {
            "match_no": mn,
            "group": group,
            "team1": team1,
            "team2": team2,
        }
        for outcome in ("team1", "draw", "team2"):
            add_prob(row, f"p_{outcome}", c[outcome], n, z)
        add_prob(row, "modal_outcome_probability", modal_k, n, z)
        row["modal_outcome"] = modal_outcome
        if isinstance(modal_score, tuple):
            row["modal_score"] = f"{modal_score[0]}-{modal_score[1]}"
        else:
            row["modal_score"] = ""
        add_prob(row, "modal_score_probability", modal_score_k, n, z)
        gs = goal_sums[key]
        row["avg_goals1"] = gs[0] / gs[2]
        row["avg_goals2"] = gs[1] / gs[2]
        rows.append(row)
    write_csv(out_dir / "group_match_forecast.csv", rows)


def write_third_place_outputs(out_dir: Path, state: Mapping[str, object], n: int, z: float) -> None:
    third_group_counts: Counter = state["third_group_counts"]  # type: ignore[assignment]
    third_combination_counts: Counter = state["third_combination_counts"]  # type: ignore[assignment]
    annex_option_counts: Counter = state["annex_option_counts"]  # type: ignore[assignment]

    rows: List[dict] = []
    for g in GROUPS:
        row: Dict[str, object] = {"group": g}
        add_prob(row, "third_place_group_qualifies", third_group_counts[g], n, z)
        rows.append(row)
    write_csv(out_dir / "third_place_group_qualification.csv", rows)

    combo_rows: List[dict] = []
    for rank, (combo, k) in enumerate(third_combination_counts.most_common(50), start=1):
        row: Dict[str, object] = {
            "rank": rank,
            "third_place_groups": "/".join(combo),
        }
        add_prob(row, "probability", k, n, z)
        combo_rows.append(row)
    write_csv(out_dir / "third_place_combination_forecast.csv", combo_rows)

    opt_rows: List[dict] = []
    for rank, (option, k) in enumerate(annex_option_counts.most_common(50), start=1):
        row = {"rank": rank, "annex_c_option": option}
        add_prob(row, "probability", k, n, z)
        opt_rows.append(row)
    write_csv(out_dir / "annex_c_option_forecast.csv", opt_rows)


def knockout_schedule_rows(r32_rows: Sequence[Mapping[str, str]], bracket_rows: Sequence[Mapping[str, str]]) -> List[dict]:
    rows: List[dict] = []
    for r in r32_rows:
        rows.append({"match_no": int(r["match_no"]), "round": "R32", "source1": r["side1"], "source2": r["side2"]})
    for r in bracket_rows:
        rows.append({"match_no": int(r["match_no"]), "round": r["round"], "source1": r["side1"], "source2": r["side2"]})
    return sorted(rows, key=lambda r: int(r["match_no"]))


def write_knockout_match_forecast(out_dir: Path, state: Mapping[str, object], n: int, z: float) -> None:
    r32_rows: Sequence[Mapping[str, str]] = state["r32_rows"]  # type: ignore[assignment]
    bracket_rows: Sequence[Mapping[str, str]] = state["bracket_rows"]  # type: ignore[assignment]
    ko_side_counts: Mapping[Tuple[int, str], Counter] = state["ko_side_counts"]  # type: ignore[assignment]
    ko_pair_counts: Mapping[int, Counter] = state["ko_pair_counts"]  # type: ignore[assignment]
    ko_pair_winner_counts: Mapping[Tuple[int, Tuple[str, str]], Counter] = state["ko_pair_winner_counts"]  # type: ignore[assignment]
    ko_winner_counts: Mapping[int, Counter] = state["ko_winner_counts"]  # type: ignore[assignment]

    rows: List[dict] = []
    for sched in knockout_schedule_rows(r32_rows, bracket_rows):
        mn = int(sched["match_no"])
        row: Dict[str, object] = {
            "match_no": mn,
            "round": sched["round"],
            "source1": sched["source1"],
            "source2": sched["source2"],
        }
        for side in ("side1", "side2"):
            team, k = most_common(ko_side_counts[(mn, side)])
            row[f"{side}_mode_team"] = team
            add_prob(row, f"{side}_mode_probability", k, n, z)
        pair, pair_k = most_common(ko_pair_counts[mn])
        if isinstance(pair, tuple):
            row["modal_pair"] = f"{pair[0]} vs {pair[1]}"
            pair_winner, pair_winner_k = most_common(ko_pair_winner_counts[(mn, pair)])
        else:
            row["modal_pair"] = ""
            pair_winner, pair_winner_k = "", 0
        add_prob(row, "modal_pair_probability", pair_k, n, z)
        row["modal_pair_winner"] = pair_winner
        # Conditional probability of the modal winner given the modal pair occurred.
        if pair_k > 0:
            lo, hi = wilson_interval(pair_winner_k, pair_k, z)
            row["modal_pair_winner_conditional_probability"] = pair_winner_k / pair_k
            row["modal_pair_winner_conditional_ci_low"] = lo
            row["modal_pair_winner_conditional_ci_high"] = hi
            row["modal_pair_winner_conditional_count"] = pair_winner_k
        else:
            row["modal_pair_winner_conditional_probability"] = math.nan
            row["modal_pair_winner_conditional_ci_low"] = math.nan
            row["modal_pair_winner_conditional_ci_high"] = math.nan
            row["modal_pair_winner_conditional_count"] = 0
        winner, winner_k = most_common(ko_winner_counts[mn])
        row["winner_mode_team"] = winner
        add_prob(row, "winner_mode_probability_unconditional", winner_k, n, z)
        rows.append(row)
    write_csv(out_dir / "knockout_match_forecast.csv", rows)


def write_modal_sampled_scenario(out_dir: Path, state: Mapping[str, object], args: argparse.Namespace, z: float) -> Optional[dict]:
    scenario_counts: Counter = state["scenario_counts"]  # type: ignore[assignment]
    scenario_examples: Mapping[str, dict] = state["scenario_examples"]  # type: ignore[assignment]
    if not scenario_counts:
        return None
    key, k = scenario_counts.most_common(1)[0]
    scenario = dict(scenario_examples[key])
    lo, hi = wilson_interval(k, args.n_sim, z)
    scenario["scenario_count"] = k
    scenario["scenario_probability"] = k / args.n_sim
    scenario["scenario_probability_ci_low"] = lo
    scenario["scenario_probability_ci_high"] = hi
    scenario["n_sim"] = args.n_sim
    scenario["seed"] = args.seed
    scenario["model_parameters"] = {
        "base_mu": args.base_mu,
        "goal_elo_scale": args.goal_elo_scale,
        "ko_elo_scale": args.ko_elo_scale,
        "host_advantage": args.host_advantage,
        "uncertainty_profile": args.uncertainty_profile,
        "team_rating_sd": args.team_rating_sd,
        "match_rating_sd": args.match_rating_sd,
        "match_shock_dist": args.match_shock_dist,
        "match_shock_df": args.match_shock_df,
        "goal_overdispersion": args.goal_overdispersion,
        "penalty_randomness": args.penalty_randomness,
        "elo_shrink": args.elo_shrink,
        "upset_prob": args.upset_prob,
        "upset_underdog_bonus": args.upset_underdog_bonus,
        "upset_shock_sd": args.upset_shock_sd,
        "depth_weight": args.depth_weight,
        "heat_weight": args.heat_weight,
        "humidity_weight": args.humidity_weight,
        "altitude_weight": args.altitude_weight,
        "travel_weight": args.travel_weight,
        "fallback_ranking": args.fallback_ranking,
        "ci_level": args.ci_level,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "modal_sampled_scenario.json").open("w", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# Modal full sampled scenario")
    lines.append("")
    lines.append(
        f"This is the most frequent complete scenario among the sampled simulations; it is not the same as the marginally most likely champion or match-by-match forecast. "
        f"It occurred {k}/{args.n_sim} times: {pct(k / args.n_sim)} "
        f"(Wilson {int(args.ci_level * 100)}% CI: {pct(lo)}--{pct(hi)})."
    )
    lines.append("")
    lines.append(f"Champion in this scenario: **{scenario['champion']}**")
    lines.append(f"Annexe C option: {scenario['annex_c_option']}")
    lines.append(f"Advanced third-place groups: {'/'.join(scenario['advanced_third_groups'])}")
    lines.append("")
    lines.append("## Group exact standings")
    for g in GROUPS:
        lines.append(f"- Group {g}: " + " > ".join(scenario["group_rankings"][g]))
    lines.append("")
    lines.append("## Knockout matches")
    for m in scenario["knockout_matches"]:
        score = f"{m['goals1']}-{m['goals2']}"
        suffix = "" if m.get("decided_by") == "90min" else " after ET/penalty proxy"
        lines.append(
            f"- M{m['match_no']} {m['round']}: {m['team1']} {score} {m['team2']} -> {m['winner']}{suffix}"
        )
    (out_dir / "modal_sampled_scenario.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A flat CSV version for quick inspection.
    scenario_rows: List[dict] = []
    for g in GROUPS:
        for pos, team in enumerate(scenario["group_rankings"][g], start=1):
            scenario_rows.append({"section": "group", "group": g, "position": pos, "match_no": "", "round": "", "team1": team, "team2": "", "score": "", "winner": ""})
    for m in scenario["knockout_matches"]:
        scenario_rows.append(
            {
                "section": "knockout",
                "group": "",
                "position": "",
                "match_no": m["match_no"],
                "round": m["round"],
                "team1": m["team1"],
                "team2": m["team2"],
                "score": f"{m['goals1']}-{m['goals2']}",
                "winner": m["winner"],
            }
        )
    write_csv(out_dir / "modal_sampled_scenario.csv", scenario_rows)
    return scenario


def write_upset_metrics_output(out_dir: Path, state: Mapping[str, object], z: float) -> None:
    metric_counts: Mapping[Tuple[str, str], Counter] = state["upset_metric_counts"]  # type: ignore[assignment]
    rows: List[dict] = []
    for (round_name, gap_bin), c in sorted(metric_counts.items()):
        total = c["total"]
        row: Dict[str, object] = {"round": round_name, "elo_gap_bin": gap_bin, "total_matches": total}
        add_prob(row, "underdog_win_rate", c["underdog_win"], total, z)
        add_prob(row, "favorite_nonwin_rate", c["favorite_nonwin"], total, z)
        add_prob(row, "explicit_upset_shock_rate", c["upset_applied"], total, z)
        rows.append(row)
    write_csv(out_dir / "upset_metrics.csv", rows)


def write_full_match_log_if_requested(out_dir: Path, state: Mapping[str, object]) -> None:
    full_match_log: Sequence[Mapping[str, object]] = state["full_match_log"]  # type: ignore[assignment]
    if full_match_log:
        preferred = [
            "sim_id", "match_no", "round", "group", "source1", "source2",
            "team1", "team2", "goals1", "goals2", "winner", "loser",
            "outcome", "decided_by", "elo_delta", "xg1_model_mean", "xg2_model_mean",
            "base_elo_gap", "elo_favorite", "elo_underdog", "underdog_win_by_elo",
            "favorite_nonwin_by_elo", "upset_applied", "upset_adjustment",
            "factor_delta", "depth_delta", "match_shock",
        ]
        all_keys = set().union(*(row.keys() for row in full_match_log))
        fieldnames = preferred + sorted(k for k in all_keys if k not in preferred)
        write_csv(out_dir / "simulated_match_log.csv", full_match_log, fieldnames)


def write_marginal_predictions_report(
    out_dir: Path,
    state: Mapping[str, object],
    team_rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    z: float,
) -> None:
    """Write a compact human-readable report of marginal modes.

    These are the relevant "most likely" predictions for practical use: the
    most likely champion, the modal exact order within each group, and the
    modal winner of each knockout slot.  Knockout match modes are marginal by
    match number and are not forced to form one globally coherent bracket.
    """
    group_order_counts: Mapping[str, Counter] = state["group_order_counts"]  # type: ignore[assignment]
    ko_pair_counts: Mapping[int, Counter] = state["ko_pair_counts"]  # type: ignore[assignment]
    ko_winner_counts: Mapping[int, Counter] = state["ko_winner_counts"]  # type: ignore[assignment]
    ko_pair_winner_counts: Mapping[Tuple[int, Tuple[str, str]], Counter] = state["ko_pair_winner_counts"]  # type: ignore[assignment]
    r32_rows: Sequence[Mapping[str, str]] = state["r32_rows"]  # type: ignore[assignment]
    bracket_rows: Sequence[Mapping[str, str]] = state["bracket_rows"]  # type: ignore[assignment]

    top = sorted(team_rows, key=lambda x: -float(x["champion"]))[:12]
    lines: List[str] = []
    lines.append("# Marginal most likely predictions")
    lines.append("")
    lines.append(
        "Probabilities are Monte Carlo estimates from the configured Elo model. "
        "Confidence intervals are Wilson intervals for Monte Carlo simulation error under the configured hierarchical uncertainty model. They do not cover all real-world model misspecification."
    )
    lines.append("")
    lines.append("## Model configuration")
    lines.append(f"- n_sim: {args.n_sim}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- base_mu: {args.base_mu}")
    lines.append(f"- goal_elo_scale: {args.goal_elo_scale}")
    lines.append(f"- ko_elo_scale: {args.ko_elo_scale}")
    lines.append(f"- host_advantage: {args.host_advantage}")
    lines.append(f"- uncertainty_profile: {args.uncertainty_profile}")
    lines.append(f"- team_rating_sd: {args.team_rating_sd}")
    lines.append(f"- match_rating_sd: {args.match_rating_sd}")
    lines.append(f"- goal_overdispersion: {args.goal_overdispersion}")
    lines.append(f"- penalty_randomness: {args.penalty_randomness}")
    lines.append(f"- fallback_ranking: {args.fallback_ranking}")
    lines.append("")
    lines.append("## Champion mode")
    for i, r in enumerate(top, start=1):
        lines.append(
            f"{i}. {r['team']}: {pct(float(r['champion']))} "
            f"[{pct(float(r['champion_ci_low']))}, {pct(float(r['champion_ci_high']))}]"
        )
    lines.append("")
    lines.append("## Modal exact group standings")
    for g in GROUPS:
        order, k = group_order_counts[g].most_common(1)[0]
        lo, hi = wilson_interval(k, args.n_sim, z)
        lines.append(f"- Group {g}: {' > '.join(order)}  —  {pct(k / args.n_sim)} [{pct(lo)}, {pct(hi)}]")
    lines.append("")
    lines.append("## Knockout match modal forecasts")
    lines.append(
        "Each line gives the modal participant pair for that match slot, the probability of that pair, "
        "and the unconditional modal winner probability for that match number."
    )
    for sched in knockout_schedule_rows(r32_rows, bracket_rows):
        mn = int(sched["match_no"])
        pair, pair_k = most_common(ko_pair_counts[mn])
        winner, winner_k = most_common(ko_winner_counts[mn])
        win_lo, win_hi = wilson_interval(winner_k, args.n_sim, z)
        if isinstance(pair, tuple):
            pair_str = f"{pair[0]} vs {pair[1]}"
            pair_lo, pair_hi = wilson_interval(pair_k, args.n_sim, z)
            pair_winner, pair_winner_k = most_common(ko_pair_winner_counts[(mn, pair)])
            cond = pair_winner_k / pair_k if pair_k else math.nan
            cond_text = f", modal-pair winner={pair_winner} conditional {pct(cond)}" if pair_k else ""
            lines.append(
                f"- M{mn} {sched['round']}: {pair_str} "
                f"pair {pct(pair_k / args.n_sim)} [{pct(pair_lo)}, {pct(pair_hi)}]; "
                f"winner mode={winner} {pct(winner_k / args.n_sim)} [{pct(win_lo)}, {pct(win_hi)}]"
                f"{cond_text}"
            )
        else:
            lines.append(f"- M{mn} {sched['round']}: no data")
    lines.append("")
    lines.append(
        "For a coherent but usually very low-frequency full sampled bracket, see modal_sampled_scenario.md."
    )
    (out_dir / "most_likely_predictions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary(out_dir: Path, team_rows: Sequence[Mapping[str, object]], scenario: Optional[dict], args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.append("World Cup 2026 Elo Monte Carlo simulation")
    lines.append(f"n_sim={args.n_sim}, seed={args.seed}")
    lines.append(
        "parameters="
        f"base_mu={args.base_mu}, goal_elo_scale={args.goal_elo_scale}, "
        f"ko_elo_scale={args.ko_elo_scale}, host_advantage={args.host_advantage}, "
        f"uncertainty_profile={args.uncertainty_profile}, team_rating_sd={args.team_rating_sd}, "
        f"match_rating_sd={args.match_rating_sd}, match_shock_dist={args.match_shock_dist}, "
        f"match_shock_df={args.match_shock_df}, goal_overdispersion={args.goal_overdispersion}, "
        f"penalty_randomness={args.penalty_randomness}, elo_shrink={args.elo_shrink}, "
        f"upset_prob={args.upset_prob}, upset_underdog_bonus={args.upset_underdog_bonus}, "
        f"depth_weight={args.depth_weight}, heat_weight={args.heat_weight}, "
        f"humidity_weight={args.humidity_weight}, altitude_weight={args.altitude_weight}, "
        f"travel_weight={args.travel_weight}, fallback_ranking={args.fallback_ranking}, "
        f"ci_level={args.ci_level}, dynamic_elo={args.dynamic_elo}"
    )
    lines.append("")
    lines.append("Top champion probabilities:")
    for r in sorted(team_rows, key=lambda x: -float(x["champion"]))[:12]:
        lines.append(
            f"  {str(r['team']):<24} {pct(float(r['champion'])):>7} "
            f"CI[{pct(float(r['champion_ci_low']))}, {pct(float(r['champion_ci_high']))}]"
        )
    if scenario is not None:
        lines.append("")
        lines.append(
            "Most frequent exact full sampled scenario, usually sparse at n=1000: "
            f"{scenario['champion']} champion, "
            f"prob={pct(float(scenario['scenario_probability']))} "
            f"CI[{pct(float(scenario['scenario_probability_ci_low']))}, {pct(float(scenario['scenario_probability_ci_high']))}], "
            f"Annexe C option={scenario['annex_c_option']}. "
            "Use most_likely_predictions.md and the CSV probability tables for marginal forecasts."
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "run_summary.txt").write_text(text, encoding="utf-8")
    return text


def write_outputs(state: Mapping[str, object], args: argparse.Namespace) -> str:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    z = state["z"]  # type: ignore[assignment]
    assert isinstance(z, float)

    team_rows = write_team_stage_probabilities(out_dir, state, args.n_sim, z)
    write_group_position_outputs(out_dir, state, args.n_sim, z)
    write_group_match_forecast(out_dir, state, args.n_sim, z)
    write_third_place_outputs(out_dir, state, args.n_sim, z)
    write_knockout_match_forecast(out_dir, state, args.n_sim, z)
    write_upset_metrics_output(out_dir, state, z)
    scenario = write_modal_sampled_scenario(out_dir, state, args, z)
    write_marginal_predictions_report(out_dir, state, team_rows, args, z)
    write_full_match_log_if_requested(out_dir, state)

    metadata = {
        "n_sim": args.n_sim,
        "seed": args.seed,
        "model": "Elo-Poisson group/knockout Monte Carlo with FIFA Annexe C third-place lookup",
        "parameters": {
            "teams_file": str(args.teams_file),
            "base_mu": args.base_mu,
            "goal_elo_scale": args.goal_elo_scale,
            "ko_elo_scale": args.ko_elo_scale,
            "host_advantage": args.host_advantage,
            "host_scope": args.host_scope,
            "contextual_effects": args.contextual_effects,
            "match_context_file": args.match_context_file,
            "include_static_environment_factors": args.include_static_environment_factors,
            "uncertainty_profile": args.uncertainty_profile,
            "team_rating_sd": args.team_rating_sd,
            "match_rating_sd": args.match_rating_sd,
            "goal_overdispersion": args.goal_overdispersion,
            "penalty_randomness": args.penalty_randomness,
            "fallback_ranking": args.fallback_ranking,
            "ci_level": args.ci_level,
            "save_match_log": args.save_match_log,
            "track_scenarios": args.track_scenarios,
            "dynamic_elo": {
                "enabled": args.dynamic_elo,
                "group_k": args.dynamic_elo_group_k,
                "k_by_round": knockout_dynamic_elo_k_by_round_from_args(args),
                "scale": args.dynamic_elo_scale,
                "host_advantage": args.dynamic_elo_home_advantage,
                "use_goal_difference_multiplier": not args.dynamic_elo_no_margin_multiplier,
                "round_delta": args.dynamic_elo_round_delta,
                "knockout_draw_value_for_advancer": args.dynamic_elo_knockout_draw_value,
                "formula": "R_new = R_old + K * G * (W - W_e); updated ratings affect subsequent simulated matches in the same tournament path",
            },
        },
        "confidence_interval": "Wilson interval for Monte Carlo sampling error under the configured hierarchical uncertainty model. It still does not cover all real-world model misspecification.",
        "limitations": [
            "Yellow/red card team-conduct scores are not simulated; if a tie reaches that criterion it is effectively neutral.",
            "If teams.csv does not contain fifa_rank, Elo is used as the deterministic ranking fallback.",
            "Elo-to-goals calibration is a modelling assumption controlled by base_mu and goal_elo_scale.",
            "The uncertainty layers widen predictive distributions, but they are still stylised rather than learned from historical match data in this package.",
            "Dynamic Elo, when enabled, is a path-dependent simulation update; it is not fitted by calibrating the whole tournament model to historical data in this package.",
            "team_factors.csv values are editable scenario priors; the supplied defaults are not a validated player-level model.",
            "Bundled match_context.csv venue values are heuristic scenario inputs, not measured weather forecasts.",
        ],
    }
    (out_dir / "simulation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return write_run_summary(out_dir, team_rows, scenario, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    ap = argparse.ArgumentParser(
        description="Elo-based Monte Carlo simulation for FIFA World Cup 2026 with Annexe C lookup."
    )
    ap.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory containing input CSV files.")
    ap.add_argument(
        "--teams-file",
        type=Path,
        default=Path("teams.csv"),
        help="Teams CSV to use, relative to --data-dir unless absolute. Use outputs_.../elo_recent/teams_elo_updated.csv after running update_elo_from_results.py.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for output CSV/JSON/MD files.")
    ap.add_argument("--n-sim", type=int, default=1000, help="Number of Monte Carlo repetitions.")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress and update progress.json every this many simulations. Use 0 to disable.",
    )
    ap.add_argument("--seed", type=int, default=20260609, help="Random seed.")
    ap.add_argument("--base-mu", type=float, default=1.35, help="Equal-Elo expected goals per team in 90 minutes.")
    ap.add_argument(
        "--goal-elo-scale",
        type=float,
        default=1600.0,
        help="Elo scale for converting rating differences to expected-goal ratios.",
    )
    ap.add_argument(
        "--ko-elo-scale",
        type=float,
        default=400.0,
        help="Elo-logistic scale used to resolve drawn knockout matches after 90 minutes.",
    )
    ap.add_argument(
        "--host-advantage",
        type=float,
        default=0.0,
        help="Elo bonus added to host teams marked host=1 in teams.csv.",
    )
    ap.add_argument(
        "--uncertainty-profile",
        choices=tuple(UNCERTAINTY_PRESETS.keys()),
        default="moderate",
        help="Preset for additional uncertainty. Individual uncertainty options override the preset.",
    )
    ap.add_argument(
        "--team-rating-sd",
        type=float,
        default=None,
        help="Persistent per-team tournament-level Elo uncertainty SD. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--match-rating-sd",
        type=float,
        default=None,
        help="Independent match-day shock SD added to the Elo difference. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--goal-overdispersion",
        type=float,
        default=None,
        help="Variance of a mean-one Gamma shock on each Poisson goal mean. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--penalty-randomness",
        type=float,
        default=None,
        help="Blend weight in [0,1] moving drawn knockout tie-breakers toward a 50/50 coin flip. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--elo-shrink",
        type=float,
        default=None,
        help="Multiplier on the input Elo gap before adding uncertainty/factors. Smaller values make favourites less dominant. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--match-shock-dist",
        choices=("normal", "student_t"),
        default=None,
        help="Distribution for match-day rating shocks. student_t gives more extreme one-off performances. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--match-shock-df",
        type=float,
        default=None,
        help="Degrees of freedom for Student-t match shocks. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--upset-prob",
        type=float,
        default=None,
        help="Per-match probability of an explicit underdog-favouring shock. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--upset-underdog-bonus",
        type=float,
        default=None,
        help="Mean Elo bonus applied to the effective underdog when an explicit upset shock occurs. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--upset-shock-sd",
        type=float,
        default=None,
        help="SD of the explicit underdog upset bonus. Default comes from --uncertainty-profile.",
    )
    ap.add_argument(
        "--upset-min-abs-delta",
        type=float,
        default=50.0,
        help="Minimum absolute effective Elo gap for explicit upset shocks to be considered.",
    )
    ap.add_argument(
        "--team-factors-file",
        default="team_factors.csv",
        help="Optional CSV under --data-dir with depth/climate/travel/upset prior factors.",
    )
    ap.add_argument(
        "--match-context-file",
        default="match_context.csv",
        help="Optional CSV under --data-dir with venue/date context by match_id or match_no.",
    )
    ap.add_argument(
        "--contextual-effects",
        action="store_true",
        help="Apply heat/humidity/altitude/travel effects match-by-match from --match-context-file.",
    )
    ap.add_argument(
        "--include-static-environment-factors",
        action="store_true",
        help="Also apply heat/humidity/altitude/travel as static team bonuses when --contextual-effects is enabled. Normally off to avoid double counting.",
    )
    ap.add_argument(
        "--host-scope",
        choices=("tournament", "venue", "hybrid", "none"),
        default="tournament",
        help="How --host-advantage is applied: tournament=any co-host anywhere; venue=own host country only; hybrid=half anywhere/full at own host country; none=off.",
    )
    ap.add_argument(
        "--confederation-bonus",
        action="append",
        default=[],
        help="Scenario Elo bonuses, e.g. CONMEBOL=30,CAF=10,UEFA=-5. Can be repeated.",
    )
    ap.add_argument(
        "--team-bonus",
        action="append",
        default=[],
        help="Scenario Elo bonuses by team, e.g. Japan=15,United States=25. Can be repeated.",
    )
    ap.add_argument("--depth-weight", type=float, default=0.0, help="Elo-point effect of depth factor at Final multiplier=1; earlier rounds use smaller multipliers.")
    ap.add_argument("--heat-weight", type=float, default=0.0, help="Elo-point effect of heat_adapt factor.")
    ap.add_argument("--humidity-weight", type=float, default=0.0, help="Elo-point effect of humidity_adapt factor.")
    ap.add_argument("--altitude-weight", type=float, default=0.0, help="Elo-point effect of altitude_adapt factor.")
    ap.add_argument("--travel-weight", type=float, default=0.0, help="Elo-point effect of travel_resilience factor.")
    ap.add_argument("--upset-potential-weight", type=float, default=0.0, help="Multiplicative effect of underdog upset_potential on explicit upset bonus.")
    ap.add_argument("--upset-resilience-weight", type=float, default=0.0, help="Multiplicative damping effect of favourite upset_resilience on explicit upset bonus.")
    ap.add_argument(
        "--fallback-ranking",
        choices=("elo", "fifa_rank", "name"),
        default="elo",
        help="Fallback after football tie-breakers and neutral conduct score. fifa_rank requires a fifa_rank column.",
    )
    ap.add_argument("--ci-level", type=float, default=0.95, help="Confidence level for Wilson intervals.")

    # Dynamic Elo updates inside each simulated tournament path.
    ap.add_argument("--dynamic-elo", action="store_true", help="Update Elo ratings after every simulated match and use the updated ratings for later matches in the same tournament path.")
    ap.add_argument("--dynamic-elo-group-k", type=float, default=40.0, help="Elo K-factor for simulated group-stage matches when --dynamic-elo is used.")
    ap.add_argument("--dynamic-elo-r32-k", type=float, default=50.0, help="Elo K-factor for simulated Round-of-32 matches.")
    ap.add_argument("--dynamic-elo-r16-k", type=float, default=50.0, help="Elo K-factor for simulated Round-of-16 matches.")
    ap.add_argument("--dynamic-elo-qf-k", type=float, default=55.0, help="Elo K-factor for simulated quarter-finals.")
    ap.add_argument("--dynamic-elo-sf-k", type=float, default=60.0, help="Elo K-factor for simulated semi-finals.")
    ap.add_argument("--dynamic-elo-final-k", type=float, default=60.0, help="Elo K-factor for the simulated final.")
    ap.add_argument("--dynamic-elo-third-place-k", type=float, default=50.0, help="Elo K-factor for the simulated third-place match.")
    ap.add_argument("--dynamic-elo-scale", type=float, default=400.0, help="Logistic scale used in simulated Elo updates.")
    ap.add_argument("--dynamic-elo-no-margin-multiplier", action="store_true", help="Disable the goal-difference multiplier in simulated Elo updates.")
    ap.add_argument("--dynamic-elo-round-delta", action="store_true", help="Round each simulated Elo update to an integer point.")
    ap.add_argument("--dynamic-elo-home-advantage", type=float, default=None, help="Home advantage used only in the expected-result term of Elo updates. Default: --host-advantage.")
    ap.add_argument("--dynamic-elo-knockout-draw-value", type=float, default=0.75, help="For drawn knockout matches, Elo score assigned to the advancing team. 0.5 gives no shootout credit; 1.0 treats it as a full win.")
    ap.add_argument(
        "--save-match-log",
        action="store_true",
        help="Save one row per simulated match. Useful for debugging; large for many simulations.",
    )
    ap.add_argument(
        "--no-track-scenarios",
        dest="track_scenarios",
        action="store_false",
        help="Disable tracking the most frequent complete sampled scenario.",
    )
    ap.set_defaults(track_scenarios=True)
    args = ap.parse_args()

    preset = UNCERTAINTY_PRESETS[args.uncertainty_profile]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if args.n_sim <= 0:
        raise ValueError("--n-sim must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.base_mu <= 0:
        raise ValueError("--base-mu must be positive")
    if args.goal_elo_scale <= 0:
        raise ValueError("--goal-elo-scale must be positive")
    if args.ko_elo_scale <= 0:
        raise ValueError("--ko-elo-scale must be positive")
    if args.team_rating_sd < 0:
        raise ValueError("--team-rating-sd must be non-negative")
    if args.match_rating_sd < 0:
        raise ValueError("--match-rating-sd must be non-negative")
    if args.goal_overdispersion < 0:
        raise ValueError("--goal-overdispersion must be non-negative")
    if not (0.0 <= args.penalty_randomness <= 1.0):
        raise ValueError("--penalty-randomness must be in [0, 1]")
    if not (0.0 <= args.elo_shrink <= 2.0):
        raise ValueError("--elo-shrink must be in [0, 2]")
    if args.match_shock_df <= 0:
        raise ValueError("--match-shock-df must be positive")
    if not (0.0 <= args.upset_prob <= 1.0):
        raise ValueError("--upset-prob must be in [0, 1]")
    if args.upset_underdog_bonus < 0:
        raise ValueError("--upset-underdog-bonus must be non-negative")
    if args.upset_shock_sd < 0:
        raise ValueError("--upset-shock-sd must be non-negative")
    if args.upset_min_abs_delta < 0:
        raise ValueError("--upset-min-abs-delta must be non-negative")
    if args.dynamic_elo:
        for opt in (
            "dynamic_elo_group_k",
            "dynamic_elo_r32_k",
            "dynamic_elo_r16_k",
            "dynamic_elo_qf_k",
            "dynamic_elo_sf_k",
            "dynamic_elo_final_k",
            "dynamic_elo_third_place_k",
        ):
            if getattr(args, opt) < 0:
                raise ValueError(f"--{opt.replace('_', '-')} must be non-negative")
        if args.dynamic_elo_scale <= 0:
            raise ValueError("--dynamic-elo-scale must be positive")
        if not (0.0 <= args.dynamic_elo_knockout_draw_value <= 1.0):
            raise ValueError("--dynamic-elo-knockout-draw-value must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    state = run_simulation(args)
    summary = write_outputs(state, args)
    print(summary, end="")
    print(f"\nWrote outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
