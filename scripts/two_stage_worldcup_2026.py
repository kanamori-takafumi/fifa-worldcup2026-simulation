#!/usr/bin/env python3
"""Two-stage World Cup 2026 simulator.

Stage 1 simulates only the group stage and extracts coherent group-stage
scenarios: group orders, best-third groups, Annexe C option and Round-of-32
fixture list.

Stage 2 fixes one or more selected group-stage scenarios and re-simulates the
knockout tournament conditional on each fixed Round-of-32 bracket.  This avoids
mixing unrelated marginal modes such as a Group F forecast that contains Japan
with a Round-of-32 modal card that does not.

The script reuses the match model and FIFA Annexe C lookup implemented in
simulate_worldcup_2026.py.  It has no third-party runtime dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Allow running as a standalone script from scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import simulate_worldcup_2026 as sim  # noqa: E402


@dataclass
class GroupScenario:
    scenario_id: str
    count: int
    probability: float
    ci_low: float
    ci_high: float
    selected_weight: float
    group_rankings: Dict[str, List[str]]
    qualifiers: Dict[str, str]
    advanced_third_groups: List[str]
    annex_c_option: int
    r32_matches: List[dict]
    scenario_type: str = "sampled_exact"
    selection_note: str = ""
    knockout_initial_elos: Dict[str, float] = field(default_factory=dict)
    knockout_initial_elos_source: str = ""


def parse_bonus_entries(entries: Optional[Sequence[str]]) -> Dict[str, float]:
    return sim.parse_bonus_entries(entries)


def apply_uncertainty_preset(args: argparse.Namespace) -> None:
    preset = sim.UNCERTAINTY_PRESETS[args.uncertainty_profile]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def normalise_args(args: argparse.Namespace) -> argparse.Namespace:
    apply_uncertainty_preset(args)
    if args.group_n_sim <= 0:
        raise ValueError("--group-n-sim must be positive")
    if args.knockout_n_sim <= 0:
        raise ValueError("--knockout-n-sim must be positive")
    if args.scenario_topk <= 0:
        raise ValueError("--scenario-topk must be positive")
    if args.base_mu <= 0:
        raise ValueError("--base-mu must be positive")
    if args.goal_elo_scale <= 0:
        raise ValueError("--goal-elo-scale must be positive")
    if args.ko_elo_scale <= 0:
        raise ValueError("--ko-elo-scale must be positive")
    if args.team_rating_sd < 0 or args.match_rating_sd < 0:
        raise ValueError("rating SDs must be non-negative")
    if args.goal_overdispersion < 0:
        raise ValueError("--goal-overdispersion must be non-negative")
    if not (0.0 <= args.penalty_randomness <= 1.0):
        raise ValueError("--penalty-randomness must be in [0,1]")
    if not (0.0 <= args.elo_shrink <= 2.0):
        raise ValueError("--elo-shrink must be in [0,2]")
    if args.match_shock_df <= 0:
        raise ValueError("--match-shock-df must be positive")
    if not (0.0 <= args.upset_prob <= 1.0):
        raise ValueError("--upset-prob must be in [0,1]")
    if args.upset_underdog_bonus < 0 or args.upset_shock_sd < 0:
        raise ValueError("upset bonus and SD must be non-negative")
    if args.upset_min_abs_delta < 0:
        raise ValueError("--upset-min-abs-delta must be non-negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
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
            raise ValueError("--dynamic-elo-knockout-draw-value must be in [0,1]")
        if args.dynamic_elo_audit_limit < 0:
            raise ValueError("--dynamic-elo-audit-limit must be non-negative")
    return args


def knockout_dynamic_elo_k_by_round(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "R32": args.dynamic_elo_r32_k,
        "R16": args.dynamic_elo_r16_k,
        "QF": args.dynamic_elo_qf_k,
        "SF": args.dynamic_elo_sf_k,
        "Final": args.dynamic_elo_final_k,
        "Third place": args.dynamic_elo_third_place_k,
    }


def scenario_key(group_rankings: Mapping[str, Sequence[str]], advanced_third_groups: Sequence[str]) -> str:
    payload = {
        "groups": {g: list(group_rankings[g]) for g in sim.GROUPS},
        "advanced_third_groups": sorted(advanced_third_groups),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_scenario_key(key: str) -> Tuple[Dict[str, List[str]], List[str]]:
    payload = json.loads(key)
    return {g: list(payload["groups"][g]) for g in sim.GROUPS}, list(payload["advanced_third_groups"])


def qualifiers_from_group_rankings(group_rankings: Mapping[str, Sequence[str]], advanced_third_groups: Sequence[str]) -> Dict[str, str]:
    q: Dict[str, str] = {}
    advanced_set = set(advanced_third_groups)
    for g in sim.GROUPS:
        order = list(group_rankings[g])
        q[f"{g}1"] = order[0]
        q[f"{g}2"] = order[1]
        q[f"{g}3"] = order[2]
        if g in advanced_set:
            q[f"{g}3q"] = order[2]
    return q


def make_r32_matches(
    qualifiers: Mapping[str, str],
    advanced_third_groups: Sequence[str],
    r32_rows: Sequence[Mapping[str, str]],
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]],
) -> Tuple[List[dict], Dict[int, str]]:
    third_assignment = sim.assign_thirds_to_slots(advanced_third_groups, annex_c_assignments)
    rows: List[dict] = []
    for r in r32_rows:
        mn = int(r["match_no"])
        team1 = sim.resolve_slot(r["side1"], qualifiers, third_assignment, mn)
        team2 = sim.resolve_slot(r["side2"], qualifiers, third_assignment, mn)
        source2 = f"3{third_assignment[mn]}" if r["side2"] == "3rd" else r["side2"]
        rows.append({
            "match_no": mn,
            "round": "R32",
            "source1": r["side1"],
            "source2": source2,
            "team1": team1,
            "team2": team2,
        })
    rows.sort(key=lambda x: int(x["match_no"]))
    return rows, dict(third_assignment)


def stage_for_match(match_no: int) -> str:
    return sim.ROUND_BY_MATCH[int(match_no)]


def knockout_schedule(r32_rows: Sequence[Mapping[str, str]], bracket_rows: Sequence[Mapping[str, str]]) -> List[dict]:
    rows: List[dict] = []
    for r in r32_rows:
        rows.append({"match_no": int(r["match_no"]), "round": "R32", "side1": r["side1"], "side2": r["side2"]})
    for r in bracket_rows:
        rows.append({"match_no": int(r["match_no"]), "round": r["round"], "side1": r["side1"], "side2": r["side2"]})
    return sorted(rows, key=lambda x: int(x["match_no"]))


def add_prob(row: Dict[str, object], prefix: str, k: int, n: int, z: float) -> None:
    lo, hi = sim.wilson_interval(k, n, z)
    row[prefix] = k / n if n else math.nan
    row[f"{prefix}_ci_low"] = lo
    row[f"{prefix}_ci_high"] = hi
    row[f"{prefix}_count"] = k


def pct(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if math.isnan(v):
        return ""
    return f"{100.0 * v:.1f}%"


def load_model_inputs(args: argparse.Namespace) -> dict:
    teams_path = sim.resolve_data_path(args.data_dir, args.teams_file)
    teams = sim.load_teams(teams_path)
    team_factors = sim.load_team_factors(args.data_dir / args.team_factors_file, teams)
    match_contexts = sim.load_match_contexts(args.data_dir / args.match_context_file) if getattr(args, "contextual_effects", False) else {}
    static_factor_adjustments = sim.compute_static_factor_adjustments(teams, team_factors, args)
    group_fixtures = sim.read_csv(args.data_dir / "group_stage.csv")
    r32_rows = sim.read_csv(args.data_dir / "round32_slots.csv")
    bracket_rows = sim.read_csv(args.data_dir / "knockout_bracket.csv")
    annex_c_assignments, annex_option_by_key = sim.load_annex_c_assignments(
        args.data_dir / "annex_c_third_place_assignments.csv", r32_rows
    )
    return {
        "teams": teams,
        "team_factors": team_factors,
        "match_contexts": match_contexts,
        "static_factor_adjustments": static_factor_adjustments,
        "group_fixtures": group_fixtures,
        "r32_rows": r32_rows,
        "bracket_rows": bracket_rows,
        "annex_c_assignments": annex_c_assignments,
        "annex_option_by_key": annex_option_by_key,
    }


def run_group_stage(args: argparse.Namespace, data: Mapping[str, object]) -> dict:
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    team_factors: Mapping[str, sim.TeamFactors] = data["team_factors"]  # type: ignore[assignment]
    match_contexts: Mapping[str, sim.MatchContext] = data["match_contexts"]  # type: ignore[assignment]
    static_factor_adjustments: Mapping[str, float] = data["static_factor_adjustments"]  # type: ignore[assignment]
    fixtures: Sequence[Mapping[str, str]] = data["group_fixtures"]  # type: ignore[assignment]
    r32_rows: Sequence[Mapping[str, str]] = data["r32_rows"]  # type: ignore[assignment]
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]] = data["annex_c_assignments"]  # type: ignore[assignment]
    annex_option_by_key: Mapping[Tuple[str, ...], int] = data["annex_option_by_key"]  # type: ignore[assignment]

    rng = random.Random(args.group_seed if args.group_seed is not None else args.seed)
    z = sim.normal_z(args.ci_level)
    scenario_counts: Counter[str] = Counter()
    scenario_examples: Dict[str, dict] = {}
    group_order_counts: Dict[str, Counter] = {g: Counter() for g in sim.GROUPS}
    group_pos_counts: DefaultDict[Tuple[str, int], Counter] = defaultdict(Counter)
    r32_counts: Counter[str] = Counter()
    third_group_counts: Counter[str] = Counter()
    annex_option_counts: Counter[int] = Counter()
    group_match_counts: DefaultDict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    dynamic_elo_audit_rows: List[dict] = []
    scenario_terminal_elo_sums: DefaultDict[str, Counter] = defaultdict(Counter)
    scenario_terminal_elo_counts: Counter[str] = Counter()
    overall_terminal_elo_sums: Counter[str] = Counter()
    overall_terminal_elo_count = 0

    start = time.time()
    for sid in range(1, args.group_n_sim + 1):
        rating_adjustments = sim.sample_tournament_rating_adjustments(teams, rng, args.team_rating_sd)
        dynamic_elo_state: Dict[str, float] = {}
        qualifiers, group_rankings, group_records, g_log, advanced_third_groups = sim.simulate_group_stage(
            teams=teams,
            fixtures=fixtures,
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
        key = scenario_key(group_rankings, advanced_third_groups)
        scenario_counts[key] += 1
        if args.dynamic_elo and dynamic_elo_state:
            scenario_terminal_elo_counts[key] += 1
            overall_terminal_elo_count += 1
            for team_name, elo_value in dynamic_elo_state.items():
                scenario_terminal_elo_sums[key][team_name] += float(elo_value)
                overall_terminal_elo_sums[team_name] += float(elo_value)

        third_key = tuple(sorted(advanced_third_groups))
        annex_option = annex_option_by_key[third_key]
        annex_option_counts[annex_option] += 1
        for g in advanced_third_groups:
            third_group_counts[g] += 1

        r32_matches, third_assignment = make_r32_matches(qualifiers, advanced_third_groups, r32_rows, annex_c_assignments)
        if key not in scenario_examples:
            scenario_examples[key] = {
                "group_rankings": {g: list(group_rankings[g]) for g in sim.GROUPS},
                "qualifiers": dict(qualifiers),
                "advanced_third_groups": list(advanced_third_groups),
                "annex_c_option": annex_option,
                "third_assignment": {str(k): v for k, v in third_assignment.items()},
                "r32_matches": r32_matches,
            }

        for g in sim.GROUPS:
            order = tuple(group_rankings[g])
            group_order_counts[g][order] += 1
            for pos, name in enumerate(order, start=1):
                group_pos_counts[(g, pos)][name] += 1
        for name in set([qualifiers[f"{g}1"] for g in sim.GROUPS] + [qualifiers[f"{g}2"] for g in sim.GROUPS] + [qualifiers[f"{g}3q"] for g in advanced_third_groups]):
            r32_counts[name] += 1
        for m in g_log:
            group_match_counts[(str(m["match_no"]), str(m["team1"]), str(m["team2"]))][str(m["outcome"])] += 1
            if args.dynamic_elo and len(dynamic_elo_audit_rows) < args.dynamic_elo_audit_limit:
                audit = {"sim_id": sid}
                audit.update(m)
                dynamic_elo_audit_rows.append(audit)

        if args.progress_every and (sid % args.progress_every == 0 or sid == args.group_n_sim):
            elapsed = time.time() - start
            rate = sid / elapsed if elapsed > 0 else float("nan")
            eta = (args.group_n_sim - sid) / rate if rate > 0 else float("nan")
            print(f"group progress: {sid}/{args.group_n_sim} ({sid/args.group_n_sim:.1%}), elapsed={elapsed:.1f}s, eta={eta:.1f}s", flush=True)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "group_progress.json").write_text(json.dumps({
                "stage": "group",
                "completed": sid,
                "total": args.group_n_sim,
                "percent": round(100.0 * sid / args.group_n_sim, 2),
                "elapsed_sec": round(elapsed, 1),
                "sim_per_sec": round(rate, 3),
                "eta_sec": round(eta, 1),
            }, indent=2), encoding="utf-8")

    return {
        "scenario_counts": scenario_counts,
        "scenario_examples": scenario_examples,
        "group_order_counts": group_order_counts,
        "group_pos_counts": group_pos_counts,
        "r32_counts": r32_counts,
        "third_group_counts": third_group_counts,
        "annex_option_counts": annex_option_counts,
        "group_match_counts": group_match_counts,
        "dynamic_elo_audit_rows": dynamic_elo_audit_rows,
        "scenario_terminal_elo_sums": scenario_terminal_elo_sums,
        "scenario_terminal_elo_counts": scenario_terminal_elo_counts,
        "overall_terminal_elo_sums": overall_terminal_elo_sums,
        "overall_terminal_elo_count": overall_terminal_elo_count,
        "z": z,
    }


def group_terminal_elo_mean_for_key(
    key: str,
    teams: Mapping[str, sim.Team],
    group_state: Mapping[str, object],
) -> Tuple[Dict[str, float], str]:
    """Mean dynamic Elo after the simulated group stage for a scenario key.

    Exact sampled scenarios get their scenario-conditional mean.  A constructed
    consensus scenario may never have appeared exactly, so it falls back to the
    unconditional mean terminal group-stage Elo.
    """
    counts: Counter[str] = group_state.get("scenario_terminal_elo_counts", Counter())  # type: ignore[assignment]
    sums: Mapping[str, Counter] = group_state.get("scenario_terminal_elo_sums", {})  # type: ignore[assignment]
    n_key = int(counts.get(key, 0))
    if n_key > 0 and key in sums:
        return (
            {name: float(sums[key].get(name, teams[name].elo)) / n_key for name in teams},
            f"scenario_conditional_group_terminal_mean_n={n_key}",
        )

    overall_n = int(group_state.get("overall_terminal_elo_count", 0) or 0)
    overall_sums: Counter[str] = group_state.get("overall_terminal_elo_sums", Counter())  # type: ignore[assignment]
    if overall_n > 0:
        return (
            {name: float(overall_sums.get(name, teams[name].elo)) / overall_n for name in teams},
            f"unconditional_group_terminal_mean_n={overall_n}",
        )

    return {}, "static_pre_tournament_elo"


def select_group_scenarios(args: argparse.Namespace, data: Mapping[str, object], group_state: Mapping[str, object]) -> List[GroupScenario]:
    """Select fixed group-stage scenarios for conditional knockout simulation.

    topk_exact: use the most frequent exact sampled 12-group scenario.  This is
    mathematically clean but usually very low-mass because the state space is huge.

    consensus: construct one representative scenario from marginal group modes
    plus the eight groups with highest third-place qualification probability.
    This is usually better for explanatory bracket graphics.  It is coherent,
    but not an observed exact sample and therefore has no honest exact scenario
    probability.
    """
    scenario_counts: Counter[str] = group_state["scenario_counts"]  # type: ignore[assignment]
    examples: Mapping[str, dict] = group_state["scenario_examples"]  # type: ignore[assignment]
    group_order_counts: Mapping[str, Counter] = group_state["group_order_counts"]  # type: ignore[assignment]
    third_group_counts: Counter[str] = group_state["third_group_counts"]  # type: ignore[assignment]
    z: float = group_state["z"]  # type: ignore[assignment]
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    r32_rows: Sequence[Mapping[str, str]] = data["r32_rows"]  # type: ignore[assignment]
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]] = data["annex_c_assignments"]  # type: ignore[assignment]
    annex_option_by_key: Mapping[Tuple[str, ...], int] = data["annex_option_by_key"]  # type: ignore[assignment]

    if args.scenario_selection == "consensus":
        group_rankings: Dict[str, List[str]] = {}
        for g in sim.GROUPS:
            order, _k = group_order_counts[g].most_common(1)[0]
            group_rankings[g] = list(order)
        # Top eight groups by probability that their third-place team qualifies.
        advanced = sorted([g for g, _ in third_group_counts.most_common(8)])
        qualifiers = qualifiers_from_group_rankings(group_rankings, advanced)
        r32_matches, _third_assignment = make_r32_matches(qualifiers, advanced, r32_rows, annex_c_assignments)
        annex_option = annex_option_by_key[tuple(sorted(advanced))]
        consensus_key = scenario_key(group_rankings, advanced)
        initial_elos, initial_elos_source = group_terminal_elo_mean_for_key(consensus_key, teams, group_state) if args.dynamic_elo else ({}, "static_pre_tournament_elo")
        return [GroupScenario(
            scenario_id="C01",
            count=0,
            probability=math.nan,
            ci_low=math.nan,
            ci_high=math.nan,
            selected_weight=1.0,
            group_rankings=group_rankings,
            qualifiers=qualifiers,
            advanced_third_groups=advanced,
            annex_c_option=annex_option,
            r32_matches=r32_matches,
            scenario_type="consensus_marginal",
            selection_note="Constructed from each group's modal order and the eight groups with the largest third-place qualification probability; not an exact sampled scenario.",
            knockout_initial_elos=initial_elos,
            knockout_initial_elos_source=initial_elos_source,
        )]

    top = scenario_counts.most_common(args.scenario_topk)
    mass = sum(k for _, k in top) / args.group_n_sim
    selected: List[GroupScenario] = []
    for i, (key, k) in enumerate(top, start=1):
        ex = examples[key]
        lo, hi = sim.wilson_interval(k, args.group_n_sim, z)
        initial_elos, initial_elos_source = group_terminal_elo_mean_for_key(key, teams, group_state) if args.dynamic_elo else ({}, "static_pre_tournament_elo")
        selected.append(GroupScenario(
            scenario_id=f"S{i:02d}",
            count=k,
            probability=k / args.group_n_sim,
            ci_low=lo,
            ci_high=hi,
            selected_weight=(k / args.group_n_sim) / mass if mass > 0 else math.nan,
            group_rankings={g: list(ex["group_rankings"][g]) for g in sim.GROUPS},
            qualifiers=dict(ex["qualifiers"]),
            advanced_third_groups=list(ex["advanced_third_groups"]),
            annex_c_option=int(ex["annex_c_option"]),
            r32_matches=list(ex["r32_matches"]),
            scenario_type="sampled_exact",
            selection_note="Exact group-stage scenario sampled during Stage 1.",
            knockout_initial_elos=initial_elos,
            knockout_initial_elos_source=initial_elos_source,
        ))
    return selected


def collect_knockout_stage_counts(stage_counts: Dict[str, Counter], winners: Mapping[int, str], losers: Mapping[int, str]) -> None:
    for mn in range(73, 89):
        stage_counts["reach_r16"][winners[mn]] += 1
    for mn in range(89, 97):
        stage_counts["reach_qf"][winners[mn]] += 1
    for mn in range(97, 101):
        stage_counts["reach_sf"][winners[mn]] += 1
    for mn in (101, 102):
        stage_counts["reach_final"][winners[mn]] += 1
    stage_counts["third_place"][winners[103]] += 1
    stage_counts["runner_up"][losers[104]] += 1
    stage_counts["champion"][winners[104]] += 1


def run_knockout_for_scenario(args: argparse.Namespace, data: Mapping[str, object], scenario: GroupScenario, scenario_index: int) -> dict:
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    team_factors: Mapping[str, sim.TeamFactors] = data["team_factors"]  # type: ignore[assignment]
    match_contexts: Mapping[str, sim.MatchContext] = data["match_contexts"]  # type: ignore[assignment]
    static_factor_adjustments: Mapping[str, float] = data["static_factor_adjustments"]  # type: ignore[assignment]
    r32_rows: Sequence[Mapping[str, str]] = data["r32_rows"]  # type: ignore[assignment]
    bracket_rows: Sequence[Mapping[str, str]] = data["bracket_rows"]  # type: ignore[assignment]
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]] = data["annex_c_assignments"]  # type: ignore[assignment]

    base_seed = args.knockout_seed if args.knockout_seed is not None else args.seed + 100000
    rng = random.Random(base_seed + scenario_index)
    z = sim.normal_z(args.ci_level)

    # reach_r32 is fixed by the selected group-stage scenario.
    stage_counts: Dict[str, Counter] = {s: Counter() for s in ["reach_r32", "reach_r16", "reach_qf", "reach_sf", "reach_final", "runner_up", "third_place", "champion"]}
    r32_participants = set()
    for g in sim.GROUPS:
        r32_participants.add(scenario.qualifiers[f"{g}1"])
        r32_participants.add(scenario.qualifiers[f"{g}2"])
    for g in scenario.advanced_third_groups:
        r32_participants.add(scenario.qualifiers[f"{g}3q"])
    for t in r32_participants:
        stage_counts["reach_r32"][t] = args.knockout_n_sim

    ko_pair_counts: DefaultDict[int, Counter] = defaultdict(Counter)
    ko_pair_winner_counts: DefaultDict[Tuple[int, Tuple[str, str]], Counter] = defaultdict(Counter)
    ko_winner_counts: DefaultDict[int, Counter] = defaultdict(Counter)
    score_counts: DefaultDict[Tuple[int, Tuple[str, str]], Counter] = defaultdict(Counter)
    dynamic_elo_audit_rows: List[dict] = []

    start = time.time()
    for sid in range(1, args.knockout_n_sim + 1):
        rating_adjustments = sim.sample_tournament_rating_adjustments(teams, rng, args.team_rating_sd)
        winners, losers, k_log, _third_assignment = sim.simulate_knockout(
            teams=teams,
            qualifiers=scenario.qualifiers,
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
            dynamic_elo_k_by_round=knockout_dynamic_elo_k_by_round(args),
            dynamic_elo_scale=args.dynamic_elo_scale,
            dynamic_elo_margin=not args.dynamic_elo_no_margin_multiplier,
            dynamic_elo_round_delta=args.dynamic_elo_round_delta,
            dynamic_elo_home_advantage=args.dynamic_elo_home_advantage,
            dynamic_elo_knockout_draw_value=args.dynamic_elo_knockout_draw_value,
            dynamic_elo_initial=scenario.knockout_initial_elos if args.dynamic_elo and scenario.knockout_initial_elos else None,
            match_contexts=match_contexts,
            contextual_effects=args.contextual_effects,
            host_scope=args.host_scope,
            heat_weight=args.heat_weight,
            humidity_weight=args.humidity_weight,
            altitude_weight=args.altitude_weight,
            travel_weight=args.travel_weight,
        )
        collect_knockout_stage_counts(stage_counts, winners, losers)
        for m in k_log:
            mn = int(m["match_no"])
            pair = (str(m["team1"]), str(m["team2"]))
            winner = str(m["winner"])
            ko_pair_counts[mn][pair] += 1
            ko_pair_winner_counts[(mn, pair)][winner] += 1
            ko_winner_counts[mn][winner] += 1
            score_counts[(mn, pair)][(int(m["goals1"]), int(m["goals2"]))] += 1
            if args.dynamic_elo and len(dynamic_elo_audit_rows) < args.dynamic_elo_audit_limit:
                audit = {"scenario_id": scenario.scenario_id, "sim_id": sid}
                audit.update(m)
                dynamic_elo_audit_rows.append(audit)

        if args.progress_every and (sid % args.progress_every == 0 or sid == args.knockout_n_sim):
            elapsed = time.time() - start
            rate = sid / elapsed if elapsed > 0 else float("nan")
            eta = (args.knockout_n_sim - sid) / rate if rate > 0 else float("nan")
            print(f"knockout {scenario.scenario_id} progress: {sid}/{args.knockout_n_sim} ({sid/args.knockout_n_sim:.1%}), elapsed={elapsed:.1f}s, eta={eta:.1f}s", flush=True)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"{scenario.scenario_id}_knockout_progress.json").write_text(json.dumps({
                "stage": "knockout",
                "scenario_id": scenario.scenario_id,
                "completed": sid,
                "total": args.knockout_n_sim,
                "percent": round(100.0 * sid / args.knockout_n_sim, 2),
                "elapsed_sec": round(elapsed, 1),
                "sim_per_sec": round(rate, 3),
                "eta_sec": round(eta, 1),
            }, indent=2), encoding="utf-8")

    return {
        "scenario": scenario,
        "stage_counts": stage_counts,
        "ko_pair_counts": ko_pair_counts,
        "ko_pair_winner_counts": ko_pair_winner_counts,
        "ko_winner_counts": ko_winner_counts,
        "score_counts": score_counts,
        "dynamic_elo_audit_rows": dynamic_elo_audit_rows,
        "z": z,
    }


def write_group_outputs(args: argparse.Namespace, data: Mapping[str, object], group_state: Mapping[str, object], selected: Sequence[GroupScenario]) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    z: float = group_state["z"]  # type: ignore[assignment]
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    group_order_counts: Mapping[str, Counter] = group_state["group_order_counts"]  # type: ignore[assignment]
    group_pos_counts: Mapping[Tuple[str, int], Counter] = group_state["group_pos_counts"]  # type: ignore[assignment]
    r32_counts: Counter[str] = group_state["r32_counts"]  # type: ignore[assignment]
    third_group_counts: Counter[str] = group_state["third_group_counts"]  # type: ignore[assignment]
    annex_option_counts: Counter[int] = group_state["annex_option_counts"]  # type: ignore[assignment]
    scenario_counts: Counter[str] = group_state["scenario_counts"]  # type: ignore[assignment]

    # All scenarios, top 200 by default for inspection.
    rows = []
    for rank, (key, k) in enumerate(scenario_counts.most_common(args.write_scenario_limit), start=1):
        gr, adv = decode_scenario_key(key)
        lo, hi = sim.wilson_interval(k, args.group_n_sim, z)
        rows.append({
            "rank": rank,
            "count": k,
            "probability": k / args.group_n_sim,
            "ci_low": lo,
            "ci_high": hi,
            "advanced_third_groups": "/".join(adv),
            "group_orders": " | ".join(f"{g}:{' > '.join(gr[g])}" for g in sim.GROUPS),
        })
    sim.write_csv(out / "group_stage_scenarios_top.csv", rows)

    selected_mass = (math.nan if args.scenario_selection == "consensus" else sum(s.probability for s in selected if not math.isnan(float(s.probability))))
    selected_rows = []
    r32_rows = []
    group_rows = []
    for s in selected:
        selected_rows.append({
            "scenario_id": s.scenario_id,
            "group_count": s.count,
            "group_probability": s.probability,
            "group_ci_low": s.ci_low,
            "group_ci_high": s.ci_high,
            "selected_weight_normalized": s.selected_weight,
            "selected_mass_total": selected_mass,
            "scenario_type": s.scenario_type,
            "selection_note": s.selection_note,
            "knockout_initial_elos_source": s.knockout_initial_elos_source,
            "annex_c_option": s.annex_c_option,
            "advanced_third_groups": "/".join(s.advanced_third_groups),
            "champion_after_conditional_ko": "",
            "group_orders": " | ".join(f"{g}:{' > '.join(s.group_rankings[g])}" for g in sim.GROUPS),
        })
        for g in sim.GROUPS:
            for pos, team in enumerate(s.group_rankings[g], start=1):
                group_rows.append({
                    "scenario_id": s.scenario_id,
                    "group": g,
                    "position": pos,
                    "team": team,
                    "status": "auto" if pos <= 2 else ("best_third" if g in s.advanced_third_groups else "out"),
                })
        for m in s.r32_matches:
            row = {"scenario_id": s.scenario_id}
            row.update(m)
            r32_rows.append(row)
        (out / f"{s.scenario_id}_group_scenario.json").write_text(json.dumps({
            "scenario_id": s.scenario_id,
            "group_count": s.count,
            "group_probability": s.probability,
            "group_ci_low": s.ci_low,
            "group_ci_high": s.ci_high,
            "selected_weight_normalized": s.selected_weight,
            "scenario_type": s.scenario_type,
            "selection_note": s.selection_note,
            "knockout_initial_elos_source": s.knockout_initial_elos_source,
            "knockout_initial_elos": s.knockout_initial_elos,
            "annex_c_option": s.annex_c_option,
            "advanced_third_groups": s.advanced_third_groups,
            "group_rankings": s.group_rankings,
            "qualifiers": s.qualifiers,
            "r32_matches": s.r32_matches,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    sim.write_csv(out / "selected_group_scenarios.csv", selected_rows)
    sim.write_csv(out / "selected_group_scenario_standings.csv", group_rows)
    sim.write_csv(out / "selected_group_scenario_r32.csv", r32_rows)

    # Marginal group-stage outputs.
    team_rows = []
    for name, t in sorted(teams.items(), key=lambda kv: (kv[1].group, kv[1].position)):
        row: Dict[str, object] = {"team": name, "group": t.group, "position": t.position, "elo": t.elo, "confederation": t.confederation}
        add_prob(row, "reach_r32", r32_counts[name], args.group_n_sim, z)
        for pos in range(1, 5):
            add_prob(row, f"group_pos_{pos}", group_pos_counts[(t.group, pos)][name], args.group_n_sim, z)
        team_rows.append(row)
    sim.write_csv(out / "group_stage_team_probabilities.csv", team_rows)

    exact_rows = []
    for g in sim.GROUPS:
        order, k = group_order_counts[g].most_common(1)[0]
        lo, hi = sim.wilson_interval(k, args.group_n_sim, z)
        exact_rows.append({"group": g, "modal_order": " > ".join(order), "probability": k / args.group_n_sim, "ci_low": lo, "ci_high": hi, "count": k})
    sim.write_csv(out / "group_stage_modal_orders.csv", exact_rows)

    third_rows = []
    for g in sim.GROUPS:
        row = {"group": g}
        add_prob(row, "third_place_group_qualifies", third_group_counts[g], args.group_n_sim, z)
        third_rows.append(row)
    sim.write_csv(out / "third_place_group_qualification.csv", third_rows)

    annex_rows = []
    for option, k in annex_option_counts.most_common(50):
        lo, hi = sim.wilson_interval(k, args.group_n_sim, z)
        annex_rows.append({"annex_c_option": option, "count": k, "probability": k / args.group_n_sim, "ci_low": lo, "ci_high": hi})
    sim.write_csv(out / "annex_c_option_forecast.csv", annex_rows)

    if args.dynamic_elo:
        audit_rows: Sequence[Mapping[str, object]] = group_state.get("dynamic_elo_audit_rows", [])  # type: ignore[assignment]
        if audit_rows:
            sim.write_csv(out / "dynamic_elo_group_updates_sample.csv", audit_rows)

        overall_n = int(group_state.get("overall_terminal_elo_count", 0) or 0)
        overall_sums: Counter[str] = group_state.get("overall_terminal_elo_sums", Counter())  # type: ignore[assignment]
        terminal_rows: List[dict] = []
        if overall_n > 0:
            for name, t in sorted(teams.items(), key=lambda kv: (kv[1].group, kv[1].position)):
                mean_elo = float(overall_sums.get(name, t.elo)) / overall_n
                terminal_rows.append({
                    "team": name,
                    "group": t.group,
                    "initial_elo": t.elo,
                    "mean_terminal_group_elo": mean_elo,
                    "mean_group_stage_elo_change": mean_elo - t.elo,
                    "n_group_sim": overall_n,
                })
            sim.write_csv(out / "dynamic_elo_group_terminal_mean.csv", terminal_rows)

        selected_elo_rows: List[dict] = []
        for s in selected:
            for name, t in sorted(teams.items(), key=lambda kv: (kv[1].group, kv[1].position)):
                ko_elo = s.knockout_initial_elos.get(name, t.elo)
                selected_elo_rows.append({
                    "scenario_id": s.scenario_id,
                    "team": name,
                    "group": t.group,
                    "initial_elo": t.elo,
                    "knockout_initial_elo": ko_elo,
                    "knockout_initial_elo_change": ko_elo - t.elo,
                    "knockout_initial_elos_source": s.knockout_initial_elos_source,
                })
        if selected_elo_rows:
            sim.write_csv(out / "selected_scenario_knockout_initial_elos.csv", selected_elo_rows)


def write_knockout_outputs(args: argparse.Namespace, data: Mapping[str, object], k_state: Mapping[str, object]) -> Tuple[List[dict], List[dict], List[dict]]:
    out = args.out_dir
    scenario: GroupScenario = k_state["scenario"]  # type: ignore[assignment]
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    r32_rows: Sequence[Mapping[str, str]] = data["r32_rows"]  # type: ignore[assignment]
    bracket_rows: Sequence[Mapping[str, str]] = data["bracket_rows"]  # type: ignore[assignment]
    z: float = k_state["z"]  # type: ignore[assignment]
    stage_counts: Mapping[str, Counter] = k_state["stage_counts"]  # type: ignore[assignment]
    ko_pair_counts: Mapping[int, Counter] = k_state["ko_pair_counts"]  # type: ignore[assignment]
    ko_pair_winner_counts: Mapping[Tuple[int, Tuple[str, str]], Counter] = k_state["ko_pair_winner_counts"]  # type: ignore[assignment]
    ko_winner_counts: Mapping[int, Counter] = k_state["ko_winner_counts"]  # type: ignore[assignment]
    score_counts: Mapping[Tuple[int, Tuple[str, str]], Counter] = k_state["score_counts"]  # type: ignore[assignment]

    team_rows: List[dict] = []
    for name, t in sorted(teams.items(), key=lambda kv: -stage_counts["champion"][kv[0]]):
        row: Dict[str, object] = {
            "scenario_id": scenario.scenario_id,
            "team": name,
            "group": t.group,
            "elo": t.elo,
            "knockout_initial_elo": scenario.knockout_initial_elos.get(name, t.elo) if args.dynamic_elo else t.elo,
            "knockout_initial_elos_source": scenario.knockout_initial_elos_source if args.dynamic_elo else "static_pre_tournament_elo",
        }
        for st in ["reach_r32", "reach_r16", "reach_qf", "reach_sf", "reach_final", "runner_up", "third_place", "champion"]:
            add_prob(row, st, stage_counts[st][name], args.knockout_n_sim, z)
        team_rows.append(row)
    sim.write_csv(out / f"{scenario.scenario_id}_conditional_team_stage_probabilities.csv", team_rows)

    match_rows: List[dict] = []
    for sched in knockout_schedule(r32_rows, bracket_rows):
        mn = int(sched["match_no"])
        pair, pair_k = ko_pair_counts[mn].most_common(1)[0] if ko_pair_counts[mn] else (("", ""), 0)
        winner, winner_k = ko_winner_counts[mn].most_common(1)[0] if ko_winner_counts[mn] else ("", 0)
        pair_lo, pair_hi = sim.wilson_interval(pair_k, args.knockout_n_sim, z)
        win_lo, win_hi = sim.wilson_interval(winner_k, args.knockout_n_sim, z)
        pair_winner, pair_winner_k = ("", 0)
        cond = math.nan
        score_mode = ""
        score_mode_prob = math.nan
        if isinstance(pair, tuple) and pair_k:
            pair_winner, pair_winner_k = ko_pair_winner_counts[(mn, pair)].most_common(1)[0]
            cond = pair_winner_k / pair_k
            score, score_k = score_counts[(mn, pair)].most_common(1)[0]
            score_mode = f"{score[0]}-{score[1]}"
            score_mode_prob = score_k / pair_k
        match_rows.append({
            "scenario_id": scenario.scenario_id,
            "match_no": mn,
            "round": sched["round"],
            "modal_pair_team1": pair[0] if isinstance(pair, tuple) else "",
            "modal_pair_team2": pair[1] if isinstance(pair, tuple) else "",
            "pair_probability": pair_k / args.knockout_n_sim if args.knockout_n_sim else math.nan,
            "pair_ci_low": pair_lo,
            "pair_ci_high": pair_hi,
            "modal_pair_winner": pair_winner,
            "modal_pair_winner_conditional_probability": cond,
            "modal_score_given_pair": score_mode,
            "modal_score_given_pair_probability": score_mode_prob,
            "marginal_winner": winner,
            "marginal_winner_probability": winner_k / args.knockout_n_sim if args.knockout_n_sim else math.nan,
            "marginal_winner_ci_low": win_lo,
            "marginal_winner_ci_high": win_hi,
        })
    sim.write_csv(out / f"{scenario.scenario_id}_conditional_knockout_match_forecast.csv", match_rows)

    greedy_rows = build_greedy_bracket(args, data, scenario, ko_pair_counts, ko_pair_winner_counts, score_counts)
    sim.write_csv(out / f"{scenario.scenario_id}_coherent_projected_bracket.csv", greedy_rows)
    if args.dynamic_elo:
        audit_rows: Sequence[Mapping[str, object]] = k_state.get("dynamic_elo_audit_rows", [])  # type: ignore[assignment]
        if audit_rows:
            sim.write_csv(out / f"{scenario.scenario_id}_dynamic_elo_knockout_updates_sample.csv", audit_rows)
    return team_rows, match_rows, greedy_rows


def build_greedy_bracket(
    args: argparse.Namespace,
    data: Mapping[str, object],
    scenario: GroupScenario,
    ko_pair_counts: Mapping[int, Counter],
    ko_pair_winner_counts: Mapping[Tuple[int, Tuple[str, str]], Counter],
    score_counts: Mapping[Tuple[int, Tuple[str, str]], Counter],
) -> List[dict]:
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    r32_rows: Sequence[Mapping[str, str]] = data["r32_rows"]  # type: ignore[assignment]
    bracket_rows: Sequence[Mapping[str, str]] = data["bracket_rows"]  # type: ignore[assignment]
    annex_c_assignments: Mapping[Tuple[str, ...], Mapping[int, str]] = data["annex_c_assignments"]  # type: ignore[assignment]
    third_assignment = sim.assign_thirds_to_slots(scenario.advanced_third_groups, annex_c_assignments)

    winners: Dict[int, str] = {}
    losers: Dict[int, str] = {}
    rows: List[dict] = []

    def choose(mn: int, rnd: str, team1: str, team2: str, source1: str, source2: str) -> None:
        pair = (team1, team2)
        pair_k = ko_pair_counts[mn][pair]
        pair_prob = pair_k / args.knockout_n_sim if args.knockout_n_sim else math.nan
        counts = ko_pair_winner_counts[(mn, pair)]
        if pair_k and counts:
            winner, wk = counts.most_common(1)[0]
            conditional = wk / pair_k
            mode_score, score_k = score_counts[(mn, pair)].most_common(1)[0]
            mode_score_txt = f"{mode_score[0]}-{mode_score[1]}"
            mode_score_prob = score_k / pair_k
        else:
            # Very rare fallback when the greedy pair combination never appeared.
            # Use raw Elo as deterministic display fallback and mark support_count=0.
            winner = team1 if teams[team1].elo >= teams[team2].elo else team2
            conditional = math.nan
            mode_score_txt = ""
            mode_score_prob = math.nan
        loser = team2 if winner == team1 else team1
        winners[mn] = winner
        losers[mn] = loser
        rows.append({
            "scenario_id": scenario.scenario_id,
            "match_no": mn,
            "round": rnd,
            "source1": source1,
            "source2": source2,
            "team1": team1,
            "team2": team2,
            "projected_winner": winner,
            "projected_loser": loser,
            "pair_probability_within_scenario": pair_prob,
            "support_count": pair_k,
            "winner_conditional_probability_given_pair": conditional,
            "modal_score_given_pair": mode_score_txt,
            "modal_score_given_pair_probability": mode_score_prob,
        })

    for r in r32_rows:
        mn = int(r["match_no"])
        team1 = sim.resolve_slot(r["side1"], scenario.qualifiers, third_assignment, mn)
        team2 = sim.resolve_slot(r["side2"], scenario.qualifiers, third_assignment, mn)
        source2 = f"3{third_assignment[mn]}" if r["side2"] == "3rd" else r["side2"]
        choose(mn, "R32", team1, team2, r["side1"], source2)

    for r in bracket_rows:
        mn = int(r["match_no"])
        def deref(x: str) -> str:
            if x.startswith("W"):
                return winners[int(x[1:])]
            if x.startswith("L"):
                return losers[int(x[1:])]
            raise ValueError(f"Unexpected bracket reference: {x}")
        choose(mn, r["round"], deref(r["side1"]), deref(r["side2"]), r["side1"], r["side2"])
    return rows


def write_weighted_outputs(args: argparse.Namespace, data: Mapping[str, object], selected: Sequence[GroupScenario], all_team_rows: Sequence[Sequence[Mapping[str, object]]]) -> None:
    teams: Mapping[str, sim.Team] = data["teams"]  # type: ignore[assignment]
    stages = ["reach_r32", "reach_r16", "reach_qf", "reach_sf", "reach_final", "runner_up", "third_place", "champion"]
    combined: Dict[str, Dict[str, float]] = {name: {stage: 0.0 for stage in stages} for name in teams}
    selected_mass = (math.nan if args.scenario_selection == "consensus" else sum(s.probability for s in selected if not math.isnan(float(s.probability))))
    scenario_champ_rows = []
    for scenario, team_rows in zip(selected, all_team_rows):
        by_team = {str(r["team"]): r for r in team_rows}
        top = sorted(team_rows, key=lambda r: -float(r["champion"]))[:12]
        for i, r in enumerate(top, start=1):
            scenario_champ_rows.append({
                "scenario_id": scenario.scenario_id,
                "scenario_group_probability": scenario.probability,
                "scenario_selected_weight_normalized": scenario.selected_weight,
                "rank_within_scenario": i,
                "team": r["team"],
                "champion_conditional_probability": r["champion"],
                "weighted_contribution_within_selected": scenario.selected_weight * float(r["champion"]),
            })
        for name in teams:
            r = by_team[name]
            for stage in stages:
                combined[name][stage] += scenario.selected_weight * float(r[stage])

    rows = []
    for name, vals in sorted(combined.items(), key=lambda kv: -kv[1]["champion"]):
        t = teams[name]
        row = {
            "team": name,
            "group": t.group,
            "elo": t.elo,
            "selected_scenario_mass": selected_mass,
            "weighting": ("fixed_consensus_scenario" if args.scenario_selection == "consensus" else "normalized_within_selected_topk"),
        }
        row.update(vals)
        rows.append(row)
    sim.write_csv(args.out_dir / "weighted_stage_probabilities_topk.csv", rows)
    sim.write_csv(args.out_dir / "scenario_champion_probabilities.csv", scenario_champ_rows)


def write_summary_md(args: argparse.Namespace, selected: Sequence[GroupScenario], weighted_rows_path: Path) -> None:
    rows = sim.read_csv(weighted_rows_path)
    top = sorted(rows, key=lambda r: -float(r["champion"]))[:12]
    selected_mass = sum(s.probability for s in selected if not math.isnan(float(s.probability)))
    lines: List[str] = []
    lines.append("# Two-stage coherent scenario forecast")
    lines.append("")
    lines.append("This report first simulates the group stage, selects coherent group-stage scenarios, and then re-simulates knockout tournaments conditional on those selected scenarios.")
    lines.append("")
    lines.append("## Run configuration")
    lines.append(f"- group_n_sim: {args.group_n_sim}")
    lines.append(f"- knockout_n_sim per scenario: {args.knockout_n_sim}")
    lines.append(f"- scenario_topk: {args.scenario_topk}")
    if args.scenario_selection == "consensus":
        lines.append("- selected scenario mass: n/a; constructed consensus scenario")
    else:
        lines.append(f"- selected scenario mass: {pct(selected_mass)}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- teams_file: {args.teams_file}")
    lines.append(f"- uncertainty_profile: {args.uncertainty_profile}")
    lines.append(f"- host_advantage: {args.host_advantage} ({args.host_scope})")
    lines.append(f"- contextual_effects: {args.contextual_effects}, match_context_file={args.match_context_file}")
    lines.append(f"- elo_shrink: {args.elo_shrink}")
    if args.dynamic_elo:
        lines.append(
            "- dynamic Elo update: ON "
            f"(group K={args.dynamic_elo_group_k}, R32={args.dynamic_elo_r32_k}, "
            f"R16={args.dynamic_elo_r16_k}, QF={args.dynamic_elo_qf_k}, "
            f"SF={args.dynamic_elo_sf_k}, Final={args.dynamic_elo_final_k}; "
            f"scale={args.dynamic_elo_scale}, knockout_draw_value={args.dynamic_elo_knockout_draw_value})"
        )
    else:
        lines.append("- dynamic Elo update: OFF; Elo is fixed during simulated World Cup matches")
    lines.append(f"- match_rating_sd: {args.match_rating_sd}")
    lines.append(f"- upset_prob: {args.upset_prob}")
    lines.append("")
    lines.append("## Selected group-stage scenarios")
    for s in selected:
        if math.isnan(float(s.probability)):
            prob_text = "constructed scenario; no exact sampled-scenario probability"
        else:
            prob_text = f"group probability {pct(s.probability)} [{pct(s.ci_low)}, {pct(s.ci_high)}]"
        lines.append(f"- {s.scenario_id} ({s.scenario_type}): {prob_text}, scenario weight {pct(s.selected_weight)}, Annex C option {s.annex_c_option}, 3rd groups {'/'.join(s.advanced_third_groups)}")
        if s.selection_note:
            lines.append(f"  - note: {s.selection_note}")
        if args.dynamic_elo:
            lines.append(f"  - knockout initial Elo: {s.knockout_initial_elos_source}")
    lines.append("")
    lines.append("## Weighted champion probabilities within selected scenarios")
    for i, r in enumerate(top, start=1):
        lines.append(f"{i}. {r['team']}: {float(r['champion'])*100:.1f}%")
    lines.append("")
    lines.append("## Interpretation")
    if args.scenario_selection == "consensus":
        lines.append("The weighted probabilities above are conditional on the constructed consensus group-stage scenario C01. They are not an unconditional tournament forecast.")
    else:
        lines.append("The weighted probabilities above are normalized inside the selected top-K group-stage scenarios. They are therefore conditional on the selected scenario set, not a full unconditional tournament forecast unless the selected mass is close to 100%.")
    lines.append("The coherent projected bracket CSV/SVG uses a greedy conditional-modal path inside each fixed group-stage scenario, so teams such as Japan appear in Round of 32 whenever the selected group scenario contains them.")
    args.out_dir.joinpath("two_stage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(args: argparse.Namespace, selected: Sequence[GroupScenario]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": "Two-stage Elo-Poisson World Cup 2026 simulation with FIFA Annexe C lookup",
        "group_stage": {
            "n_sim": args.group_n_sim,
            "selection": "top-K coherent group-stage scenarios",
            "scenario_topk": args.scenario_topk,
            "scenario_selection": args.scenario_selection,
            "selected_scenario_mass": (None if args.scenario_selection == "consensus" else sum(s.probability for s in selected if not math.isnan(float(s.probability)))),
        },
        "knockout": {
            "n_sim_per_selected_scenario": args.knockout_n_sim,
            "conditioning": "fixed group order, fixed best-third set, fixed Annexe C Round-of-32 bracket",
            "rating_adjustments": "resampled in knockout stage; the selected group scenario fixes bracket structure, not a posterior over latent team strength",
            "dynamic_elo_note": (
                "Within each simulated knockout path, Elo is updated after every simulated match and used for later matches."
                if args.dynamic_elo else
                "Elo is fixed during simulated knockout paths."
            ),
        },
        "parameters": {
            "teams_file": str(args.teams_file),
            "seed": args.seed,
            "group_seed": args.group_seed,
            "knockout_seed": args.knockout_seed,
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
            "match_shock_dist": args.match_shock_dist,
            "match_shock_df": args.match_shock_df,
            "goal_overdispersion": args.goal_overdispersion,
            "penalty_randomness": args.penalty_randomness,
            "elo_shrink": args.elo_shrink,
            "dynamic_elo": {
                "enabled": args.dynamic_elo,
                "group_k": args.dynamic_elo_group_k,
                "k_by_round": knockout_dynamic_elo_k_by_round(args),
                "scale": args.dynamic_elo_scale,
                "host_advantage": args.dynamic_elo_home_advantage,
                "use_goal_difference_multiplier": not args.dynamic_elo_no_margin_multiplier,
                "round_delta": args.dynamic_elo_round_delta,
                "knockout_draw_value_for_advancer": args.dynamic_elo_knockout_draw_value,
                "audit_limit": args.dynamic_elo_audit_limit,
                "formula": "R_new = R_old + K * G * (W - W_e); updated ratings affect subsequent simulated matches in the same path",
            },
            "upset_prob": args.upset_prob,
            "upset_underdog_bonus": args.upset_underdog_bonus,
            "upset_shock_sd": args.upset_shock_sd,
            "host/climate/depth": {
                "confederation_bonus": args.confederation_bonus,
                "team_bonus": args.team_bonus,
                "depth_weight": args.depth_weight,
                "heat_weight": args.heat_weight,
                "humidity_weight": args.humidity_weight,
                "altitude_weight": args.altitude_weight,
                "travel_weight": args.travel_weight,
                "contextual_effects": args.contextual_effects,
                "match_context_file": args.match_context_file,
                "include_static_environment_factors": args.include_static_environment_factors,
            },
        },
        "selected_scenarios": [
            {
                "scenario_id": s.scenario_id,
                "probability": s.probability,
                "selected_weight_normalized": s.selected_weight,
                "scenario_type": s.scenario_type,
                "selection_note": s.selection_note,
                "knockout_initial_elos_source": s.knockout_initial_elos_source,
                "annex_c_option": s.annex_c_option,
                "advanced_third_groups": s.advanced_third_groups,
            }
            for s in selected
        ],
        "limitations": [
            "Top-K selected scenarios typically have limited total probability because there are many possible group-stage outcomes.",
            "Knockout conditional simulation fixes the Round-of-32 bracket. Dynamic Elo, when enabled, updates ratings along each simulated knockout path; for consensus scenarios it is not a posterior conditioned on exact group-stage scorelines because those scorelines are not fixed by the scenario.",
            "Greedy projected brackets are coherent display paths, not maximum-probability full tournament histories.",
        ],
    }
    (args.out_dir / "two_stage_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_model_inputs(args)
    group_state = run_group_stage(args, data)
    selected = select_group_scenarios(args, data, group_state)
    write_group_outputs(args, data, group_state, selected)

    all_team_rows: List[List[dict]] = []
    all_match_rows: List[dict] = []
    all_greedy_rows: List[dict] = []
    for idx, scenario in enumerate(selected, start=1):
        k_state = run_knockout_for_scenario(args, data, scenario, idx)
        team_rows, match_rows, greedy_rows = write_knockout_outputs(args, data, k_state)
        all_team_rows.append(team_rows)
        all_match_rows.extend(match_rows)
        all_greedy_rows.extend(greedy_rows)
    sim.write_csv(args.out_dir / "conditional_knockout_match_forecast_all.csv", all_match_rows)
    sim.write_csv(args.out_dir / "coherent_projected_brackets_all.csv", all_greedy_rows)
    write_weighted_outputs(args, data, selected, all_team_rows)
    write_summary_md(args, selected, args.out_dir / "weighted_stage_probabilities_topk.csv")
    write_metadata(args, selected)
    print(f"\nWrote two-stage outputs to {args.out_dir.resolve()}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    ap = argparse.ArgumentParser(description="Two-stage coherent-scenario World Cup 2026 simulator.")
    ap.add_argument("--data-dir", type=Path, default=default_data_dir)
    ap.add_argument(
        "--teams-file",
        type=Path,
        default=Path("teams.csv"),
        help="Teams CSV to use, relative to --data-dir unless absolute. Use outputs_.../elo_recent/teams_elo_updated.csv after running update_elo_from_results.py.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("outputs_two_stage"))
    ap.add_argument("--group-n-sim", type=int, default=10000, help="Number of group-stage-only simulations.")
    ap.add_argument("--knockout-n-sim", type=int, default=5000, help="Number of conditional knockout simulations per selected scenario.")
    ap.add_argument("--scenario-topk", type=int, default=10, help="Number of exact sampled group-stage scenarios to select when --scenario-selection=topk_exact.")
    ap.add_argument("--scenario-selection", choices=("consensus", "topk_exact"), default="consensus", help="consensus constructs a representative coherent scenario from marginal group modes; topk_exact uses exact sampled scenarios.")
    ap.add_argument("--write-scenario-limit", type=int, default=200, help="Number of top group-stage scenarios to write for inspection.")
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--group-seed", type=int, default=None)
    ap.add_argument("--knockout-seed", type=int, default=None)

    # Same model options as simulate_worldcup_2026.py.
    ap.add_argument("--base-mu", type=float, default=1.35)
    ap.add_argument("--goal-elo-scale", type=float, default=1600.0)
    ap.add_argument("--ko-elo-scale", type=float, default=400.0)
    ap.add_argument("--host-advantage", type=float, default=0.0)
    ap.add_argument("--uncertainty-profile", choices=tuple(sim.UNCERTAINTY_PRESETS.keys()), default="moderate")
    ap.add_argument("--team-rating-sd", type=float, default=None)
    ap.add_argument("--match-rating-sd", type=float, default=None)
    ap.add_argument("--goal-overdispersion", type=float, default=None)
    ap.add_argument("--penalty-randomness", type=float, default=None)
    ap.add_argument("--elo-shrink", type=float, default=None)
    ap.add_argument("--match-shock-dist", choices=("normal", "student_t"), default=None)
    ap.add_argument("--match-shock-df", type=float, default=None)
    ap.add_argument("--upset-prob", type=float, default=None)
    ap.add_argument("--upset-underdog-bonus", type=float, default=None)
    ap.add_argument("--upset-shock-sd", type=float, default=None)
    ap.add_argument("--upset-min-abs-delta", type=float, default=50.0)

    # Dynamic Elo update inside simulated World Cup paths. Disabled by default
    # in the Python module for backward-compatible experiments; the wrapper
    # run_two_stage_n.sh enables it by default for the current workflow.
    ap.add_argument("--dynamic-elo", action="store_true", help="Update Elo after each simulated World Cup match and use the updated rating in later matches of the same simulation path.")
    ap.add_argument("--dynamic-elo-group-k", type=float, default=40.0)
    ap.add_argument("--dynamic-elo-r32-k", type=float, default=60.0)
    ap.add_argument("--dynamic-elo-r16-k", type=float, default=60.0)
    ap.add_argument("--dynamic-elo-qf-k", type=float, default=60.0)
    ap.add_argument("--dynamic-elo-sf-k", type=float, default=60.0)
    ap.add_argument("--dynamic-elo-final-k", type=float, default=60.0)
    ap.add_argument("--dynamic-elo-third-place-k", type=float, default=40.0)
    ap.add_argument("--dynamic-elo-scale", type=float, default=400.0)
    ap.add_argument("--dynamic-elo-home-advantage", type=float, default=None, help="Home advantage used in the Elo expected-score update. Default: reuse --host-advantage.")
    ap.add_argument("--dynamic-elo-knockout-draw-value", type=float, default=0.5, help="Rating credit for the team that advances after a drawn knockout score. 0.5 gives no extra penalty-proxy credit; 1.0 treats advancement as a full win.")
    ap.add_argument("--dynamic-elo-no-margin-multiplier", action="store_true", help="Disable goal-difference multiplier G in the Elo update.")
    ap.add_argument("--dynamic-elo-round-delta", action="store_true", help="Round each Elo update to the nearest integer point.")
    ap.add_argument("--dynamic-elo-audit-limit", type=int, default=5000, help="Maximum rows to write to dynamic Elo audit sample CSVs per stage/scenario.")

    ap.add_argument("--team-factors-file", default="team_factors.csv")
    ap.add_argument("--match-context-file", default="match_context.csv")
    ap.add_argument("--contextual-effects", action="store_true", help="Apply venue/date heat, humidity, altitude and travel effects from --match-context-file.")
    ap.add_argument("--include-static-environment-factors", action="store_true", help="Also apply environmental team factors as static bonuses when --contextual-effects is enabled; normally off to avoid double counting.")
    ap.add_argument("--host-scope", choices=("tournament", "venue", "hybrid", "none"), default="tournament", help="How --host-advantage is applied.")
    ap.add_argument("--confederation-bonus", action="append", default=[])
    ap.add_argument("--team-bonus", action="append", default=[])
    ap.add_argument("--depth-weight", type=float, default=0.0)
    ap.add_argument("--heat-weight", type=float, default=0.0)
    ap.add_argument("--humidity-weight", type=float, default=0.0)
    ap.add_argument("--altitude-weight", type=float, default=0.0)
    ap.add_argument("--travel-weight", type=float, default=0.0)
    ap.add_argument("--upset-potential-weight", type=float, default=0.0)
    ap.add_argument("--upset-resilience-weight", type=float, default=0.0)
    ap.add_argument("--fallback-ranking", choices=("elo", "fifa_rank", "name"), default="elo")
    ap.add_argument("--ci-level", type=float, default=0.95)

    return normalise_args(ap.parse_args(argv))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
