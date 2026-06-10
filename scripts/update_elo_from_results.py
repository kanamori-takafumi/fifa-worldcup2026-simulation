#!/usr/bin/env python3
"""Build recent match-by-match Elo ratings and a World Cup teams file.

This script is deliberately dependency-free.  It reads a match results CSV with
columns compatible with martj42/international_results::results.csv:

    date,home_team,away_team,home_score,away_score,tournament,city,country,neutral

It then replays matches in chronological order, updating Elo after every match,
and writes:

    teams_elo_updated.csv       teams.csv copy with the elo column replaced
    team_elo_history.csv        one row per relevant team per match
    elo_match_updates.csv       one row per replayed match
    team_elo_summary.csv        start/end/change/forecast Elo by team
    elo_recent_trends.svg       dependency-free SVG trend chart
    elo_metadata.json           reproducibility metadata

The update formula follows the common World Football Elo form

    R_new = R_old + K * G * (W - W_e)

with configurable K factors, goal-difference multiplier G, and a home-advantage
term used only in W_e.  This is not an official FIFA rating; it is a transparent,
editable input layer for the tournament simulator.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import unicodedata
from collections import defaultdict
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ELO_TABLE_URL_TEMPLATE = (
    "https://www.international-football.net/elo-ratings-table"
    "?confed=&day={day:02d}&month={month:02d}&old-team=&year={year:04d}"
)

REQUIRED_RESULTS_COLUMNS = {"date", "home_team", "away_team", "home_score", "away_score"}

CONTINENTAL_FINAL_TOKENS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "africa cup of nations",
    "afc asian cup",
    "concacaf gold cup",
    "ofc nations cup",
    "conmebol-uefa",
    "conmebol–uefa",
    "finalissima",
)

QUALIFIER_TOKENS = (
    "qualification",
    "qualifier",
    "qualifiers",
)

OTHER_MAJOR_TOURNAMENT_TOKENS = (
    "uefa nations league",
    "concacaf nations league",
    "copa centroamericana",
    "copa centroamericana",
    "gulf cup",
    "baltic cup",
    "kings cup",
    "king's cup",
    "kirin cup",
    "kirin challenge cup",
    "eaﬀ championship",
    "eaff championship",
)


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def strip_accents(text: str) -> str:
    decomp = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomp if not unicodedata.combining(ch))


def norm_name(name: str) -> str:
    text = strip_accents(str(name)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    candidate = base_dir / p
    if candidate.exists():
        return candidate
    return p


def load_alias_map(path: Optional[Path], project_teams: Iterable[str]) -> Dict[str, str]:
    aliases = {norm_name(team): team for team in project_teams}
    if path is None or not path.exists():
        return aliases
    for row in read_csv(path):
        source = str(row.get("source_name", "")).strip()
        team = str(row.get("team", "")).strip()
        if not source or not team:
            continue
        aliases[norm_name(source)] = team
    return aliases


def canonical_team(raw: str, aliases: Mapping[str, str]) -> str:
    cleaned = str(raw).strip()
    return aliases.get(norm_name(cleaned), cleaned)


def load_team_rows(path: Path) -> Tuple[List[dict], List[str], Dict[str, float]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No teams found in {path}")
    if "team" not in rows[0] or "elo" not in rows[0]:
        raise ValueError(f"{path} must contain team and elo columns")
    teams: List[str] = []
    ratings: Dict[str, float] = {}
    for row in rows:
        team = str(row["team"]).strip()
        if not team:
            raise ValueError(f"Blank team name in {path}")
        if team in ratings:
            raise ValueError(f"Duplicate team in {path}: {team}")
        ratings[team] = float(row["elo"])
        teams.append(team)
    return rows, teams, ratings


def read_text_path_or_url(path_or_url: str, timeout: float = 45.0) -> str:
    if re.match(r"^https?://", path_or_url):
        req = Request(path_or_url, headers={"User-Agent": "wc2026-elo-updater/1.0"})
        with urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    return Path(path_or_url).read_text(encoding="utf-8")


def strip_html_to_text(raw_html: str) -> str:
    text = re.sub(r"</(tr|td|li|div|p|h[1-6]|br)>", "\n", raw_html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def collapse_duplicate_words(name: str) -> str:
    parts = name.split()
    if len(parts) % 2 == 0 and parts:
        mid = len(parts) // 2
        if parts[:mid] == parts[mid:]:
            return " ".join(parts[:mid])
    return name


def parse_international_football_elo_table(raw_html: str, aliases: Mapping[str, str]) -> Dict[str, float]:
    """Parse international-football.net Elo table into a rating dictionary.

    The page is HTML, not a formal CSV API. The parser therefore first strips
    markup and then looks for repeated rank/name/rating patterns such as
    "1. Spain 2155 2. Argentina 2113". For World Cup teams, the alias map
    canonicalises common source-name variants.
    """
    text = strip_html_to_text(raw_html)
    if "World football Elo ratings as on" in text:
        section = text.split("World football Elo ratings as on", 1)[1]
    elif "Elo ratings table" in text:
        section = text.split("Elo ratings table", 1)[-1]
    else:
        section = text
    collapsed = re.sub(r"\s+", " ", section)
    pattern = re.compile(
        r"(?:^|\s)(\d{1,3})\.\s*(?:Image:\s*[^0-9]+?\s+)?"
        r"([A-Za-zÀ-ÖØ-öø-ÿ .&'()\-/]+?)\s+([0-9]{3,4})"
        r"(?=\s+\d{1,3}\.\s|\s+\*\s+|\s*$)"
    )
    ratings: Dict[str, float] = {}
    for m in pattern.finditer(collapsed):
        raw_name = collapse_duplicate_words(m.group(2).strip())
        team = canonical_team(raw_name, aliases)
        ratings[team] = float(m.group(3))
    return ratings


def load_initial_ratings_from_source(
    args: argparse.Namespace,
    start_date: dt.date,
    aliases: Mapping[str, str],
    focus_teams: Sequence[str],
    input_focus_ratings: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """Return (ratings_all, focus_ratings, metadata).

    ratings_all may contain non-tournament teams when an external ratings table
    is supplied. focus_ratings always contains one value for each team in
    teams.csv.
    """
    source = args.initial_elo_source
    meta = {"initial_elo_source": source}
    if source == "teams":
        ratings_all = dict(input_focus_ratings)
        meta["initial_elo_note"] = "Baseline ratings taken from --teams-file."
        return ratings_all, dict(input_focus_ratings), meta

    if source == "csv":
        if not args.initial_ratings_file:
            raise ValueError("--initial-ratings-file is required when --initial-elo-source csv")
        path = resolve_path(args.data_dir, args.initial_ratings_file)
        rows = read_csv(path)
        if not rows or "team" not in rows[0] or "elo" not in rows[0]:
            raise ValueError(f"{path} must contain team and elo columns")
        ratings_all = {canonical_team(str(r["team"]), aliases): float(r["elo"]) for r in rows if str(r.get("team", "")).strip()}
        meta["initial_ratings_file"] = str(path)
    elif source == "international-football":
        initial_date = parse_date(args.initial_elo_date) if args.initial_elo_date else start_date
        url = args.initial_elo_url_template.format(year=initial_date.year, month=initial_date.month, day=initial_date.day)
        raw_html = read_text_path_or_url(url)
        ratings_all = parse_international_football_elo_table(raw_html, aliases)
        meta.update({
            "initial_elo_date": initial_date.isoformat(),
            "initial_elo_url": url,
            "initial_elo_parsed_count": len(ratings_all),
        })
    else:
        raise ValueError("--initial-elo-source must be teams, csv, or international-football")

    focus_ratings: Dict[str, float] = {}
    missing: List[str] = []
    for team in focus_teams:
        if team in ratings_all:
            focus_ratings[team] = ratings_all[team]
        else:
            missing.append(team)
            if args.fallback_to_input_elo:
                focus_ratings[team] = float(input_focus_ratings[team])
                ratings_all[team] = focus_ratings[team]
    if missing and not args.fallback_to_input_elo:
        raise ValueError("Initial rating source is missing tournament teams: " + ", ".join(missing))
    meta["initial_missing_focus_teams"] = missing
    meta["fallback_to_input_elo"] = bool(missing and args.fallback_to_input_elo)
    return ratings_all, focus_ratings, meta


def expected_result(delta: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-delta / scale))


def result_points(score_for: int, score_against: int) -> float:
    if score_for > score_against:
        return 1.0
    if score_for == score_against:
        return 0.5
    return 0.0


def goal_difference_multiplier(abs_goal_diff: int) -> float:
    if abs_goal_diff <= 1:
        return 1.0
    if abs_goal_diff == 2:
        return 1.5
    return (11.0 + abs_goal_diff) / 8.0


def k_factor(tournament: str, args: argparse.Namespace) -> float:
    t = str(tournament or "").strip().casefold()
    if "world cup" in t and not any(token in t for token in QUALIFIER_TOKENS):
        return args.k_world_cup
    if any(token in t for token in CONTINENTAL_FINAL_TOKENS) and not any(token in t for token in QUALIFIER_TOKENS):
        return args.k_continental_final
    if any(token in t for token in QUALIFIER_TOKENS):
        return args.k_qualifier
    if "friendly" in t:
        return args.k_friendly
    if any(token in t for token in OTHER_MAJOR_TOURNAMENT_TOKENS):
        return args.k_other_tournament
    return args.k_other_tournament


def safe_int_score(value: object, row: Mapping[str, object]) -> int:
    try:
        return int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Invalid score {value!r} in row {row}") from exc


def validate_results_columns(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in results file {path}")
    cols = set(rows[0].keys())
    missing = sorted(REQUIRED_RESULTS_COLUMNS - cols)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def date_window(rows: Sequence[Mapping[str, object]], args: argparse.Namespace) -> Tuple[dt.date, dt.date]:
    dates = [parse_date(str(r["date"])) for r in rows if str(r.get("date", "")).strip()]
    if not dates:
        raise ValueError("No valid dates in results file")
    end = parse_date(args.end_date) if args.end_date else max(dates)
    if args.start_date:
        start = parse_date(args.start_date)
    else:
        start = end - dt.timedelta(days=args.lookback_days)
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")
    return start, end


def maybe_round_delta(delta: float, enabled: bool) -> float:
    if not enabled:
        return delta
    # Python round uses bankers' rounding; Elo descriptions usually mean nearest integer.
    # Use half-away-from-zero for deterministic rating-point updates.
    if delta >= 0:
        return math.floor(delta + 0.5)
    return -math.floor(abs(delta) + 0.5)


def replay_matches(args: argparse.Namespace) -> dict:
    teams_path = resolve_path(args.data_dir, args.teams_file)
    results_path = resolve_path(Path.cwd(), args.results_file)
    aliases_path = None if args.aliases_file is None else resolve_path(args.data_dir, args.aliases_file)

    team_rows, focus_teams, input_focus_ratings = load_team_rows(teams_path)
    focus_set = set(focus_teams)
    aliases = load_alias_map(aliases_path, focus_teams)
    raw_results = read_csv(results_path)
    validate_results_columns(raw_results, results_path)
    start_date, end_date = date_window(raw_results, args)

    # Ratings for tournament teams can start from teams.csv, a supplied CSV, or
    # an international-football.net table at the beginning of the lookback window.
    # Opponents outside the tournament field are retained when the external table
    # contains them; otherwise they are initialized lazily at --unknown-initial-elo.
    ratings, initial_focus_ratings, initial_source_metadata = load_initial_ratings_from_source(
        args, start_date, aliases, focus_teams, input_focus_ratings
    )
    initial_all_ratings: Dict[str, float] = dict(ratings)
    non_focus_teams = set()

    match_updates: List[dict] = []
    team_history: List[dict] = []
    team_points: Dict[str, List[Tuple[dt.date, float]]] = {team: [(start_date, initial_focus_ratings[team])] for team in focus_teams}

    skipped_outside_window = 0
    skipped_no_focus_team = 0
    skipped_bad_row = 0
    replayed = 0

    sortable_rows = []
    for idx, row in enumerate(raw_results):
        try:
            date = parse_date(str(row["date"]))
            if date < start_date or date > end_date:
                skipped_outside_window += 1
                continue
            sortable_rows.append((date, idx, row))
        except Exception:
            skipped_bad_row += 1
            if args.strict:
                raise

    sortable_rows.sort(key=lambda x: (x[0], x[1]))

    for date, input_order, row in sortable_rows:
        try:
            home_raw = str(row["home_team"]).strip()
            away_raw = str(row["away_team"]).strip()
            home = canonical_team(home_raw, aliases)
            away = canonical_team(away_raw, aliases)
            if not home or not away or home == away:
                skipped_bad_row += 1
                continue
            has_focus = home in focus_set or away in focus_set
            if args.focus_teams_only and not has_focus:
                skipped_no_focus_team += 1
                continue
            hs = safe_int_score(row["home_score"], row)
            aw = safe_int_score(row["away_score"], row)
            tournament = str(row.get("tournament", "") or "")
            neutral = parse_bool(row.get("neutral", "FALSE"))
        except Exception:
            skipped_bad_row += 1
            if args.strict:
                raise
            continue

        for team in (home, away):
            if team not in ratings:
                ratings[team] = args.unknown_initial_elo
                initial_all_ratings[team] = args.unknown_initial_elo
                non_focus_teams.add(team)

        old_home = ratings[home]
        old_away = ratings[away]
        home_adv = 0.0 if neutral else args.home_advantage_elo
        exp_home = expected_result((old_home + home_adv) - old_away, args.elo_scale)
        exp_away = 1.0 - exp_home
        res_home = result_points(hs, aw)
        res_away = 1.0 - res_home
        k = k_factor(tournament, args)
        g = goal_difference_multiplier(abs(hs - aw))
        delta_home_raw = k * g * (res_home - exp_home)
        delta_home = maybe_round_delta(delta_home_raw, args.round_delta)
        delta_away = -delta_home

        ratings[home] = old_home + delta_home
        ratings[away] = old_away + delta_away
        replayed += 1

        match_update = {
            "match_index": replayed,
            "input_order": input_order,
            "date": date.isoformat(),
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": aw,
            "tournament": tournament,
            "neutral": int(neutral),
            "k_factor": round(k, 6),
            "goal_diff_multiplier": round(g, 6),
            "home_elo_before": round(old_home, 6),
            "away_elo_before": round(old_away, 6),
            "home_expected": round(exp_home, 6),
            "away_expected": round(exp_away, 6),
            "home_result": res_home,
            "away_result": res_away,
            "home_delta_raw": round(delta_home_raw, 6),
            "home_delta": round(delta_home, 6),
            "away_delta": round(delta_away, 6),
            "home_elo_after": round(ratings[home], 6),
            "away_elo_after": round(ratings[away], 6),
            "has_focus_team": int(has_focus),
        }
        match_updates.append(match_update)

        for side, team, opponent, gf, ga, old, new, exp, res, delta in (
            ("home", home, away, hs, aw, old_home, ratings[home], exp_home, res_home, delta_home),
            ("away", away, home, aw, hs, old_away, ratings[away], exp_away, res_away, delta_away),
        ):
            if team not in focus_set:
                continue
            team_history.append({
                "match_index": replayed,
                "date": date.isoformat(),
                "team": team,
                "opponent": opponent,
                "side": side,
                "goals_for": gf,
                "goals_against": ga,
                "result": res,
                "expected_result": round(exp, 6),
                "tournament": tournament,
                "neutral": int(neutral),
                "k_factor": round(k, 6),
                "goal_diff_multiplier": round(g, 6),
                "elo_before": round(old, 6),
                "elo_delta": round(delta, 6),
                "elo_after": round(new, 6),
            })
            team_points[team].append((date, new))

    if replayed == 0 and not args.allow_empty:
        raise ValueError(
            "No matches were replayed in the selected date window. "
            "Check --start-date/--end-date, --lookback-days, team aliases, and whether the results CSV is current enough. "
            "Use --allow-empty only when you intentionally want a no-op Elo preprocessing run."
        )

    summary_rows: List[dict] = []
    updated_team_rows: List[dict] = []
    by_team_history = defaultdict(list)
    for row in team_history:
        by_team_history[str(row["team"])].append(row)

    for row in team_rows:
        team = str(row["team"]).strip()
        start_elo = initial_focus_ratings[team]
        end_elo = ratings.get(team, start_elo)
        change = end_elo - start_elo
        forecast_adjustment = args.trend_weight * change
        forecast_elo = end_elo + forecast_adjustment
        hist = by_team_history.get(team, [])
        elo_values = [start_elo] + [float(h["elo_after"]) for h in hist]
        last_match_date = hist[-1]["date"] if hist else ""
        row_out = dict(row)
        row_out["elo"] = f"{forecast_elo:.3f}"
        row_out["elo_recent_start"] = f"{start_elo:.3f}"
        row_out["elo_recent_end"] = f"{end_elo:.3f}"
        row_out["elo_recent_change"] = f"{change:.3f}"
        row_out["elo_recent_trend_weight"] = f"{args.trend_weight:.3f}"
        row_out["elo_recent_forecast_adjustment"] = f"{forecast_adjustment:.3f}"
        row_out["elo_recent_matches"] = str(len(hist))
        row_out["elo_recent_last_match_date"] = last_match_date
        updated_team_rows.append(row_out)

        summary_rows.append({
            "team": team,
            "group": row.get("group", ""),
            "position": row.get("position", ""),
            "confederation": row.get("confederation", ""),
            "matches": len(hist),
            "start_elo": round(start_elo, 3),
            "end_elo": round(end_elo, 3),
            "change": round(change, 3),
            "trend_weight": round(args.trend_weight, 3),
            "forecast_adjustment": round(forecast_adjustment, 3),
            "forecast_elo": round(forecast_elo, 3),
            "min_elo": round(min(elo_values), 3),
            "max_elo": round(max(elo_values), 3),
            "last_match_date": last_match_date,
        })

    summary_rows.sort(key=lambda r: (-float(r["forecast_elo"]), str(r["team"])))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "elo_match_updates.csv", match_updates)
    write_csv(out_dir / "team_elo_history.csv", team_history)
    write_csv(out_dir / "team_elo_summary.csv", summary_rows)
    write_csv(out_dir / "teams_elo_updated.csv", updated_team_rows)
    write_trend_svg(out_dir / "elo_recent_trends.svg", team_points, summary_rows, start_date, end_date, args.plot_top_n)

    metadata = {
        "teams_file": str(teams_path),
        "results_file": str(results_path),
        "aliases_file": str(aliases_path) if aliases_path else None,
        **initial_source_metadata,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "lookback_days": args.lookback_days,
        "focus_teams_only": args.focus_teams_only,
        "focus_team_count": len(focus_teams),
        "replayed_match_count": replayed,
        "team_history_rows": len(team_history),
        "non_focus_opponents_initialized": len(non_focus_teams),
        "unknown_initial_elo": args.unknown_initial_elo,
        "skipped": {
            "outside_window": skipped_outside_window,
            "no_focus_team": skipped_no_focus_team,
            "bad_row": skipped_bad_row,
        },
        "elo_formula": {
            "rating_update": "R_new = R_old + K * G * (W - W_e)",
            "expected_result": "W_e = 1 / (1 + 10^(-delta/elo_scale)); non-neutral home delta adds home_advantage_elo to the home team only for W_e",
            "elo_scale": args.elo_scale,
            "home_advantage_elo": args.home_advantage_elo,
            "round_delta": args.round_delta,
            "goal_difference_multiplier": "1 for draw/one-goal margin, 1.5 for two-goal margin, (11+N)/8 for N>=3",
            "k_factors": {
                "world_cup": args.k_world_cup,
                "continental_final": args.k_continental_final,
                "qualifier": args.k_qualifier,
                "other_tournament": args.k_other_tournament,
                "friendly": args.k_friendly,
            },
        },
        "trend_weight": args.trend_weight,
        "outputs": {
            "teams_file_for_simulation": str(out_dir / "teams_elo_updated.csv"),
            "team_elo_summary": str(out_dir / "team_elo_summary.csv"),
            "team_elo_history": str(out_dir / "team_elo_history.csv"),
            "elo_match_updates": str(out_dir / "elo_match_updates.csv"),
            "trend_svg": str(out_dir / "elo_recent_trends.svg"),
        },
        "limitations": [
            "For exact public-rating reconstruction, use --initial-elo-source international-football or a complete --initial-ratings-file at the start date.",
            "If --initial-elo-source teams is used with current ratings, interpret the result as a recent-form stress test rather than a historical reconstruction.",
            "Non-tournament opponents not present in teams.csv are initialized at --unknown-initial-elo unless supplied in the input teams universe.",
        ],
    }
    (out_dir / "elo_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": out_dir,
        "start_date": start_date,
        "end_date": end_date,
        "replayed": replayed,
        "summary_rows": summary_rows,
    }


def write_trend_svg(
    path: Path,
    team_points: Mapping[str, Sequence[Tuple[dt.date, float]]],
    summary_rows: Sequence[Mapping[str, object]],
    start_date: dt.date,
    end_date: dt.date,
    top_n: int,
) -> None:
    width, height = 1180, 760
    left, right, top, bottom = 90, 260, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    ranked = sorted(summary_rows, key=lambda r: -abs(float(r["change"])))[:max(1, top_n)]
    teams = [str(r["team"]) for r in ranked]
    points = {team: list(team_points.get(team, [])) for team in teams}
    all_elos = [elo for pts in points.values() for _, elo in pts]
    if not all_elos:
        all_elos = [1500.0]
    y_min = math.floor((min(all_elos) - 20.0) / 50.0) * 50.0
    y_max = math.ceil((max(all_elos) + 20.0) / 50.0) * 50.0
    if y_max <= y_min:
        y_max = y_min + 100.0
    day_span = max(1, (end_date - start_date).days)

    def x(date: dt.date) -> float:
        return left + ((date - start_date).days / day_span) * plot_w

    def y(elo: float) -> float:
        return top + (y_max - elo) / (y_max - y_min) * plot_h

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#003f5c", "#ffa600",
    ]
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append('<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#111}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.note{fill:#555;font-size:13px}.label{font-size:14px}.title{font-size:24px;font-weight:700}</style>')
    lines.append(f'<text class="title" x="{left}" y="38">Recent match-by-match Elo trend</text>')
    lines.append(f'<text class="note" x="{left}" y="58">{html.escape(start_date.isoformat())} to {html.escape(end_date.isoformat())}; top {len(teams)} teams by absolute Elo change</text>')
    # horizontal grid
    step = 50.0
    grid_val = y_min
    while grid_val <= y_max + 1e-9:
        yy = y(grid_val)
        lines.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        lines.append(f'<text class="note" x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end">{grid_val:.0f}</text>')
        grid_val += step
    lines.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    lines.append(f'<text class="note" x="{left}" y="{height - 35}">{html.escape(start_date.isoformat())}</text>')
    lines.append(f'<text class="note" x="{left + plot_w}" y="{height - 35}" text-anchor="end">{html.escape(end_date.isoformat())}</text>')

    for i, team in enumerate(teams):
        pts = points.get(team) or []
        if not pts:
            continue
        colour = palette[i % len(palette)]
        # Collapse consecutive entries on the same date by keeping the last rating of that date.
        by_date: Dict[dt.date, float] = {}
        for d, rating in pts:
            by_date[d] = rating
        coords = [(d, by_date[d]) for d in sorted(by_date)]
        if len(coords) == 1:
            coords = [(start_date, coords[0][1]), (end_date, coords[0][1])]
        path_data = " ".join(("M" if j == 0 else "L") + f" {x(d):.1f} {y(r):.1f}" for j, (d, r) in enumerate(coords))
        lines.append(f'<path d="{path_data}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        lx = left + plot_w + 22
        ly = top + 24 + i * 28
        change = next(float(r["change"]) for r in summary_rows if str(r["team"]) == team)
        end_elo = next(float(r["end_elo"]) for r in summary_rows if str(r["team"]) == team)
        lines.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly - 5}" y2="{ly - 5}" stroke="{colour}" stroke-width="3"/>')
        lines.append(f'<text class="label" x="{lx + 30}" y="{ly}">{html.escape(team)}  {end_elo:.0f} ({change:+.0f})</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Update World Cup 2026 team Elo ratings from recent match results.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Project data directory. Default: data/.")
    ap.add_argument("--teams-file", default="teams.csv", help="Base teams CSV. Relative paths are resolved under --data-dir when present.")
    ap.add_argument("--results-file", required=True, help="Match results CSV with date/home_team/away_team/home_score/away_score/tournament/neutral columns.")
    ap.add_argument("--aliases-file", default="team_name_aliases.csv", help="Optional source_name,team alias CSV under --data-dir.")
    ap.add_argument(
        "--initial-elo-source",
        choices=("teams", "csv", "international-football"),
        default="teams",
        help="Baseline ratings at the start of the replay window. Default: teams.",
    )
    ap.add_argument("--initial-ratings-file", default="", help="CSV with team,elo used when --initial-elo-source csv.")
    ap.add_argument("--initial-elo-date", default="", help="Date for international-football baseline. Default: start-date.")
    ap.add_argument("--initial-elo-url-template", default=DEFAULT_ELO_TABLE_URL_TEMPLATE)
    ap.add_argument("--fallback-to-input-elo", action=argparse.BooleanOptionalAction, default=True, help="Fallback to --teams-file Elo when baseline source misses a tournament team. Default: true.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs_elo_recent"), help="Output directory for Elo update artifacts.")
    ap.add_argument("--start-date", default="", help="Start date YYYY-MM-DD. Default: end-date minus --lookback-days.")
    ap.add_argument("--end-date", default="", help="End date YYYY-MM-DD. Default: latest date in results file.")
    ap.add_argument("--lookback-days", type=int, default=365, help="Used when --start-date is omitted. Default: 365.")
    ap.add_argument("--focus-teams-only", action=argparse.BooleanOptionalAction, default=True, help="Replay only matches with at least one team from teams.csv. Default: true.")
    ap.add_argument("--unknown-initial-elo", type=float, default=1500.0, help="Initial Elo for opponents absent from teams.csv. Default: 1500.")
    ap.add_argument("--elo-scale", type=float, default=400.0, help="Logistic scale in W_e. Default: 400.")
    ap.add_argument("--home-advantage-elo", type=float, default=100.0, help="Non-neutral home rating boost for W_e only. Default: 100.")
    ap.add_argument("--round-delta", action=argparse.BooleanOptionalAction, default=True, help="Round each match Elo point change to nearest integer. Default: true.")
    ap.add_argument("--trend-weight", type=float, default=0.0, help="Momentum adjustment: forecast_elo=end_elo+trend_weight*(end_elo-start_elo). Default: 0.")
    ap.add_argument("--plot-top-n", type=int, default=12, help="Number of trend lines in SVG. Default: 12.")
    ap.add_argument("--k-world-cup", type=float, default=60.0)
    ap.add_argument("--k-continental-final", type=float, default=50.0)
    ap.add_argument("--k-qualifier", type=float, default=40.0)
    ap.add_argument("--k-other-tournament", type=float, default=30.0)
    ap.add_argument("--k-friendly", type=float, default=20.0)
    ap.add_argument("--strict", action="store_true", help="Raise on malformed rows instead of skipping them.")
    args = ap.parse_args(argv)

    if args.lookback_days <= 0:
        raise ValueError("--lookback-days must be positive")
    if args.elo_scale <= 0:
        raise ValueError("--elo-scale must be positive")
    if args.plot_top_n <= 0:
        raise ValueError("--plot-top-n must be positive")
    for name in ("k_world_cup", "k_continental_final", "k_qualifier", "k_other_tournament", "k_friendly"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    state = replay_matches(args)
    top = state["summary_rows"][:8]
    print(f"[elo] replayed matches: {state['replayed']}")
    print(f"[elo] window: {state['start_date']} to {state['end_date']}")
    print(f"[elo] wrote: {state['out_dir'] / 'teams_elo_updated.csv'}")
    print("[elo] top forecast ratings:")
    for i, row in enumerate(top, start=1):
        print(f"  {i:>2}. {row['team']}: {float(row['forecast_elo']):.1f} ({float(row['change']):+.1f} over window, {row['matches']} matches)")


if __name__ == "__main__":
    main()
