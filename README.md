# World Cup 2026 two-stage Elo simulation

This repository contains a reproducible two-stage Monte Carlo pipeline for a FIFA World Cup 2026 forecast-style analysis.

The model is not an official forecast. It is an Elo-Poisson simulation with explicit scenario assumptions, uncertainty shocks, contextual venue factors, dynamic Elo updates inside simulated tournaments, and FIFA Annexe C third-place assignment logic.

## Public compact report

The GitHub-ready compact report is under `docs/`.

- `docs/index.html` links to the English and Japanese compact reports.
- `docs/en/compact_report_en.html` is the English compact report.
- `docs/ja/compact_report_ja.html` is the Japanese compact report.
- Only compact HTML/SVG visual outputs are included in the public report tree.

For GitHub Pages, set **Settings → Pages → Build and deployment → Deploy from a branch → `/docs`**.

## Repository layout

```text
.
├── data/                         # Tournament input tables and model factors
├── docs/                         # Compact HTML/SVG report for publication
├── examples/                     # Small example input for Elo update testing
├── outputs/latest/               # Curated latest run snapshot, without heavy audit samples
├── scripts/                      # Simulation, Elo update, validation, compact visualizer
├── run_two_stage_n.sh            # Main wrapper
├── smoke_test.sh                 # Small reproducibility check
├── requirements.txt              # Optional PNG export dependency
└── README.md
```

Removed from the upload package: macOS metadata, Python bytecode/cache files, merge notes, duplicated quickstarts, non-compact visual reports, old full visualizer entry points, and large dynamic Elo audit sample CSVs.

## Quick start

Run a small smoke test:

```bash
bash smoke_test.sh
```

This writes a small run to `outputs_smoke_test/` and creates compact HTML/SVG under:

```text
outputs_smoke_test/visuals_compact/index.html
```

Run a larger simulation without PNG export:

```bash
bash run_two_stage_n.sh \
  --n-sim 10000 \
  --out-dir outputs_run_10000 \
  --no-png
```

Run with recent match-by-match Elo preprocessing using the bundled local results file:

```bash
bash run_two_stage_n.sh \
  --n-sim 10000 \
  --recent-elo \
  --elo-results-csv data/external/international_results.csv \
  --elo-end-date 2026-06-09 \
  --out-dir outputs_recent_elo_10000 \
  --no-png
```

The wrapper validates the Annexe C lookup table before the simulation and then writes compact reports to `<out-dir>/visuals_compact/`.

## Latest included snapshot

`outputs/latest/` is a curated snapshot of the latest 100,000-by-100,000 consensus-scenario run.

Important files:

- `outputs/latest/two_stage_summary.md`
- `outputs/latest/two_stage_metadata.json`
- `outputs/latest/weighted_stage_probabilities_topk.csv`
- `outputs/latest/group_stage_team_probabilities.csv`
- `outputs/latest/C01_conditional_team_stage_probabilities.csv`
- `outputs/latest/C01_conditional_knockout_match_forecast.csv`
- `outputs/latest/C01_coherent_projected_bracket.csv`
- `outputs/latest/elo_recent/team_elo_summary.csv`
- `outputs/latest/elo_recent/teams_elo_updated.csv`

The large dynamic Elo sample audit files are intentionally omitted from this public package. Re-run the pipeline with `--dynamic-elo-audit-limit` if those row-level samples are needed.

## Method summary

The pipeline has two stages.

1. Simulate the group stage many times and select a coherent group-stage scenario.
2. Condition on the selected group-stage scenario and re-simulate the knockout stage.

The match model combines Elo-based expected strength, Poisson score generation, configurable overdispersion, heavy-tailed match shocks, host/venue context, and underdog-shock mechanisms. Knockout draws are resolved by an Elo-logistic tie-breaker blended toward random penalty outcomes. Dynamic Elo updates can update ratings after each simulated World Cup match and use the updated ratings in later simulated matches from the same tournament path.

## Data and assumptions

Core tournament tables are stored in `data/`.

- `teams.csv`: teams, groups, Elo values, and optional ranking metadata.
- `group_stage.csv`: group-stage fixture pattern.
- `round32_slots.csv`: Round-of-32 slot definitions.
- `annex_c_third_place_assignments.csv`: FIFA Annexe C third-place lookup table.
- `match_context.csv`: venue/date heat, humidity, altitude, and travel indices.
- `team_factors.csv`: editable heuristic team-level factors.
- `data/external/international_results.csv`: local copy of recent international results used for reproducible Elo preprocessing.

Several factor files are heuristic scenario inputs, not validated measurements. For serious forecasting, replace them or perform sensitivity analysis.

## Validation

Validate the Annexe C table directly:

```bash
python3 scripts/validate_annex_c.py --data-dir data
```

The smoke test also runs this validation.
