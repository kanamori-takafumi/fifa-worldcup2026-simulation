# Two-stage coherent scenario forecast

This report first simulates the group stage, selects coherent group-stage scenarios, and then re-simulates knockout tournaments conditional on those selected scenarios.

## Run configuration
- group_n_sim: 100000
- knockout_n_sim per scenario: 100000
- scenario_topk: 10
- selected scenario mass: n/a; constructed consensus scenario
- seed: 682
- teams_file: outputs/latest/elo_recent/teams_elo_updated.csv
- uncertainty_profile: moderate
- host_advantage: 100.0 (venue)
- contextual_effects: True, match_context_file=match_context.csv
- elo_shrink: 0.82
- dynamic Elo update: ON (group K=40.0, R32=60.0, R16=60.0, QF=60.0, SF=60.0, Final=60.0; scale=400.0, knockout_draw_value=0.5)
- match_rating_sd: 150.0
- upset_prob: 0.1

## Selected group-stage scenarios
- C01 (consensus_marginal): constructed scenario; no exact sampled-scenario probability, scenario weight 100.0%, Annex C option 37, 3rd groups C/D/E/F/G/I/K/L
  - note: Constructed from each group's modal order and the eight groups with the largest third-place qualification probability; not an exact sampled scenario.
  - knockout initial Elo: unconditional_group_terminal_mean_n=100000

## Weighted champion probabilities within selected scenarios
1. Spain: 9.6%
2. Argentina: 8.5%
3. Colombia: 6.3%
4. France: 6.3%
5. Brazil: 5.9%
6. Ecuador: 4.5%
7. England: 4.2%
8. Mexico: 4.0%
9. Japan: 3.3%
10. Portugal: 3.2%
11. Turkey: 3.0%
12. Morocco: 2.9%

## Interpretation
The weighted probabilities above are conditional on the constructed consensus group-stage scenario C01. They are not an unconditional tournament forecast.
The coherent projected bracket CSV/SVG uses a greedy conditional-modal path inside each fixed group-stage scenario, so teams such as Japan appear in Round of 32 whenever the selected group scenario contains them.
