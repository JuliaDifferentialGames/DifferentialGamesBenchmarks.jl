# DifferentialGamesBenchmarks

<p align="center"><img src="https://raw.githubusercontent.com/JuliaDifferentialGames/DifferentialGames.jl/main/docs/src/assets/logo.png" alt="DifferentialGames.jl logo" width="160"></p>

[![Build Status](https://github.com/JuliaDifferentialGames/DifferentialGamesBenchmarks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDifferentialGames/DifferentialGamesBenchmarks.jl/actions/workflows/CI.yml?query=branch%3Amain)

Benchmark cases, assumption-removal challenge transformations, independent
evaluation metrics, experiment orchestration, and Makie-based visualization
for the [JuliaDifferentialGames](https://github.com/JuliaDifferentialGames)
ecosystem. Built on top of
[DifferentialGamesBase.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl)'s
game/player/solver types — a benchmark case's `truth` is always a real Base
game, never a private copy.

## Classical Games pack

Four official two-player pursuit-evasion cases, each solvable with
[DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl)'s
`ALGAMES` (all four are built via the PD-GNEP/`SeparableDynamics` path, and
Homicidal Chauffeur/Game of Two Cars additionally need `:ConstrainedGame` for
their turning-rate bounds — `iLQGames` does not declare `:SeparableDynamics`
and will error on any of them):

| Case | `CaseID` | What's being stressed |
| --- | --- | --- |
| Simple Pursuit | `CaseID(:SimplePursuit, :Nominal, v"1")` | Minimal single-integrator pursuit-evasion |
| Homicidal Chauffeur | `CaseID(:HomicidalChauffeur, :Nominal, v"1")` | Dynamics *asymmetry* — nonholonomic, bounded-turning-radius pursuer vs. a free point evader |
| Game of Two Cars | `CaseID(:GameOfTwoCars, :Nominal, v"1")` | Dynamics *symmetry* — both players share the same bounded-turning-radius car dynamics |
| Lady in the Lake | `CaseID(:LadyInTheLake, :Nominal, v"1")` | Asymmetric *domain* — a free-swimming evader vs. a pursuer confined to a 1D angular shore |

```julia
using DifferentialGamesBenchmarks, DifferentialGamesBase, DifferentialGamesBaseSolvers, CairoMakie

case = HOMICIDAL_CHAUFFEUR_NOMINAL_V1
sol  = DifferentialGamesBase.solve(game(case), ALGAMES())
res  = evaluate(homicidal_chauffeur_evaluator(), game(case), sol)

fig = gameplot(case, sol; preset = :publication)
```

Or run the whole pipeline through `Experiment`/`run`:

```julia
experiment = Experiment(case, ALGAMES(); evaluator = homicidal_chauffeur_evaluator())
result = run(experiment)
dashboard(result)
```

See `test/` for a runnable example of every case, and
`src/cases/common.jl` for the shared two-player pursuit-evasion helpers
(`capture_evaluator`, `two_agent_visual_model`) that the first three cases
build on. Lady in the Lake needs bespoke geometry/cost functions instead —
see `src/cases/lady_in_the_lake.jl` for why.

## FALCON suite

Two hard, non-convex benchmarks ported from the validation suite of FALCON
(Outland & Arya, *A Fast, Convergent Algorithm for Solving Non-convex
Partially-Decoupled Generalized Nash Equilibrium Problems*). They are
solver-agnostic Base games; the differences from the FALCON scripts are listed
at the top of each source file.

| Case | `CaseID` | What's being stressed |
| --- | --- | --- |
| Hallway passing | `CaseID(:HallwayPassing, :Nominal, v"1")` | Three unicycles through an S-shaped double-choke corridor; single-file passage forced by chokes narrower than the safe distance |
| Racing (Austin) | `CaseID(:Racing, :AustinNominal, v"1")` | Two dynamic-bicycle cars (Pacejka tyres) racing head-to-head on a 20 m track segment; competitive terminal progress, contouring/lag cost, track-boundary and collision constraints |

```julia
using DifferentialGamesBenchmarks, DifferentialGamesBase, DifferentialGamesBaseSolvers

P  = HallwayParameters()
g  = hallway_game(P)
ws = hallway_warmstart(P)                        # the straight-line initialisation crosses the walls
sol = DifferentialGamesBase.solve(g, ALGAMES(; outer_iter = 30, inner_iter = 15); warmstart = ws)
evaluate(hallway_passing_evaluator(; params = P), g, sol).quality[:min_separation]

track = austin_track()
g2  = racing_game(track)
sol2 = DifferentialGamesBase.solve(g2, ALGAMES(; outer_iter = 20, inner_iter = 15); warmstart = racing_warmstart(track))
evaluate(racing_evaluator(; track), g2, sol2).quality[:track_clearance]
```

**Neither game is solved to convergence by a solver in this ecosystem.**
`ALGAMES` reaches an almost feasible hallway plan (slowly, and only from a good
warm start); on racing its line search returns a zero step and the result is
the warm-start rollout. The default `:nash_residual` metric is a cost-consistency
check, not an optimality test, and reports 0 for such a result. See the
documentation's *Benchmark suite* section.

## LQ Games family

The second scientific family (analytic/reference backbone): genuine
linear-quadratic Nash games built via `DifferentialGamesBase.LQGameProblem`
and solved *exactly* by `FNELQ`'s coupled Riccati recursion, rather than
approximately by ALGAMES/iLQGames.

| Case | `CaseID` | What's being stressed |
| --- | --- | --- |
| LQ Formation (rendezvous) | `CaseID(:LQFormation, :Nominal, v"1")` | Leader/follower LQ Nash game with a known closed-form solution |

```julia
using DifferentialGamesBenchmarks, DifferentialGamesBase, DifferentialGamesBaseSolvers

case = LQ_FORMATION_NOMINAL_V1
sol  = DifferentialGamesBase.solve(game(case), FNELQ())
res  = evaluate(lq_formation_evaluator(), game(case), sol)
```

**A real bug in `FNELQ` was found while building this case** and is documented in
detail at the top of `src/cases/lq_formation.jl`: its affine (`q`/`r`) cost-term
handling in the backward Riccati recursion has a sign error, causing it to
converge to the mirror image of the intended target whenever a stage/terminal
cost has a nonzero linear term. Reproduced independently against a from-scratch
Riccati recursion. This case works around it by using a zero-offset (rendezvous)
target, which never exercises the buggy code path — the underlying `FNELQ` bug
is not fixed here and should be tracked separately in
`DifferentialGamesBaseSolvers.jl`.

## FALCON suite additions

| Case / function | What it is |
| --- | --- |
| `CaseID(:Racing, :AustinHairpin, v"1")` | The first hairpin of the Austin track (30 m segment, 4 s), the sharp-corner example of the FALCON paper |
| `CaseID(:HallwayPassing, :Communication, v"1")` | The hallway with a hard communication range (convoy constraint), 30 s |
| `austin_benchmark_initial_conditions()` | The 50 initial conditions of the FALCON-vs-DG-SQP comparison and the Python DG-SQP results on them |
| `racing_game_from_ic(i)` | The Austin racing game started from the `i`-th of them |
