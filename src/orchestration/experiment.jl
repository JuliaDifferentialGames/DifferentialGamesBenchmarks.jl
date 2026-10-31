# ============================================================================
# orchestration/experiment.jl — Experiment: Case x Challenge x Access x
# Method x Budget x Evaluator. Benchmark: a named collection of Experiments.
#
# `run` extends `Base.run` (rather than shadowing it) so `using
# DifferentialGamesBenchmarks` doesn't produce an export conflict with
# `Base.run`. A `DifferentialGamesBase.GameSolver` goes through the existing
# Base `solve` path directly — no mandatory wrapper class, per the ecosystem
# invariant that adding a benchmark never requires solver changes.
# ============================================================================

import Base: run

"""
    Experiment

# Fields
- `case::BenchmarkCase`
- `challenge::ChallengeProfile`
- `access::AccessSpec`
- `solver::DifferentialGamesBase.GameSolver`
- `budget::Budget`
- `evaluator::Evaluator`
"""
struct Experiment
    case::BenchmarkCase
    challenge::ChallengeProfile
    access::AccessSpec
    solver::DifferentialGamesBase.GameSolver
    budget::Budget
    evaluator::Evaluator
end

Experiment(
    case::BenchmarkCase, solver::DifferentialGamesBase.GameSolver;
    challenge::ChallengeProfile = NOMINAL_CHALLENGE,
    access::AccessSpec = case.default_access,
    budget::Budget = Budget(),
    evaluator::Evaluator = default_evaluator(),
) = Experiment(case, challenge, access, solver, budget, evaluator)

"""
    run(experiment::Experiment) -> ExperimentResult

Build `experiment.case`'s truth game, apply the challenge transforms and
access restriction, solve it with `experiment.solver` through
`DifferentialGamesBase.solve`, and hand the result to `experiment.evaluator`.
Wall-clock time is measured here (authoritative), independent of anything the
solver self-reports.
"""
function run(experiment::Experiment)
    truth = game(experiment.case)
    challenged = apply(experiment.challenge, truth)
    solver_view = access(experiment.access, challenged)

    t0 = time()
    sol = DifferentialGamesBase.solve(
        solver_view, experiment.solver;
        verbose = false,
    )
    wall_time_s = time() - t0

    ev = evaluate(experiment.evaluator, solver_view, sol)

    return ExperimentResult(
        experiment, sol, wall_time_s,
        ev.validity, ev.quality, ev.robustness, ev.efficiency,
        Dict{Symbol,Any}(:case_id => string(experiment.case.id)),
    )
end

"""
    Benchmark

A named collection of `Experiment`s.
"""
struct Benchmark
    name::Symbol
    experiments::Vector{Experiment}
end

"""
    run(benchmark::Benchmark) -> Vector{ExperimentResult}
"""
run(benchmark::Benchmark) = [run(e) for e in benchmark.experiments]
