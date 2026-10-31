# ============================================================================
# evaluation/evaluator.jl
#
# Independent evaluation: metrics are recomputed from Base trajectories,
# objectives, and constraints — never trusted from solver_info/converged.
# ============================================================================

"""
    Evaluator

Bundles independent metric functions, grouped by the evaluation hierarchy
(validity, quality — including task-specific metrics, robustness, efficiency).

# Fields
- `validity_checks::Vector{Function}`      : `(game, sol) -> Bool`
- `quality_metrics::Dict{Symbol,Function}` : `name => (game, sol) -> Real`
- `robustness_metrics::Dict{Symbol,Function}` : `name => (game, sol) -> Real`
- `efficiency_metrics::Dict{Symbol,Function}` : `name => (sol,) -> Dict{Symbol,Any}`
  (efficiency reads solver-reported data directly; nothing to recompute against)
"""
struct Evaluator
    validity_checks::Vector{Function}
    quality_metrics::Dict{Symbol,Function}
    robustness_metrics::Dict{Symbol,Function}
    efficiency_metrics::Dict{Symbol,Function}
end

# ============================================================================
# Forward-game unwrapping — metric functions operate on the concrete
# GameProblem at the bottom of any wrapper chain (Stochastic/Robust/PO/
# Convex/Lexicographic all wrap one).
# ============================================================================

_forward_game(g::DifferentialGamesBase.GameProblem) = g
_forward_game(g::DifferentialGamesBase.StochasticGameProblem) = _forward_game(g.nominal_game)
_forward_game(g::DifferentialGamesBase.RobustGameProblem) = _forward_game(g.nominal_game)
_forward_game(g::DifferentialGamesBase.POGameProblem) = _forward_game(g.underlying_game)
_forward_game(g::DifferentialGamesBase.ConvexGameProblem) = _forward_game(g.forward_game)
_forward_game(g::DifferentialGamesBase.LexicographicGameProblem) = _forward_game(g.forward_game)

# ============================================================================
# Joint state/control reconstruction — works for both shared-state games
# (state_trajectory present) and PD-GNEPs (per-player private states, stacked
# in player-id order to match GameMetadata's offsets).
# ============================================================================

function _joint_state_control_matrices(game, sol::DifferentialGamesBase.GNEPSolution)
    fg = _forward_game(game)
    n = DifferentialGamesBase.n_players(fg)
    trajs = [DifferentialGamesBase.get_trajectory(sol, i) for i in 1:n]
    Xm = DifferentialGamesBase.has_shared_state(sol) ? sol.state_trajectory :
         vcat([t.states for t in trajs]...)
    Um = vcat([t.controls for t in trajs]...)
    return Xm, Um, trajs[1].times
end

_columns(M) = [M[:, k] for k in axes(M, 2)]

# ============================================================================
# Metric functions
# ============================================================================

"""
    validity_constraint_satisfaction(game, sol; tol=1e-3) -> Bool

Recomputes every private/shared constraint over the joint (state, control) at
every time step directly from `game`'s constraint list and `sol`'s
trajectories — never trusts solver-reported feasibility. Default tolerance
matches ALGAMES's own `tol_con` default (1e-3): tight enough to catch a
genuinely broken solution, loose enough not to fail on ordinary numerical
solver slop in a converged constrained NLP solve.
"""
function validity_constraint_satisfaction(game, sol::DifferentialGamesBase.GNEPSolution; tol::Real = 1e-3)
    fg = _forward_game(game)
    all_constraints = vcat(fg.private_constraints, fg.shared_constraints)
    isempty(all_constraints) && return true
    Xm, Um, _ = _joint_state_control_matrices(game, sol)
    N = size(Um, 2)
    max_violation = maximum(
        DifferentialGamesBase.constraint_violation(c, Xm[:, k], Um[:, k], nothing, k)
        for c in all_constraints for k in 1:N
    )
    return max_violation <= tol
end

"""
    game_quality_nash_residual(game, sol) -> Dict{Int,Float64}

For each player, recomputes total cost from the joint trajectory via
`total_cost` and compares against `get_cost(sol, player_id)`. A large residual
means the solver misreported its own objective value, independent of whether
it claims to have converged.

Argument convention differs by game type and this recomputation must match
whichever one the game's own cost type expects: `LQStageCost` (pure LQ games,
e.g. `LQGameProblem`) sizes `R` to the player's own control dimension and
expects `u` pre-sliced to that player; PD-GNEP nonlinear cost terms
(`ProximityCost`/`QuadraticControlCost` via `player_slice`) instead expect
the full joint `u` and slice internally via stored offsets. `is_lq_game`
distinguishes the two.
"""
function game_quality_nash_residual(game, sol::DifferentialGamesBase.GNEPSolution)
    fg = _forward_game(game)
    Xm, Um, _ = _joint_state_control_matrices(game, sol)
    X = _columns(Xm)
    lq = DifferentialGamesBase.is_lq_game(fg)
    Dict(
        obj.player_id => begin
            U = if lq
                off = fg.metadata.control_offsets[obj.player_id]
                dim = fg.metadata.control_dims[obj.player_id]
                _columns(Um[off+1:off+dim, :])
            else
                _columns(Um)
            end
            abs(
                DifferentialGamesBase.total_cost(obj, X, U, nothing) -
                DifferentialGamesBase.get_cost(sol, obj.player_id)
            )
        end
        for obj in fg.objectives
    )
end

"""
    robustness_metric_dynamics_residual(game, sol) -> Float64

Re-rolls out the joint dynamics from the reported initial state and joint
controls, and returns the maximum absolute deviation from the reported
trajectory — catches a solver that "converges" to a dynamically infeasible
trajectory.
"""
function robustness_metric_dynamics_residual(game, sol::DifferentialGamesBase.GNEPSolution)
    fg = _forward_game(game)
    Xm, Um, times = _joint_state_control_matrices(game, sol)
    x0 = Xm[:, 1]
    Xsim = DifferentialGamesBase.rollout(fg.dynamics, x0, Um, nothing, times)
    return maximum(abs.(Xsim .- Xm))
end

"""
    efficiency_metrics_from_solver_info(sol) -> Dict{Symbol,Any}

Efficiency is the one metric class allowed to read solver-reported data
(`sol.iterations`, `sol.solve_time`, `sol.solver_info`) — wall-clock time
cannot be independently recomputed after the fact. `Experiment`'s externally
measured wall time (see `orchestration/experiment.jl`) is the authoritative
cross-check, not this function's output alone.
"""
function efficiency_metrics_from_solver_info(sol::DifferentialGamesBase.GNEPSolution)
    Dict{Symbol,Any}(:iterations => sol.iterations, :solve_time => sol.solve_time)
end

"""
    default_evaluator() -> Evaluator

The generic evaluator: constraint validity, Nash-residual quality, dynamics
robustness, and solver-reported efficiency. Case-specific evaluators (e.g.
Simple Pursuit's capture metrics) extend `quality_metrics` with additional
entries rather than replacing this one.
"""
default_evaluator() = Evaluator(
    Function[validity_constraint_satisfaction],
    Dict{Symbol,Function}(:nash_residual => game_quality_nash_residual),
    Dict{Symbol,Function}(:dynamics_residual => robustness_metric_dynamics_residual),
    Dict{Symbol,Function}(:solver_info => efficiency_metrics_from_solver_info),
)

"""
    evaluate(evaluator::Evaluator, game, sol) -> NamedTuple

Run every metric in `evaluator` against `(game, sol)` and return
`(validity, quality, robustness, efficiency)` dicts. `run(::Experiment)`
wraps this into a full `ExperimentResult` with wall-time and provenance.
"""
function evaluate(evaluator::Evaluator, game, sol::DifferentialGamesBase.GNEPSolution)
    validity = Dict{Symbol,Bool}(
        :constraints_satisfied => all(f(game, sol) for f in evaluator.validity_checks)
    )
    quality = Dict{Symbol,Any}(name => f(game, sol) for (name, f) in evaluator.quality_metrics)
    robustness = Dict{Symbol,Any}(name => f(game, sol) for (name, f) in evaluator.robustness_metrics)
    efficiency = Dict{Symbol,Any}()
    for (_, f) in evaluator.efficiency_metrics
        merge!(efficiency, f(sol))
    end
    return (validity = validity, quality = quality, robustness = robustness, efficiency = efficiency)
end
