# ============================================================================
# cases/stochastic_lq.jl
#
# Stochastic LQ family: Gaussian LQ games with a closed-form reference
# (SFNELQ) and chance-constrained variants whose affine-Gaussian constraint
# probabilities are known exactly. The metrics here are independent of any
# solver's own bookkeeping: they simulate the returned feedback policy under
# the stated noise and compare with what the solver reports.
#
# Reference: Başar & Olsder (1999), Ch. 6; Zhong, Shimizu & Chen, RA-L 2023.
# ============================================================================

const STOCHASTIC_LQ_CHANCE_RISK = 0.1          # per-constraint risk (two rows → ε/2 on the bound row)
const STOCHASTIC_LQ_CHANCE_BOUND = 0.6

function _stochastic_lq_two_player(dt::Float64, N::Int)
    A = Matrix(1.0I, 2, 2)
    B = [reshape([dt, 0.0], 2, 1), reshape([0.0, dt], 2, 1)]
    Q1 = [4.0 0.0; 0.0 0.0]; Q2 = [0.0 0.0; 0.0 4.0]
    return DifferentialGamesBase.LQGameProblem(
        A, B, [Q1, Q2], [fill(0.1, 1, 1), fill(0.1, 1, 1)], [Q1, Q2], [0.0, 0.0], N * dt;
        dt = dt, q = [-(Q1 * [1.0, 0.0]), -(Q2 * [0.0, 1.0])])
end

function _build_stochastic_lq_additive()
    g = _stochastic_lq_two_player(0.1, 10)
    return DifferentialGamesBase.StochasticGameProblem(
        g, DifferentialGamesBase.ProcessNoiseModel(Matrix(0.01I, 2, 2));
        initial_covariance = Matrix(0.02I, 2, 2))
end

function _build_stochastic_lq_chance(; joint::Bool = false)
    g = _stochastic_lq_two_player(0.1, 10)
    cc = DifferentialGamesBase.ChanceConstraint(
        DifferentialGamesBase.StateBounds(1; state_offset = 0, state_dim = 1,
            lower = [-5.0], upper = [STOCHASTIC_LQ_CHANCE_BOUND]),
        STOCHASTIC_LQ_CHANCE_RISK)
    return DifferentialGamesBase.StochasticGameProblem(
        g, DifferentialGamesBase.ProcessNoiseModel(Matrix(0.01I, 2, 2));
        chance_constraints = [cc],
        joint_risk = joint ? DifferentialGamesBase.JointRiskBudget(STOCHASTIC_LQ_CHANCE_RISK) : nothing)
end

const STOCHASTIC_LQ_ADDITIVE_V1 = BenchmarkCase{Float64}(
    CaseID(:StochasticLQ, :AdditiveNoise, v"1"), _build_stochastic_lq_additive, FullModelAccess(),
    "Two-player Gaussian LQ game with additive process noise and an uncertain initial state. " *
    "SFNELQ is exact (certainty equivalence): reported expected costs can be checked by simulation.",
    ["Başar, T. & Olsder, G.J. (1999). Dynamic Noncooperative Game Theory. SIAM. (Ch. 6)"])
register_case!(STOCHASTIC_LQ_ADDITIVE_V1)

const CHANCE_CONSTRAINED_LQ_BOUND_V1 = BenchmarkCase{Float64}(
    CaseID(:ChanceConstrainedLQ, :PositionBound, v"1"), () -> _build_stochastic_lq_chance(), FullModelAccess(),
    "Stochastic two-player LQ game where player 1 must keep x₁ ≤ $(STOCHASTIC_LQ_CHANCE_BOUND) with " *
    "per-event probability ≥ 1-ε (affine ⇒ the Gaussian quantile is exact). Per-event risk does not " *
    "bound the trajectory-wide failure probability.",
    ["Zhong, H., Shimizu, Y. & Chen, J. (2023). Chance-Constrained Iterative Linear-Quadratic Stochastic Games. RA-L 8(1)."])
register_case!(CHANCE_CONSTRAINED_LQ_BOUND_V1)

const CHANCE_CONSTRAINED_LQ_JOINT_V1 = BenchmarkCase{Float64}(
    CaseID(:ChanceConstrainedLQ, :JointBudget, v"1"), () -> _build_stochastic_lq_chance(joint = true), FullModelAccess(),
    "Same game with a trajectory-wide JointRiskBudget: the union bound allocates the budget over every " *
    "(constraint, step, row) event, so the any-violation probability is bounded by the budget.",
    ["Zhong, H., Shimizu, Y. & Chen, J. (2023). Chance-Constrained Iterative Linear-Quadratic Stochastic Games. RA-L 8(1)."])
register_case!(CHANCE_CONSTRAINED_LQ_JOINT_V1)

# ─── independent simulation metrics ─────────────────────────────────────────

"""
    stochastic_policy_rollouts(sgame, sol; samples=20000, seed=0) -> NamedTuple

Simulate `sol.strategy` (perfect state information) under the game's own noise and
initial covariance. Returns `(costs = players × samples, constraint_values = events × samples)`
where events are the chance-constrained rows at stages `k = 1…N` (empty if none).
Own-control costs only.
"""
function stochastic_policy_rollouts(sg::DifferentialGamesBase.StochasticGameProblem, sol::DifferentialGamesBase.GNEPSolution;
                                    samples::Int = 20000, seed::Int = 0)
    g = sg.nominal_game
    np = DifferentialGamesBase.num_players(g); N = DifferentialGamesBase.n_steps(g)
    sd = DifferentialGamesBase.discretize_stochastic(sg.stochastic_dynamics, g.time_horizon.dt)
    cd = g.metadata.control_dims; offs = [0; cumsum(cd)[1:end-1]]
    ccs = DifferentialGamesBase.all_chance_constraints(sg)
    rng = Random.MersenneTwister(seed)
    L0 = DifferentialGamesBase._psd_factor(Matrix(sg.initial_covariance))
    n_events = sum(DifferentialGamesBase.constraint_output_dim(cc.constraint) for cc in ccs; init = 0) * N
    costs = zeros(np, samples); G = zeros(n_events, samples)
    for s in 1:samples
        x = g.initial_state + L0 * randn(rng, size(L0, 2)); row = 0
        for k in 1:N
            u = DifferentialGamesBase.apply_strategy(sol.strategy, x, k)
            for i in 1:np
                obj = DifferentialGamesBase.get_objective(g, i)
                costs[i, s] += obj.scaling * DifferentialGamesBase.evaluate_stage_cost(
                    obj.stage_cost, x, u[offs[i]+1:offs[i]+cd[i]], nothing, k)
            end
            for cc in ccs
                c = DifferentialGamesBase.evaluate_constraint(cc.constraint, x, u, nothing, k)
                G[row+1:row+length(c), s] = c; row += length(c)
            end
            x = DifferentialGamesBase.sample_step(rng, sd, x, u, nothing,
                    sg.stochastic_dynamics.drift isa DifferentialGamesBase.LinearDynamics ? k : (k - 1) * g.time_horizon.dt)
        end
        for i in 1:np
            obj = DifferentialGamesBase.get_objective(g, i)
            costs[i, s] += obj.scaling * DifferentialGamesBase.evaluate_terminal_cost(obj.terminal_cost, x, nothing)
        end
    end
    return (costs = costs, constraint_values = G)
end

"""
    stochastic_expected_cost_error(sgame, sol; samples, seed) -> Vector

Standardised discrepancy `(mean simulated cost - reported expected cost) / standard error`
per player; |z| ≲ 4 means the reported expected cost is consistent with simulation.
"""
function stochastic_expected_cost_error(sg, sol; samples::Int = 20000, seed::Int = 0)
    r = stochastic_policy_rollouts(sg, sol; samples, seed)
    np = size(r.costs, 1)
    reported = sol.solver_info[:expected_costs]
    return [(sum(r.costs[i, :]) / samples - reported[i]) /
            (sqrt(sum((r.costs[i, :] .- sum(r.costs[i, :]) / samples) .^ 2) / (samples - 1)) / sqrt(samples)) for i in 1:np]
end

"""
    chance_violation_rates(sgame, sol; samples, seed) -> NamedTuple

Simulated per-event and any-violation frequencies of the chance-constrained rows
(`DifferentialGamesBase.empirical_violation_rates`), independent of the solver's own Monte Carlo.
"""
function chance_violation_rates(sg, sol; samples::Int = 20000, seed::Int = 0)
    r = stochastic_policy_rollouts(sg, sol; samples, seed)
    return DifferentialGamesBase.empirical_violation_rates(r.constraint_values)
end
