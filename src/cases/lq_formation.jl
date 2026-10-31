# ============================================================================
# cases/lq_formation.jl
#
# First case of the LQ Games family (Phase 2, per the design doc: "LQ games
# follow the Classical pack as the second scientific family — analytic/
# reference backbone"). Unlike the Classical Games (nonlinear, PD-GNEP,
# solved approximately by ALGAMES), this is a genuine linear-quadratic Nash
# game built via `LQGameProblem` and solved *exactly* by FNELQ's coupled
# Riccati recursion — the point of this family is to give the ecosystem a
# case where the solver's own claim to correctness can be checked against a
# real closed-form structure, not just internal consistency.
#
# Two players, each an independent discrete-time 2D double integrator
# (uncoupled DYNAMICS — A is block-diagonal, B_i only drives player i's own
# block), coupled through COST asymmetrically: player 1 ("follower") is
# penalized for its relative-position error from player 2 ("leader"), who is
# penalized only for its own station-keeping (hold position, minimize
# control). This leader/follower asymmetry is deliberate — an earlier
# symmetric version (both players penalized for the *same* shared
# relative-offset error) is a textbook case of Nash under-provision in a
# public-good-like game: since both players' costs reward the same joint
# outcome but each bears only their own control cost, neither has full
# unilateral incentive to close the gap, and FNELQ's Nash equilibrium settled
# well short of the target — a real phenomenon, not a bug, but not a useful
# demo either.
#
# # ⚠ Known bug found while building this case: FNELQ mishandles affine (q/r)
# cost terms
#
# `LQStageCost`/`LQTerminalCost` support an affine term (`q`/`r`, e.g. to
# track a nonzero target `Cx = d` via `Q = w·CᵀC`, `q = -w·Cᵀd`) and
# `evaluate_stage_cost`/`total_cost` apply it with the textbook sign
# (verified directly against `evaluate_stage_cost`: the unconstrained minimum
# of `0.5x'Qx + q'x` really is at `Cx=d`). But `FNELQ`'s backward Riccati
# recursion (`src/solvers/FNELQ/src/fnelq.jl`, the `has_affine` branch, in
# particular `Yα[ri] = -(Bi' * ζ[i] + rk[i])`) converges to the *mirror image*
# of the intended target whenever `q`/`r` are nonzero — reproduced by solving
# a target-offset formation game and an independent from-scratch backward
# Riccati recursion side by side: same magnitude, opposite sign, every time.
# This code path is apparently untested (every other FNELQ-solved case in the
# existing test suite has `q=r=0`, which skips it entirely). Filed as a
# BaseSolvers bug to fix separately rather than patched here; this case
# sidesteps it by using a **zero** relative-position target (a rendezvous,
# not an offset formation) so `q` is identically zero and the buggy code path
# is never exercised.
#
# Modeling note: `LQStageCost`/`LQTerminalCost` also assert `Q`/`Qf` are PSD.
# A zero-sum-flavored pursuit cost (evader wants to *maximize* separation,
# i.e. a negative-definite term in Q) is therefore NOT expressible through
# `LQGameProblem` as it exists today.
# ============================================================================

const LQ_FORMATION_OFFSET = [0.0, 0.0]  # player 1 - player 2 target position offset (rendezvous)

function _lq_double_integrator_2d(dt::Float64)
    A2d = [1.0 0.0 dt 0.0;
           0.0 1.0 0.0 dt;
           0.0 0.0 1.0 0.0;
           0.0 0.0 0.0 1.0]
    B2d = [0.5*dt^2 0.0;
           0.0      0.5*dt^2;
           dt       0.0;
           0.0      dt]
    return A2d, B2d
end

# Selects (p1 - p2) from the joint state [p1x,p1y,v1x,v1y,p2x,p2y,v2x,v2y].
const _LQ_FORMATION_C = [1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0;
                         0.0 1.0 0.0 0.0 0.0 -1.0 0.0 0.0]

function _lq_formation_QqC(weight::Float64, d::Vector{Float64}, vel_weight::Float64)
    C = _LQ_FORMATION_C
    Q = weight * (C' * C)
    Q[3, 3] += vel_weight; Q[4, 4] += vel_weight
    Q[7, 7] += vel_weight; Q[8, 8] += vel_weight
    q = -weight * (C' * d)
    return Q, q
end

# Player 2 ("leader"): simple station-keeping — hold position (via a small
# velocity-damping cost) and minimize control. No dependence on player 1.
function _lq_station_keeping_Qq(vel_weight::Float64)
    Q = zeros(8, 8)
    Q[7, 7] = vel_weight; Q[8, 8] = vel_weight
    return Q, zeros(8)
end

function _build_lq_formation_nominal()
    dt, tf = 0.1, 5.0
    A2d, B2d = _lq_double_integrator_2d(dt)
    A = [A2d zeros(4, 4); zeros(4, 4) A2d]
    B1 = [B2d; zeros(4, 2)]
    B2 = [zeros(4, 2); B2d]

    d = LQ_FORMATION_OFFSET
    Q1, q1 = _lq_formation_QqC(20.0, d, 0.1)
    Q2, q2 = _lq_station_keeping_Qq(0.1)
    R = 0.05 * Matrix(I, 2, 2)

    x0 = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0]  # player 1 at origin, player 2 at (5,5)

    Qf1, _ = _lq_formation_QqC(3.0, d, 1.0)
    Qf2, _ = _lq_station_keeping_Qq(1.0)

    return DifferentialGamesBase.LQGameProblem(
        A, [B1, B2], [Q1, Q2], [R, R], [Qf1, Qf2], x0, tf;
        dt = dt, q = [q1, q2],
    )
end

const LQ_FORMATION_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:LQFormation, :Nominal, v"1"),
    _build_lq_formation_nominal,
    FullModelAccess(),
    "Two-player linear-quadratic leader/follower rendezvous game: independent double-integrator " *
    "dynamics; player 1 (follower) tracks player 2's position (player 2 only station-keeps). " *
    "Solved exactly by FNELQ's coupled Riccati recursion. Analytic/reference case for the LQ " *
    "Games family.",
    ["Başar, T. & Olsder, G.J. (1999). Dynamic Noncooperative Game Theory. SIAM. (Ch. 6, LQ Nash games)"],
)
register_case!(LQ_FORMATION_NOMINAL_V1)

"""
    lq_formation_error(game, sol) -> Float64

Independent recomputation of `‖(p1 - p2) - FORMATION_OFFSET‖` at the final
time, from the joint state trajectory — the closed-loop formation-keeping
error the game is designed to drive to zero.
"""
function lq_formation_error(game, sol::DifferentialGamesBase.GNEPSolution)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    rel = Xm[1:2, end] .- Xm[5:6, end]
    return norm(rel .- LQ_FORMATION_OFFSET)
end

"""
    lq_formation_evaluator() -> Evaluator

`default_evaluator()` extended with the formation-keeping error metric.
"""
function lq_formation_evaluator()
    ev = default_evaluator()
    ev.quality_metrics[:formation_error_final] = lq_formation_error
    return ev
end

function lq_formation_visual_model(game, sol::DifferentialGamesBase.GNEPSolution)
    Xm, _, times = _joint_state_control_matrices(game, sol)
    pos1, pos2 = Xm[1:2, :], Xm[5:6, :]
    return (
        agents = [
            (player_id = 1, label = "Player 1 (follower)", positions = pos1),
            (player_id = 2, label = "Player 2 (leader)", positions = pos2),
        ],
        times = times,
    )
end

register_visualizer!(LQ_FORMATION_NOMINAL_V1.id, lq_formation_visual_model)
