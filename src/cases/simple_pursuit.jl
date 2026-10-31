# ============================================================================
# cases/simple_pursuit.jl
#
# First official Classical Games case. Built entirely from existing Base
# types: single-integrator player dynamics, `maintain_proximity`/
# `avoid_proximity` cost terms (reused, not reinvented), no hard capture
# constraint — capture is assessed post-hoc by the evaluator, since
# `ProximityConstraint` enforces a *minimum* separation (wrong direction for
# capture, which wants small distance).
#
# Solver compatibility: this game is built via the PD-GNEP path
# (SeparableDynamics). `ALGAMES` declares `:SeparableDynamics` support and is
# the mathematically compatible solver; `iLQGames` does NOT declare
# `:SeparableDynamics` and errors on this game's per-player dynamics assembly
# despite otherwise looking like a match on paper — always check
# `solver_capabilities`, not just "unconstrained nonlinear."
# ============================================================================

"""
    SIMPLE_PURSUIT_CAPTURE_RADIUS

Distance below which the pursuer is considered to have captured the evader.
Used both in the evader's `avoid_proximity` cost term and in the evaluator's
capture metrics, so the modeled incentive and the measured outcome agree.
"""
const SIMPLE_PURSUIT_CAPTURE_RADIUS = 0.5

function _build_simple_pursuit_nominal()
    n, m = 2, 2  # planar position state, planar velocity control
    x0_pursuer = zeros(2)
    x0_evader  = [3.0, 3.0]

    R = Matrix(2.0 * I, m, m)

    pursuer_stage = DifferentialGamesBase.maintain_proximity(; i_offset = 0, j_offset = 2, pos_dim = 2,
                                                                d_max = 0.3, weight = 1.0) +
                    DifferentialGamesBase.regularize_input(R; control_offset = 0, control_dim = m)
    pursuer_obj = DifferentialGamesBase.minimize(pursuer_stage; player_id = 1)

    evader_stage = DifferentialGamesBase.avoid_proximity(; i_offset = 0, j_offset = 2, pos_dim = 2,
                                                             d_min = SIMPLE_PURSUIT_CAPTURE_RADIUS, weight = 1.0) +
                   DifferentialGamesBase.regularize_input(R; control_offset = 2, control_dim = m)
    evader_obj = DifferentialGamesBase.minimize(evader_stage; player_id = 2)

    # Single-integrator kinematics: ẋᵢ = uᵢ (continuous-time RHS; solvers
    # discretize internally per DifferentialGamesBaseSolvers' RK4 convention).
    pursuer = DifferentialGamesBase.Player{Float64}(1, n, m, x0_pursuer, (x, u, p, t) -> u, pursuer_obj, [])
    evader  = DifferentialGamesBase.Player{Float64}(2, n, m, x0_evader,  (x, u, p, t) -> u, evader_obj,  [])

    return DifferentialGamesBase.DifferentialGame([pursuer, evader], 5.0, 0.1)
end

const SIMPLE_PURSUIT_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:SimplePursuit, :Nominal, v"1"),
    _build_simple_pursuit_nominal,
    FullModelAccess(),
    "Two-player pursuit-evasion, single-integrator kinematics, unconstrained. " *
    "Capture is assessed post-hoc by the evaluator rather than enforced as a " *
    "hard constraint.",
    ["Isaacs, R. (1965). Differential Games. Wiley."],
)
register_case!(SIMPLE_PURSUIT_NOMINAL_V1)

"""
    simple_pursuit_evaluator() -> Evaluator

`default_evaluator()` extended with capture-specific quality metrics.
"""
simple_pursuit_evaluator() = capture_evaluator(SIMPLE_PURSUIT_CAPTURE_RADIUS)

simple_pursuit_visual_model(game, sol) =
    two_agent_visual_model(game, sol, ("Pursuer", "Evader"), SIMPLE_PURSUIT_CAPTURE_RADIUS)

register_visualizer!(SIMPLE_PURSUIT_NOMINAL_V1.id, simple_pursuit_visual_model)
