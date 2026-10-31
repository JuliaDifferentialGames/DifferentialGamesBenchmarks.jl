# ============================================================================
# cases/homicidal_chauffeur.jl
#
# Isaacs' Homicidal Chauffeur: a nonholonomic, speed-w pursuer with bounded
# turning rate (hence a minimum turning radius R = w/φ_max) versus an
# omnidirectional, speed-bounded evader. The dynamics ASYMMETRY between
# players is the point of this case (contrast with Game of Two Cars, where
# both players share the same dynamics class).
#
# Built entirely from existing Base machinery: per-player nonlinear kinematics
# via `PlayerSpec.dynamics`, `maintain_proximity`/`avoid_proximity` cost terms,
# and `control_bounds` private constraints (needed here — unlike Simple
# Pursuit, an unconstrained pursuer has no turning-radius limit at all, so
# the constraint isn't optional set-dressing, it's the defining assumption).
#
# Solver: ALGAMES (`:SeparableDynamics` + `:ConstrainedGame`, both required
# here — the PD-GNEP path plus the turning-rate/speed bounds).
# ============================================================================

const HOMICIDAL_CHAUFFEUR_CAPTURE_RADIUS = 0.5
const _HC_PURSUER_SPEED = 2.0
const _HC_PURSUER_TURN_RATE_MAX = 1.0   # turning radius = speed / turn_rate_max = 2.0
const _HC_EVADER_SPEED_MAX = 1.0

function _build_homicidal_chauffeur_nominal()
    x0_pursuer = [0.0, 0.0, 0.0]  # x, y, heading (rad); starts pointing along +x
    x0_evader  = [3.0, 3.0]

    Rp = Matrix(2.0 * I, 1, 1)
    Re = Matrix(2.0 * I, 2, 2)

    pursuer_stage = DifferentialGamesBase.maintain_proximity(; i_offset = 0, j_offset = 3, pos_dim = 2,
                                                                d_max = 0.3, weight = 1.0) +
                    DifferentialGamesBase.regularize_input(Rp; control_offset = 0, control_dim = 1)
    pursuer_obj = DifferentialGamesBase.minimize(pursuer_stage; player_id = 1)

    evader_stage = DifferentialGamesBase.avoid_proximity(; i_offset = 0, j_offset = 3, pos_dim = 2,
                                                             d_min = HOMICIDAL_CHAUFFEUR_CAPTURE_RADIUS, weight = 1.0) +
                   DifferentialGamesBase.regularize_input(Re; control_offset = 1, control_dim = 2)
    evader_obj = DifferentialGamesBase.minimize(evader_stage; player_id = 2)

    # Joint control vector is [φ (pursuer, 1) ; vx, vy (evader, 2)].
    pursuer_turn_bound = DifferentialGamesBase.control_bounds(
        1; control_offset = 0, control_dim = 1,
        lower = [-_HC_PURSUER_TURN_RATE_MAX], upper = [_HC_PURSUER_TURN_RATE_MAX],
    )
    evader_speed_bound = DifferentialGamesBase.control_bounds(
        2; control_offset = 1, control_dim = 2,
        lower = fill(-_HC_EVADER_SPEED_MAX, 2), upper = fill(_HC_EVADER_SPEED_MAX, 2),
    )

    # Pursuer: nonholonomic car at fixed speed w, steering via turn rate φ.
    pursuer_dynamics(x, u, p, t) = [_HC_PURSUER_SPEED * cos(x[3]), _HC_PURSUER_SPEED * sin(x[3]), u[1]]
    # Evader: omnidirectional point, velocity-controlled.
    evader_dynamics(x, u, p, t) = u

    pursuer = DifferentialGamesBase.Player{Float64}(
        1, 3, 1, x0_pursuer, pursuer_dynamics, pursuer_obj, [pursuer_turn_bound],
    )
    evader = DifferentialGamesBase.Player{Float64}(
        2, 2, 2, x0_evader, evader_dynamics, evader_obj, [evader_speed_bound],
    )

    return DifferentialGamesBase.DifferentialGame([pursuer, evader], 5.0, 0.1)
end

const HOMICIDAL_CHAUFFEUR_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:HomicidalChauffeur, :Nominal, v"1"),
    _build_homicidal_chauffeur_nominal,
    FullModelAccess(),
    "Nonholonomic, bounded-turning-rate pursuer (speed $_HC_PURSUER_SPEED, turning " *
    "radius $(_HC_PURSUER_SPEED / _HC_PURSUER_TURN_RATE_MAX)) versus an omnidirectional, " *
    "speed-bounded evader. Capture assessed post-hoc by the evaluator.",
    ["Isaacs, R. (1965). Differential Games. Wiley. (Ch. 8, \"The Homicidal Chauffeur\")"],
)
register_case!(HOMICIDAL_CHAUFFEUR_NOMINAL_V1)

"""
    homicidal_chauffeur_evaluator() -> Evaluator
"""
homicidal_chauffeur_evaluator() = capture_evaluator(HOMICIDAL_CHAUFFEUR_CAPTURE_RADIUS)

function homicidal_chauffeur_visual_model(game, sol)
    spec = two_agent_visual_model(game, sol, ("Pursuer (car)", "Evader"), HOMICIDAL_CHAUFFEUR_CAPTURE_RADIUS)
    # Attach heading so the renderer can draw the pursuer's orientation.
    fg = _forward_game(game)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    heading = Xm[3, :]
    agents = [spec.agents[1], spec.agents[2]]
    pursuer_with_heading = merge(agents[1], (heading = heading,))
    return merge(spec, (agents = [pursuer_with_heading, agents[2]],))
end

register_visualizer!(HOMICIDAL_CHAUFFEUR_NOMINAL_V1.id, homicidal_chauffeur_visual_model)
