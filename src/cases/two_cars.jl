# ============================================================================
# cases/two_cars.jl
#
# Isaacs/Merz's "Two Cars": both players are fixed-speed, bounded-turning-
# rate vehicles with the *same* dynamics class and parameters — unlike
# Homicidal Chauffeur, where the dynamics are deliberately asymmetric
# (nonholonomic pursuer vs. free point evader). The shared structure with
# `homicidal_chauffeur.jl` is intentional: this case stresses dynamics
# SYMMETRY, not novelty of mechanism.
#
# Solver: ALGAMES (`:SeparableDynamics` + `:ConstrainedGame`, same reasoning
# as Homicidal Chauffeur — both players' turning-rate bounds are hard
# constraints here, not optional).
# ============================================================================

const GAME_OF_TWO_CARS_CAPTURE_RADIUS = 0.5
const _TC_SPEED = 2.0
const _TC_TURN_RATE_MAX = 2.0  # turning radius = speed / turn_rate_max = 1.0

function _car_dynamics(x, u, p, t)
    return [_TC_SPEED * cos(x[3]), _TC_SPEED * sin(x[3]), u[1]]
end

function _build_game_of_two_cars_nominal()
    x0_p1 = [0.0, 0.0, 0.0]              # pursuer role, heading +x
    x0_p2 = [3.0, 3.0, 3 * pi / 4]       # evader role, heading away but not antiparallel

    R = Matrix(5.0 * I, 1, 1)

    p1_stage = DifferentialGamesBase.maintain_proximity(; i_offset = 0, j_offset = 3, pos_dim = 2,
                                                           d_max = 0.3, weight = 1.0) +
               DifferentialGamesBase.regularize_input(R; control_offset = 0, control_dim = 1)
    p1_obj = DifferentialGamesBase.minimize(p1_stage; player_id = 1)

    p2_stage = DifferentialGamesBase.avoid_proximity(; i_offset = 0, j_offset = 3, pos_dim = 2,
                                                        d_min = GAME_OF_TWO_CARS_CAPTURE_RADIUS, weight = 1.0) +
               DifferentialGamesBase.regularize_input(R; control_offset = 1, control_dim = 1)
    p2_obj = DifferentialGamesBase.minimize(p2_stage; player_id = 2)

    turn_bound_1 = DifferentialGamesBase.control_bounds(
        1; control_offset = 0, control_dim = 1, lower = [-_TC_TURN_RATE_MAX], upper = [_TC_TURN_RATE_MAX],
    )
    turn_bound_2 = DifferentialGamesBase.control_bounds(
        2; control_offset = 1, control_dim = 1, lower = [-_TC_TURN_RATE_MAX], upper = [_TC_TURN_RATE_MAX],
    )

    car1 = DifferentialGamesBase.Player{Float64}(1, 3, 1, x0_p1, _car_dynamics, p1_obj, [turn_bound_1])
    car2 = DifferentialGamesBase.Player{Float64}(2, 3, 1, x0_p2, _car_dynamics, p2_obj, [turn_bound_2])

    return DifferentialGamesBase.DifferentialGame([car1, car2], 5.0, 0.1)
end

const GAME_OF_TWO_CARS_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:GameOfTwoCars, :Nominal, v"1"),
    _build_game_of_two_cars_nominal,
    FullModelAccess(),
    "Two identical fixed-speed, bounded-turning-rate vehicles (speed $_TC_SPEED, " *
    "turning radius $(_TC_SPEED / _TC_TURN_RATE_MAX) each). Same dynamics class for " *
    "both players, unlike Homicidal Chauffeur. Capture assessed post-hoc by the evaluator.",
    ["Isaacs, R. (1965). Differential Games. Wiley.",
     "Merz, A.W. (1972). The Homicidal Chauffeur — A Differential Game. PhD thesis, Stanford " *
     "(Ch. on the \"Two Cars\" pursuit variant)."],
)
register_case!(GAME_OF_TWO_CARS_NOMINAL_V1)

"""
    game_of_two_cars_evaluator() -> Evaluator
"""
game_of_two_cars_evaluator() = capture_evaluator(GAME_OF_TWO_CARS_CAPTURE_RADIUS)

function game_of_two_cars_visual_model(game, sol)
    spec = two_agent_visual_model(game, sol, ("Car 1 (pursuer role)", "Car 2 (evader role)"),
                                   GAME_OF_TWO_CARS_CAPTURE_RADIUS)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    agent1 = merge(spec.agents[1], (heading = Xm[3, :],))
    agent2 = merge(spec.agents[2], (heading = Xm[6, :],))
    return merge(spec, (agents = [agent1, agent2],))
end

register_visualizer!(GAME_OF_TWO_CARS_NOMINAL_V1.id, game_of_two_cars_visual_model)
