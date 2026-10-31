# ============================================================================
# cases/lady_in_the_lake.jl
#
# The geometrically odd one out of the four classical games: an asymmetric
# DOMAIN, not just asymmetric dynamics. The swimmer (evader) moves freely in
# the plane; the pursuer is confined to the lake's shore (a 1-dimensional
# angular state), but runs faster than the swimmer swims. The swimmer wins by
# reaching the shore far in angle from the pursuer's position.
#
# This case cannot reuse `cases/common.jl`'s two-player helpers: the
# pursuer's state is a single angle, not a 2D position, so
# `two_player_positions`'s "first `pos_dim` components of each player's
# block" convention doesn't apply. Its cost also isn't expressible via the
# existing `avoid_proximity`/`maintain_proximity` cost terms (those assume
# both players live in the same Cartesian embedding); this is exactly the
# "genuinely classical-game-specific cost" case the design calls out as
# belonging in the benchmark package. Both players' costs are built directly
# from Base's generic `NonlinearStageCost`/`NonlinearTerminalCost` (raw
# functions, ForwardDiff-differentiated automatically) — no new Base types
# needed.
#
# Solver: ALGAMES (`:SeparableDynamics` + `:ConstrainedGame`).
# ============================================================================

const LADY_IN_THE_LAKE_RADIUS = 3.0
const _LITL_SWIM_SPEED_MAX = 0.5
const _LITL_SHORE_SPEED = 1.5
const _LITL_SHORE_ANGULAR_RATE_MAX = _LITL_SHORE_SPEED / LADY_IN_THE_LAKE_RADIUS

_litl_radial_cost(x) = (sqrt(x[1]^2 + x[2]^2 + 1e-9) - LADY_IN_THE_LAKE_RADIUS)^2
_litl_angular_alignment(x) = cos(atan(x[2], x[1]) - x[3])

function _build_lady_in_the_lake_nominal()
    # Swimmer starts a hair off the exact center: atan(y, x) (used in the
    # angular-alignment cost) has an undefined/NaN gradient at the origin
    # itself (∂/∂y atan(y,x) = x/(x²+y²) → 0/0), which poisons the very
    # first ForwardDiff-computed gradient if the swimmer starts at (0, 0).
    x0_swimmer = [1e-3, 0.0]
    x0_pursuer = [0.0]  # shore angle 0

    Rs = 1.0
    Rp = 1.0
    W_RADIAL_STAGE, W_RADIAL_TERM = 0.2, 10.0
    W_ANGLE_STAGE, W_ANGLE_TERM   = 0.5, 5.0

    swimmer_stage(x, u, p, t) = W_RADIAL_STAGE * _litl_radial_cost(x) +
                                W_ANGLE_STAGE * _litl_angular_alignment(x) +
                                0.5 * Rs * dot(u, u)
    swimmer_terminal(x, p) = W_RADIAL_TERM * _litl_radial_cost(x) + W_ANGLE_TERM * _litl_angular_alignment(x)
    swimmer_obj = DifferentialGamesBase.PlayerObjective(
        1,
        DifferentialGamesBase.NonlinearStageCost(swimmer_stage),
        DifferentialGamesBase.NonlinearTerminalCost(swimmer_terminal),
    )

    pursuer_stage(x, u, p, t) = -W_ANGLE_STAGE * _litl_angular_alignment(x) + 0.5 * Rp * dot(u, u)
    pursuer_terminal(x, p) = -W_ANGLE_TERM * _litl_angular_alignment(x)
    pursuer_obj = DifferentialGamesBase.PlayerObjective(
        2,
        DifferentialGamesBase.NonlinearStageCost(pursuer_stage),
        DifferentialGamesBase.NonlinearTerminalCost(pursuer_terminal),
    )

    swim_bound = DifferentialGamesBase.control_bounds(
        1; control_offset = 0, control_dim = 2,
        lower = fill(-_LITL_SWIM_SPEED_MAX, 2), upper = fill(_LITL_SWIM_SPEED_MAX, 2),
    )
    shore_bound = DifferentialGamesBase.control_bounds(
        2; control_offset = 2, control_dim = 1,
        lower = [-_LITL_SHORE_ANGULAR_RATE_MAX], upper = [_LITL_SHORE_ANGULAR_RATE_MAX],
    )

    swimmer_dynamics(x, u, p, t) = u          # ẋ = u  (2D velocity control)
    pursuer_dynamics(x, u, p, t) = [u[1]]     # θ̇ = ω  (1D angular rate control)

    swimmer = DifferentialGamesBase.Player{Float64}(1, 2, 2, x0_swimmer, swimmer_dynamics, swimmer_obj, [swim_bound])
    pursuer = DifferentialGamesBase.Player{Float64}(2, 1, 1, x0_pursuer, pursuer_dynamics, pursuer_obj, [shore_bound])

    return DifferentialGamesBase.DifferentialGame([swimmer, pursuer], 10.0, 0.2)
end

const LADY_IN_THE_LAKE_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:LadyInTheLake, :Nominal, v"1"),
    _build_lady_in_the_lake_nominal,
    FullModelAccess(),
    "Swimmer (free planar motion, speed ≤ $_LITL_SWIM_SPEED_MAX) starting at the center of a " *
    "radius-$LADY_IN_THE_LAKE_RADIUS lake, versus a pursuer confined to the shore " *
    "(angular state only, running speed $_LITL_SHORE_SPEED). The swimmer wins by reaching " *
    "the shore far in angle from the pursuer.",
    ["Isaacs, R. (1965). Differential Games. Wiley. (\"The Lady in the Lake\")"],
)
register_case!(LADY_IN_THE_LAKE_NOMINAL_V1)

"""
    lady_in_the_lake_landing_metrics(game, sol) -> NamedTuple

Independent geometric-event evaluation: the first time the swimmer's radius
reaches the lake radius ("landing", within `shore_tol` — the swimmer's cost
is a soft quadratic pull toward the shore, not a hard constraint, so it
approaches but never exactly equals `LADY_IN_THE_LAKE_RADIUS`), and the
angular separation from the pursuer at that moment. `reached_shore=false`
(with `landing_time=Inf`) if the swimmer never gets within `shore_tol` of the
boundary over the horizon.
"""
function lady_in_the_lake_landing_metrics(game, sol::DifferentialGamesBase.GNEPSolution; shore_tol::Real = 0.1)
    Xm, _, times = _joint_state_control_matrices(game, sol)
    sx, sy, θ = Xm[1, :], Xm[2, :], Xm[3, :]
    r = sqrt.(sx .^ 2 .+ sy .^ 2)
    swimmer_angle = atan.(sy, sx)
    idx = findfirst(>=(LADY_IN_THE_LAKE_RADIUS - shore_tol), r)
    idx === nothing && return (reached_shore = false, landing_time = Inf, angular_separation = NaN)
    Δθ = swimmer_angle[idx] - θ[idx]
    sep = abs(atan(sin(Δθ), cos(Δθ)))  # wrapped to [0, π]
    return (reached_shore = true, landing_time = times[idx], angular_separation = sep)
end

"""
    lady_in_the_lake_evaluator() -> Evaluator
"""
function lady_in_the_lake_evaluator()
    ev = default_evaluator()
    ev.quality_metrics[:reached_shore] = (game, sol) -> lady_in_the_lake_landing_metrics(game, sol).reached_shore
    ev.quality_metrics[:landing_time] = (game, sol) -> lady_in_the_lake_landing_metrics(game, sol).landing_time
    ev.quality_metrics[:angular_separation_at_landing] =
        (game, sol) -> lady_in_the_lake_landing_metrics(game, sol).angular_separation
    return ev
end

function lady_in_the_lake_visual_model(game, sol::DifferentialGamesBase.GNEPSolution)
    Xm, _, times = _joint_state_control_matrices(game, sol)
    swimmer_pos = Xm[1:2, :]
    pursuer_pos = vcat(
        (LADY_IN_THE_LAKE_RADIUS .* cos.(Xm[3, :]))',
        (LADY_IN_THE_LAKE_RADIUS .* sin.(Xm[3, :]))',
    )
    return (
        agents = [
            (player_id = 1, label = "Swimmer", positions = swimmer_pos),
            (player_id = 2, label = "Pursuer (shore)", positions = pursuer_pos),
        ],
        boundary = (center = [0.0, 0.0], radius = LADY_IN_THE_LAKE_RADIUS, label = "shore"),
        times = times,
    )
end

register_visualizer!(LADY_IN_THE_LAKE_NOMINAL_V1.id, lady_in_the_lake_visual_model)
