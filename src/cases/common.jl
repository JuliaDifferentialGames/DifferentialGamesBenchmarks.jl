# ============================================================================
# cases/common.jl
#
# Shared helpers for the Classical Games pack. All four games (Simple
# Pursuit, Homicidal Chauffeur, Game of Two Cars, Lady in the Lake) are
# 2-player pursuit-evasion variants built via the PD-GNEP path, so capture-
# style geometry (positions, separation, capture radius) is genuinely shared
# logic — factored here once instead of duplicated per case file.
# ============================================================================

"""
    two_player_positions(game, sol; pos_dim=2) -> (pos1, pos2, times)

Slice out each of the two players' leading `pos_dim` state components from
the *joint* state (via `_joint_state_control_matrices` +
`GameMetadata.state_offsets`), rather than trusting `get_trajectory(sol,
i).states` to already be player `i`'s private slice — some solvers (e.g.
ALGAMES) populate every player's `Trajectory` with the full joint state
rather than a private per-player slice, despite the general `Trajectory`
docstring describing the PD-GNEP private-state convention.
"""
function two_player_positions(game, sol::DifferentialGamesBase.GNEPSolution; pos_dim::Int = 2)
    fg = _forward_game(game)
    Xm, _, times = _joint_state_control_matrices(game, sol)
    off1, off2 = fg.metadata.state_offsets[1], fg.metadata.state_offsets[2]
    return Xm[off1+1:off1+pos_dim, :], Xm[off2+1:off2+pos_dim, :], times
end

"""
    two_player_separation(game, sol; pos_dim=2) -> (dists, times)

Euclidean separation between the two players' positions at every time step.
"""
function two_player_separation(game, sol::DifferentialGamesBase.GNEPSolution; pos_dim::Int = 2)
    pos1, pos2, times = two_player_positions(game, sol; pos_dim = pos_dim)
    Δ = pos1 .- pos2
    return vec(sqrt.(sum(Δ .^ 2, dims = 1))), times
end

"""
    capture_radius_min(game, sol; pos_dim=2) -> Float64

Minimum separation achieved between the two players over the trajectory.
Independent recomputation from the trajectories — never trusts solver
convergence flags.
"""
function capture_radius_min(game, sol::DifferentialGamesBase.GNEPSolution; pos_dim::Int = 2)
    dists, _ = two_player_separation(game, sol; pos_dim = pos_dim)
    return minimum(dists)
end

"""
    time_to_capture(game, sol; capture_radius, pos_dim=2) -> Float64

First time at which separation drops to `capture_radius` or below; `Inf` if
never captured over the horizon.
"""
function time_to_capture(game, sol::DifferentialGamesBase.GNEPSolution;
                          capture_radius::Real, pos_dim::Int = 2)
    dists, times = two_player_separation(game, sol; pos_dim = pos_dim)
    idx = findfirst(<=(capture_radius), dists)
    return idx === nothing ? Inf : times[idx]
end

"""
    escape_flag(game, sol; capture_radius, pos_dim=2) -> Bool

`true` if the evader is never captured over the horizon.
"""
escape_flag(game, sol::DifferentialGamesBase.GNEPSolution; capture_radius::Real, pos_dim::Int = 2) =
    isinf(time_to_capture(game, sol; capture_radius = capture_radius, pos_dim = pos_dim))

"""
    capture_evaluator(capture_radius::Real; pos_dim::Int=2) -> Evaluator

`default_evaluator()` extended with capture-specific quality metrics bound to
`capture_radius` (and `pos_dim`, for games whose position isn't the leading 2
state components). Shared constructor for every pursuit-evasion classical
case — each case only needs to supply its own capture radius.
"""
function capture_evaluator(capture_radius::Real; pos_dim::Int = 2)
    ev = default_evaluator()
    ev.quality_metrics[:capture_radius_min] = (game, sol) -> capture_radius_min(game, sol; pos_dim = pos_dim)
    ev.quality_metrics[:time_to_capture] =
        (game, sol) -> time_to_capture(game, sol; capture_radius = capture_radius, pos_dim = pos_dim)
    ev.quality_metrics[:escape] =
        (game, sol) -> escape_flag(game, sol; capture_radius = capture_radius, pos_dim = pos_dim)
    return ev
end

"""
    two_agent_visual_model(game, sol, labels, capture_radius; pos_dim=2, boundary=nothing) -> NamedTuple

Shared `visual_model` builder for two-agent pursuit-evasion cases: agent
trajectories (labelled `labels[1]`/`labels[2]`) plus a capture-radius circle
centered on the evader at the moment of minimum separation. `boundary`, if
given, is forwarded verbatim (e.g. a lake shoreline circle for Lady in the
Lake) — see `ext/DifferentialGamesBenchmarksVizExt.jl` for how it's drawn.
"""
function two_agent_visual_model(
    game, sol::DifferentialGamesBase.GNEPSolution, labels::NTuple{2,String}, capture_radius::Real;
    pos_dim::Int = 2, boundary = nothing,
)
    pos1, pos2, times = two_player_positions(game, sol; pos_dim = pos_dim)
    dists, _ = two_player_separation(game, sol; pos_dim = pos_dim)
    k_min = argmin(dists)
    agents = [
        (player_id = 1, label = labels[1], positions = pos1),
        (player_id = 2, label = labels[2], positions = pos2),
    ]
    capture_region = (center = pos2[1:2, k_min], radius = capture_radius)
    return boundary === nothing ?
        (agents = agents, capture_region = capture_region, times = times) :
        (agents = agents, capture_region = capture_region, times = times, boundary = boundary)
end
