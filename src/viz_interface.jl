# ============================================================================
# viz_interface.jl
#
# Extension points for visualization, with no Makie dependency — so case
# authors can implement `visual_spec`/`visual_model` without pulling Makie.
# The Makie-based renderer lives in ext/DifferentialGamesBenchmarksVizExt.jl
# and ext/DifferentialGamesBenchmarksGLMakieExt.jl (activated only when the
# user has Makie/CairoMakie/GLMakie loaded).
# ============================================================================

"""
    visual_spec(game) -> NamedTuple

Return a backend-agnostic, generic description of `game`'s visualizable
geometry (agent labels only — no positions/capture region, since a bare
`GameProblem` carries no solution). Plain data — no Makie types.

Every classical game builds the same concrete `GameProblem{Float64}` type, so
richer per-case visualization (positions, capture regions, boundaries)
cannot be reached via multiple dispatch on `game`'s type alone — see
`register_visualizer!`/`visual_model_for` in `registry/case_registry.jl` for
how cases attach case-specific visual models, keyed by `CaseID`.
"""
function visual_spec(game::DifferentialGamesBase.GameProblem)
    n = DifferentialGamesBase.n_players(game)
    return (agents = [(player_id = i, label = "Player $i") for i in 1:n],)
end

"""
    visual_model(game, sol) -> NamedTuple

Generic fallback for games with no registered case-specific visualizer (see
`register_visualizer!`): plots each player's own `Trajectory.states[1:2, :]`
directly as its position. This is a "best effort" assumption (leading two
state components are roughly Cartesian) — case-specific visualizers handle
the real per-solver `Trajectory` conventions (e.g. ALGAMES populating every
player's `Trajectory` with the full joint state) and richer geometry
(capture regions, boundaries) via `two_agent_visual_model`/`register_visualizer!`.
"""
function visual_model(game::DifferentialGamesBase.GameProblem, sol::DifferentialGamesBase.GNEPSolution)
    n = DifferentialGamesBase.n_players(game)
    agents = [
        (player_id = i, label = "Player $i",
         positions = DifferentialGamesBase.get_trajectory(sol, i).states[1:2, :])
        for i in 1:n
    ]
    return (agents = agents,)
end

# ============================================================================
# Makie-backed rendering API. Declared here (Makie-free) so calling one of
# these without CairoMakie/Makie loaded gives a clear error instead of an
# UndefVarError; ext/DifferentialGamesBenchmarksVizExt.jl adds the real
# methods once Makie + CairoMakie are loaded, and
# ext/DifferentialGamesBenchmarksGLMakieExt.jl adds `interactive`.
# ============================================================================

const _VIZ_EXT_HINT = "load `using CairoMakie` (or `using Makie, CairoMakie`) to enable it."

"""
    gameplot(game, solution; preset=:publication) -> Figure

Static plot of a solved game's trajectories. Requires CairoMakie (or Makie)
to be loaded — see `ext/DifferentialGamesBenchmarksVizExt.jl`.
"""
function gameplot(args...; kwargs...)
    error("gameplot requires the Viz extension; $_VIZ_EXT_HINT")
end

"""
    dashboard(result::ExperimentResult; preset=:debug) -> Figure

Multi-panel figure combining `gameplot` with the result's recomputed metrics.
Requires CairoMakie (or Makie) to be loaded.
"""
function dashboard(args...; kwargs...)
    error("dashboard requires the Viz extension; $_VIZ_EXT_HINT")
end

"""
    renderframe(game, solution, t; preset=:publication) -> Figure

Render the game state at a single time `t`. Requires CairoMakie (or Makie)
to be loaded.
"""
function renderframe(args...; kwargs...)
    error("renderframe requires the Viz extension; $_VIZ_EXT_HINT")
end

"""
    animate(path, game, solution; fps=30, preset=:conference) -> nothing

Render an animation of the solved trajectory to `path`. Requires CairoMakie
(or Makie) to be loaded.
"""
function animate(args...; kwargs...)
    error("animate requires the Viz extension; $_VIZ_EXT_HINT")
end

"""
    interactive(game, solution) -> nothing

Open an interactive GLMakie window. Requires `using GLMakie` (see
`ext/DifferentialGamesBenchmarksGLMakieExt.jl`).
"""
function interactive(args...; kwargs...)
    error("interactive requires the GLMakie extension; load `using Makie, GLMakie` to enable it.")
end
