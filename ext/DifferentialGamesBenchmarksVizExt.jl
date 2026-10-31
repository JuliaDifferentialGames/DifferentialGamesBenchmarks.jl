module DifferentialGamesBenchmarksVizExt

# ============================================================================
# Headless (CairoMakie) rendering: gameplot/renderframe/animate/dashboard.
#
# Reads only a `visual_model(...)`-produced plain-data NamedTuple; never Base
# solution internals directly. Two entry points:
#   gameplot(game, sol)          — generic fallback (visual_spec/visual_model
#                                  dispatch on game's type; works for any
#                                  GameProblem, but every classical case
#                                  builds the same concrete GameProblem{T}
#                                  type, so this can't be case-specific).
#   gameplot(case, sol)          — case-aware; looks up the case's registered
#                                  visualizer via `visual_model_for`, giving
#                                  richer output (capture regions, lake
#                                  boundary, vehicle headings, ...).
# ============================================================================

using DifferentialGamesBenchmarks
using DifferentialGamesBase
using Makie
using CairoMakie

const _KNOWN_PRESETS = (:publication, :debug)

function _resolve_preset(preset::Symbol)
    preset in _KNOWN_PRESETS && return preset
    @warn "preset :$preset not yet implemented, using :publication"
    return :publication
end

_spec_for(game, sol) = DifferentialGamesBenchmarks.visual_model(game, sol)
_spec_for(case::DifferentialGamesBenchmarks.BenchmarkCase, sol) =
    DifferentialGamesBenchmarks.visual_model_for(case, DifferentialGamesBenchmarks.game(case), sol)

function _draw_boundary!(ax, spec)
    hasproperty(spec, :boundary) || return ax
    b = spec.boundary
    θ = range(0, 2π; length = 128)
    lines!(ax, b.center[1] .+ b.radius .* cos.(θ), b.center[2] .+ b.radius .* sin.(θ);
           color = :steelblue, linewidth = 2, label = get(b, :label, "boundary"))
    return ax
end

function _draw_capture_region!(ax, spec)
    hasproperty(spec, :capture_region) || return ax
    cr = spec.capture_region
    θ = range(0, 2π; length = 64)
    lines!(ax, cr.center[1] .+ cr.radius .* cos.(θ), cr.center[2] .+ cr.radius .* sin.(θ);
           linestyle = :dash, color = :red, label = "capture region")
    return ax
end

function _draw_walls!(ax, spec)
    hasproperty(spec, :walls) || return ax
    for w in spec.walls
        lines!(ax, w.x, w.y; color = :black, linewidth = 2)
    end
    return ax
end

function _draw_agents!(ax, spec; upto::Union{Nothing,Int} = nothing)
    for agent in spec.agents
        pos = upto === nothing ? agent.positions : agent.positions[:, 1:upto]
        lines!(ax, pos[1, :], pos[2, :]; label = agent.label)
        scatter!(ax, pos[1, end:end], pos[2, end:end]; markersize = 14)
        if hasproperty(agent, :heading)
            θ = upto === nothing ? agent.heading[end] : agent.heading[upto]
            p = pos[:, end]
            arrows!(ax, [p[1]], [p[2]], [0.4 * cos(θ)], [0.4 * sin(θ)]; color = :black)
        end
    end
    return ax
end

function _draw_game!(ax, spec; upto::Union{Nothing,Int} = nothing)
    hasproperty(spec, :boundary) && _draw_boundary!(ax, spec)
    _draw_walls!(ax, spec)
    _draw_agents!(ax, spec; upto = upto)
    upto === nothing && _draw_capture_region!(ax, spec)
    return ax
end

function _gameplot(spec; preset::Symbol = :publication)
    preset = _resolve_preset(preset)
    fig = Figure(size = preset == :debug ? (900, 700) : (600, 500))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect())
    _draw_game!(ax, spec)
    axislegend(ax)
    if preset == :debug
        Label(fig[0, 1], "Debug preset — $(length(spec.agents)) agents"; tellwidth = false)
    end
    return fig
end

DifferentialGamesBenchmarks.gameplot(game_or_case, sol; preset::Symbol = :publication) =
    _gameplot(_spec_for(game_or_case, sol); preset = preset)

function DifferentialGamesBenchmarks.renderframe(game_or_case, sol, t::Real; preset::Symbol = :publication)
    preset = _resolve_preset(preset)
    spec = _spec_for(game_or_case, sol)
    k = clamp(searchsortedlast(spec.times, t), 1, length(spec.times))
    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect())
    _draw_game!(ax, spec; upto = k)
    axislegend(ax)
    return fig
end

function DifferentialGamesBenchmarks.animate(
    path::AbstractString, game_or_case, sol; fps::Int = 30, preset::Symbol = :conference
)
    preset = _resolve_preset(preset)
    spec = _spec_for(game_or_case, sol)
    n_steps = size(spec.agents[1].positions, 2)
    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect())
    _draw_boundary!(ax, spec)
    k_obs = Observable(1)
    for agent in spec.agents
        pos = agent.positions
        xs = @lift(pos[1, 1:$k_obs])
        ys = @lift(pos[2, 1:$k_obs])
        lines!(ax, xs, ys; label = agent.label)
    end
    axislegend(ax)
    record(fig, path, 1:n_steps; framerate = fps) do k
        k_obs[] = k
    end
    return nothing
end

function DifferentialGamesBenchmarks.dashboard(result::DifferentialGamesBenchmarks.ExperimentResult; preset::Symbol = :debug)
    fig = DifferentialGamesBenchmarks.gameplot(result.experiment.case, result.solution; preset = preset)
    metrics_text = join(
        ["$k = $(round(v, sigdigits=4))" for (k, v) in result.quality if v isa Real],
        "\n",
    )
    Label(fig[1, 2], "Metrics\n" * metrics_text; tellheight = false)
    return fig
end

end
