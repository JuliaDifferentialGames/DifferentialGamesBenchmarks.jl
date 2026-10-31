module DifferentialGamesBenchmarksGLMakieExt

# ============================================================================
# Interactive-only extension. Reuses the same `visual_model` data prep as the
# CairoMakie extension (no Makie-drawing code is duplicated at the primitive
# level — both extensions draw the same NamedTuple shape); this extension
# only adds a GLMakie `Screen` display on top of a figure built the same way
# `gameplot` builds one.
# ============================================================================

using DifferentialGamesBenchmarks
using DifferentialGamesBase
using Makie
using GLMakie

function DifferentialGamesBenchmarks.interactive(game, sol; preset::Symbol = :publication)
    fig = DifferentialGamesBenchmarks.gameplot(game, sol; preset = preset)
    display(GLMakie.Screen(), fig)
    return nothing
end

end
