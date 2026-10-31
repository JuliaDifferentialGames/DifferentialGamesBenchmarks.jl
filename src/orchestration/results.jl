# ============================================================================
# orchestration/results.jl — ExperimentResult: plain-data report.
# ============================================================================

"""
    ExperimentResult

Plain-data report produced by `run(::Experiment)`.

# Fields
- `experiment`  : the `Experiment` that produced this result
- `solution`    : the `AbstractSolution` returned by `DifferentialGamesBase.solve`
- `wall_time_s` : externally measured wall-clock time around the `solve` call
  (authoritative — cross-check against `efficiency[:solve_time]`, which is
  solver-self-reported)
- `validity::Dict{Symbol,Bool}`
- `quality::Dict{Symbol,Any}`
- `robustness::Dict{Symbol,Any}`
- `efficiency::Dict{Symbol,Any}`
- `metadata::Dict{Symbol,Any}`
"""
struct ExperimentResult
    experiment::Experiment
    solution::DifferentialGamesBase.AbstractSolution
    wall_time_s::Float64
    validity::Dict{Symbol,Bool}
    quality::Dict{Symbol,Any}
    robustness::Dict{Symbol,Any}
    efficiency::Dict{Symbol,Any}
    metadata::Dict{Symbol,Any}
end
