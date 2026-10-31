# ============================================================================
# orchestration/budget.jl — resource ceiling for an experiment run.
# ============================================================================

"""
    Budget

# Fields
- `max_wall_time_s::Float64`
- `max_iterations::Union{Nothing,Int}`
- `max_solver_calls::Union{Nothing,Int}`
"""
struct Budget
    max_wall_time_s::Float64
    max_iterations::Union{Nothing,Int}
    max_solver_calls::Union{Nothing,Int}
end

Budget(; max_wall_time_s::Real = 60.0, max_iterations = nothing, max_solver_calls = nothing) =
    Budget(Float64(max_wall_time_s), max_iterations, max_solver_calls)
