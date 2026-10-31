# ============================================================================
# identity/benchmark_case.jl — BenchmarkCase: the official, versioned,
# reproducible case definition. `case.truth` is always a real
# DifferentialGamesBase.AbstractGameProblem — never a benchmark-private copy.
# ============================================================================

"""
    BenchmarkCase{T}

Immutable, versioned reference to an official benchmark case.

# Fields
- `id::CaseID`
- `build::Function`              : zero-arg factory `() -> AbstractGameProblem{T}`.
  Always rebuilds a fresh instance — never a cached mutable object — so a case
  is reproducible regardless of what a previous caller mutated.
- `default_access::AccessSpec`
- `description::String`
- `references::Vector{String}`   : citations
"""
struct BenchmarkCase{T}
    id::CaseID
    build::Function
    default_access::AccessSpec
    description::String
    references::Vector{String}
end

BenchmarkCase{T}(
    id::CaseID, build::Function, default_access::AccessSpec, description::String
) where {T} = BenchmarkCase{T}(id, build, default_access, description, String[])

"""
    game(case::BenchmarkCase{T}) -> AbstractGameProblem{T}

Build a fresh instance of the case's truth game.
"""
game(case::BenchmarkCase) = case.build()

function Base.show(io::IO, case::BenchmarkCase{T}) where {T}
    print(io, "BenchmarkCase{$T}(", string(case.id), ")")
end
