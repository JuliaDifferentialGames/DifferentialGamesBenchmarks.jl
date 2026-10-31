# ============================================================================
# registry/case_registry.jl — minimal official-case registry.
# ============================================================================

const CASE_REGISTRY = Dict{CaseID,BenchmarkCase}()

"""
    register_case!(case::BenchmarkCase) -> BenchmarkCase

Register `case` under its `id` in the official case registry.
"""
register_case!(case::BenchmarkCase) = (CASE_REGISTRY[case.id] = case)

"""
    lookup_case(id::CaseID) -> BenchmarkCase

Look up a registered case by `id`. Throws `KeyError` if not registered.
"""
lookup_case(id::CaseID) = CASE_REGISTRY[id]

# ============================================================================
# Case-specific visualizers
#
# Every classical game builds the same concrete `GameProblem{Float64}` type,
# so `visual_model(game, sol)` cannot be specialized per case via ordinary
# multiple dispatch (dispatch is on type, not on which case produced the
# instance). Cases register a `(game, sol) -> NamedTuple` visualizer keyed by
# `CaseID` instead; `gameplot(case, sol)`/`dashboard`/etc. look it up here,
# falling back to the generic `visual_model` when a case hasn't registered one.
# ============================================================================

const CASE_VISUALIZERS = Dict{CaseID,Function}()

"""
    register_visualizer!(id::CaseID, f::Function) -> Function

Register `f(game, sol) -> NamedTuple` as the visual-model builder for case
`id`. Looked up by `gameplot(case::BenchmarkCase, sol)` and friends.
"""
register_visualizer!(id::CaseID, f::Function) = (CASE_VISUALIZERS[id] = f)

"""
    visual_model_for(case::BenchmarkCase, game, sol) -> NamedTuple

Resolve and call the case's registered visualizer, falling back to the
generic `visual_model(game, sol)` if none is registered for `case.id`.
"""
visual_model_for(case::BenchmarkCase, game, sol) =
    get(CASE_VISUALIZERS, case.id, visual_model)(game, sol)
