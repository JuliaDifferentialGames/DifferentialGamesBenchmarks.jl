# ============================================================================
# access/access_spec.jl
#
# AccessSpec — the "what may the algorithm know" oracle layer. Distinct from
# DifferentialGamesBase.AbstractInformationStructure (what the true game
# structure gives each player) — this is benchmark-owned and describes what a
# *method under test* is permitted to see of the (always-complete) truth game.
# ============================================================================

"""
    AccessSpec

Abstract type describing what a method under test may know about
`case.truth`. The truth game is always complete; `AccessSpec` restricts what
a solver/policy is given, never what actually exists.
"""
abstract type AccessSpec end

"""
    FullModelAccess <: AccessSpec

The method may use the exact `AbstractGameProblem` directly (dynamics,
objectives, constraints all visible). Default for ordinary numerical solvers.
"""
struct FullModelAccess <: AccessSpec end

"""
    ApproximateModelAccess <: AccessSpec

The method receives a perturbed/mismatched model instead of the exact truth.

# Fields
- `perturb::Function` : `AbstractGameProblem -> AbstractGameProblem`
"""
struct ApproximateModelAccess <: AccessSpec
    perturb::Function
end

"""
    SimulatorOnlyAccess <: AccessSpec

The method may only step a black-box simulator; no analytic dynamics/Jacobian
is exposed.

# Fields
- `step::Function` : `(x, u) -> x′`
"""
struct SimulatorOnlyAccess <: AccessSpec
    step::Function
end

"""
    ModelFreeAccess <: AccessSpec

The method may only observe trajectories; no dynamics/objective access at all.
"""
struct ModelFreeAccess <: AccessSpec end

"""
    MaskedObjectives <: AccessSpec

The method sees the objectives only of `visible_players`; other players'
objectives are hidden (the *ground truth* still has them — only the view is
restricted).

# Fields
- `visible_players::Vector{Int}`
"""
struct MaskedObjectives <: AccessSpec
    visible_players::Vector{Int}
end

"""
    ObservationOnlyAccess <: AccessSpec

The method receives observations through `observation_model` rather than the
true state. Reuses `DifferentialGamesBase.ObservationModel` directly — no
parallel observation type.

# Fields
- `observation_model::DifferentialGamesBase.ObservationModel`
"""
struct ObservationOnlyAccess <: AccessSpec
    observation_model::DifferentialGamesBase.ObservationModel
end

"""
    access(spec::AccessSpec, game::DifferentialGamesBase.AbstractGameProblem)

Return whatever reduced view of `game` a method with access `spec` is allowed
to consume. `FullModelAccess` returns `game` itself unchanged; other specs
return a restricted view (a perturbed game, a masked-objective game, or an
error directing the caller to use `spec.step`/`spec.observation_model`
directly, for access levels that aren't representable as a game object).
"""
access(::FullModelAccess, game) = game
access(spec::ApproximateModelAccess, game) = spec.perturb(game)
access(::SimulatorOnlyAccess, game) =
    error("SimulatorOnlyAccess exposes only `spec.step`; no game object view is available.")
access(::ModelFreeAccess, game) =
    error("ModelFreeAccess exposes only trajectories; no game object view is available.")
function access(spec::MaskedObjectives, game::DifferentialGamesBase.GameProblem)
    masked_objectives = [
        obj.player_id in spec.visible_players ? obj : _hidden_objective(obj)
        for obj in game.objectives
    ]
    return DifferentialGamesBase.remake(game; objectives = masked_objectives)
end
access(spec::ObservationOnlyAccess, game) =
    error("ObservationOnlyAccess exposes only `spec.observation_model`; no game object view is available.")

# Placeholder used by MaskedObjectives: a zero-cost objective so the game
# object still type-checks, without revealing the true cost weights.
function _hidden_objective(obj::DifferentialGamesBase.PlayerObjective)
    error(
        "Player $(obj.player_id)'s objective is masked by MaskedObjectives; " *
        "it cannot be substituted with a placeholder without also faking the " *
        "cost type — construct a case-specific hidden objective instead."
    )
end
