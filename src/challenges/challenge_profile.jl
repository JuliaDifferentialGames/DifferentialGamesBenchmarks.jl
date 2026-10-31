# ============================================================================
# challenges/challenge_profile.jl
#
# ChallengeProfile is a *named composition* of assumption-removal transforms.
# It is descriptive/provenance metadata for filtering and reporting — the
# actual mathematical effect always lives in the real Base type each
# transform function returns (StochasticGameProblem, RobustGameProblem,
# POGameProblem, ...), never in the profile itself.
# ============================================================================

"""
    ChallengeProfile

A named, ordered composition of challenge-transformation functions.

# Fields
- `name::Symbol`
- `transforms::Vector{Function}` : each `AbstractGameProblem -> AbstractGameProblem`
"""
struct ChallengeProfile
    name::Symbol
    transforms::Vector{Function}
end

ChallengeProfile(name::Symbol) = ChallengeProfile(name, Function[])

"""
    apply(profile::ChallengeProfile, game) -> AbstractGameProblem

Apply every transform in `profile.transforms`, in order, to `game`.
"""
apply(profile::ChallengeProfile, game) = foldl((g, f) -> f(g), profile.transforms; init = game)

const NOMINAL_CHALLENGE = ChallengeProfile(:Nominal)
