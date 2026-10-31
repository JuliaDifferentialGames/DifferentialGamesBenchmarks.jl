# ============================================================================
# challenges/stochastic_challenge.jl
#
# Assembles a DifferentialGamesBase.StochasticGameProblem — a real Base type,
# never a benchmark-only flag.
# ============================================================================

"""
    add_process_noise(game::DifferentialGamesBase.AbstractDeterministicGame, Σ) -> StochasticGameProblem

Wrap `game` with additive process noise covariance `Σ` (a `Matrix` for LTI
noise, or `Vector{Matrix}` for LTV).
"""
add_process_noise(game::DifferentialGamesBase.AbstractDeterministicGame, Σ) =
    DifferentialGamesBase.StochasticGameProblem(game, DifferentialGamesBase.ProcessNoiseModel(Σ))
