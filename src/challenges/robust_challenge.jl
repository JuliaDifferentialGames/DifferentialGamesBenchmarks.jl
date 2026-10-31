# ============================================================================
# challenges/robust_challenge.jl
#
# Assembles a DifferentialGamesBase.RobustGameProblem — a real Base type.
# ============================================================================

"""
    add_disturbance_bound(game::DifferentialGamesBase.AbstractDeterministicGame, w_min, w_max) -> RobustGameProblem

Wrap `game` with a bounded additive disturbance `w ∈ [w_min, w_max]`.
"""
add_disturbance_bound(game::DifferentialGamesBase.AbstractDeterministicGame, w_min, w_max) =
    DifferentialGamesBase.RobustGameProblem(game, DifferentialGamesBase.DisturbanceSet(w_min, w_max))
