# ============================================================================
# challenges/po_challenge.jl
#
# Assembles a DifferentialGamesBase.POGameProblem — a real Base type, reusing
# PrivateObservation/SharedObservation directly.
# ============================================================================

"""
    restrict_observation(game::DifferentialGamesBase.AbstractDeterministicGame, observation_models) -> POGameProblem

Wrap `game` so each player observes according to its entry in
`observation_models` (one `PrivateObservation`/`SharedObservation` per player,
in player-index order).
"""
restrict_observation(game::DifferentialGamesBase.AbstractDeterministicGame, observation_models) =
    DifferentialGamesBase.POGameProblem(game, observation_models)
