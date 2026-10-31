# ============================================================================
# challenges/communication_challenge.jl
#
# Restricts which players' information a solver's controller may use, via
# DifferentialGamesBase.GameMetadata.communication. Does not touch dynamics,
# objectives, or constraints — same GameProblem type, just a metadata field.
# ============================================================================

"""
    restrict_communication(game::DifferentialGamesBase.GameProblem, topology) -> GameProblem

Return a copy of `game` (O(1), shared references per `remake`) with
`metadata.communication` set to `topology`.
"""
function restrict_communication(
    game::DifferentialGamesBase.GameProblem,
    topology::DifferentialGamesBase.AbstractCommunicationTopology
)
    old_meta = game.metadata
    new_meta = DifferentialGamesBase.GameMetadata(
        old_meta.state_dims, old_meta.control_dims,
        old_meta.state_offsets, old_meta.control_offsets,
        old_meta.coupling_graph, old_meta.is_potential, old_meta.potential_function,
        topology
    )
    return DifferentialGamesBase.remake(game; metadata = new_meta)
end
