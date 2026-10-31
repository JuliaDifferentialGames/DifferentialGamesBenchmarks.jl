module DifferentialGamesBenchmarksSolversExt

# ============================================================================
# Optional convenience wiring for when a user has DifferentialGamesBaseSolvers
# loaded alongside DifferentialGamesBenchmarks. Currently a placeholder — no
# benchmark API requires BaseSolvers to compile (weakdep only); this extension
# exists as the designated home for future solver-specific convenience
# wrappers (e.g. auto-selecting a compatible default solver for a case) so
# they don't have to live in the main package and force a hard dependency.
# ============================================================================

using DifferentialGamesBenchmarks
using DifferentialGamesBaseSolvers

end
