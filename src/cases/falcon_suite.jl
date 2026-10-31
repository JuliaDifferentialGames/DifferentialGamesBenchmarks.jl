# ============================================================================
# cases/falcon_suite.jl
#
# The remaining problems of the FALCON validation suite on top of `hallway_passing.jl` and `racing.jl`:
#
#   * the Austin racing benchmark's 50 initial conditions (the comparison of the FALCON paper against DG-SQP),
#     as `austin_benchmark_initial_conditions()` and `racing_game_from_ic`;
#   * the first hairpin of the Austin track, the 30 m segment of the FALCON figure (`RACING_AUSTIN_HAIRPIN_V1`);
#   * the hallway with a communication range (`HALLWAY_PASSING_COMM_V1`).
#
# The racing games here use the continuous bicycle model integrated with one RK4 step per `dt` (default 0.05 s); the
# FALCON scripts use ten RK4 substeps per 0.1 s. The horizon of 2.5 s of the paper's benchmark is therefore N = 50.
# Solvers that take a pre-integrated step map (DGSQP and FALCON in their own packages) can reproduce the exact
# 10-substep discretisation; see `DifferentialGamesBaseSolvers.jl/src/solvers/DGSQP/examples/austin_benchmark.jl`.
# ============================================================================

"""
    austin_hairpin_track() -> RaceTrack

The 30 m segment `s ∈ [60, 90]` of the Austin tenth-scale track (re-based to 0) that contains the first hairpin turn, as in
the FALCON figure.
"""
austin_hairpin_track() = load_race_track(joinpath(@__DIR__, "..", "..", "data", "f1_austin_hairpin_60_90.csv"))

"""
    austin_benchmark_initial_conditions() -> NamedTuple

The 50 random initial conditions of the FALCON-vs-DG-SQP comparison on the segment `s ∈ [60, 80]` m
(`q0_1`, `q0_2`: 50×7 states `[vₓ, v_y, ψ̇, x, y, ψ, s]` of the two cars) together with the results of the original Python DG-SQP on
them (`py_converged`, `py_iters`, `py_time`). They come from `dgsqp_benchmark.npz` of the FALCON repository.
"""
function austin_benchmark_initial_conditions()
    rows = [parse.(Float64, split(l, ',')) for l in eachline(joinpath(@__DIR__, "..", "..", "data", "f1_austin_benchmark_ics.csv"))
            if !isempty(strip(l)) && !startswith(l, '#')]
    A = reduce(hcat, rows)'
    return (q0_1 = A[:, 1:7], q0_2 = A[:, 8:14], py_time = A[:, 15], py_iters = A[:, 16], py_converged = A[:, 17] .> 0.5)
end

"""
    racing_game_from_ic(i; params = RacingParameters(N = 50, dt = 0.05)) -> GameProblem

The Austin racing game of [`racing_game`](@ref) started from the `i`-th initial condition of
[`austin_benchmark_initial_conditions`](@ref) (`i ∈ 1:50`).
"""
function racing_game_from_ic(i::Int; params::RacingParameters = RacingParameters(N = 50, dt = 0.05),
                              track::RaceTrack = austin_track())
    ics = austin_benchmark_initial_conditions()
    @assert 1 <= i <= size(ics.q0_1, 1) "initial condition index must be in 1:$(size(ics.q0_1, 1))"
    return racing_game(track, params; q0 = (ics.q0_1[i, :], ics.q0_2[i, :]))
end

const RACING_AUSTIN_HAIRPIN_V1 = BenchmarkCase{Float64}(
    CaseID(:Racing, :AustinHairpin, v"1"),
    () -> racing_game(austin_hairpin_track(), RacingParameters(N = 80, dt = 0.05)),
    FullModelAccess(),
    "Two-car racing through the first hairpin of the Austin track (30 m segment, 4 s horizon): the sharp-corner example " *
    "of the FALCON paper. Same model, cost and constraints as `Racing.AustinNominal_v1`. A hard, non-convex benchmark.",
    ["Outland, B. & Arya, H. A Fast, Convergent Algorithm for Solving Non-convex Partially-Decoupled Generalized Nash Equilibrium Problems.",
     "Zhu, E. L. & Borrelli, F. (2024). A sequential quadratic programming approach to the solution of open-loop generalized Nash equilibria for autonomous racing."],
)
register_case!(RACING_AUSTIN_HAIRPIN_V1)

const HALLWAY_PASSING_COMM_V1 = BenchmarkCase{Float64}(
    CaseID(:HallwayPassing, :Communication, v"1"),
    () -> hallway_game(HallwayParameters(N = 300, d_comm = 0.5)),
    FullModelAccess(),
    "The FALCON hallway with a hard communication range: the three agents must also stay within 0.5 of one another, a " *
    "convoy incentive on top of the single-file passage of `HallwayPassing.Nominal_v1` (30 s horizon).",
    ["Outland, B. & Arya, H. A Fast, Convergent Algorithm for Solving Non-convex Partially-Decoupled Generalized Nash Equilibrium Problems (narrowing hallways problem)."],
)
register_case!(HALLWAY_PASSING_COMM_V1)
