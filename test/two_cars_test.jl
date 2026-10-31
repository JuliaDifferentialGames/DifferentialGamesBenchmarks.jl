using Test
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

@testset "Game of Two Cars end-to-end" begin
    case = lookup_case(CaseID(:GameOfTwoCars, :Nominal, v"1"))
    g = game(case)
    @test DifferentialGamesBase.n_players(g) == 2

    sol = DifferentialGamesBase.solve(g, ALGAMES(); verbose = false)
    @test sol.converged

    ev = game_of_two_cars_evaluator()
    result = evaluate(ev, g, sol)

    @test result.validity[:constraints_satisfied]
    @test result.quality[:capture_radius_min] >= 0.0
    @test result.robustness[:dynamics_residual] < 1e-4
    for (_, r) in result.quality[:nash_residual]
        @test r < 1e-3
    end
end
