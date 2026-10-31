using Test
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

@testset "Lady in the Lake end-to-end" begin
    case = lookup_case(CaseID(:LadyInTheLake, :Nominal, v"1"))
    g = game(case)
    @test DifferentialGamesBase.n_players(g) == 2

    sol = DifferentialGamesBase.solve(g, ALGAMES(); verbose = false)
    @test sol.converged

    ev = lady_in_the_lake_evaluator()
    result = evaluate(ev, g, sol)

    @test result.validity[:constraints_satisfied]
    @test result.robustness[:dynamics_residual] < 1e-4
    for (_, r) in result.quality[:nash_residual]
        @test r < 1e-3
    end

    metrics = lady_in_the_lake_landing_metrics(g, sol)
    if metrics.reached_shore
        @test metrics.landing_time > 0.0
        @test 0.0 <= metrics.angular_separation <= pi
    else
        @test isinf(metrics.landing_time)
    end
end
