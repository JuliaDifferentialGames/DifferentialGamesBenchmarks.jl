using Test
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

@testset "Homicidal Chauffeur end-to-end" begin
    case = lookup_case(CaseID(:HomicidalChauffeur, :Nominal, v"1"))
    g = game(case)
    @test DifferentialGamesBase.n_players(g) == 2

    sol = DifferentialGamesBase.solve(g, ALGAMES(); verbose = false)
    @test sol.converged

    ev = homicidal_chauffeur_evaluator()
    result = evaluate(ev, g, sol)

    @test result.validity[:constraints_satisfied]
    @test result.quality[:capture_radius_min] >= 0.0
    @test result.robustness[:dynamics_residual] < 1e-4
    for (_, r) in result.quality[:nash_residual]
        @test r < 1e-3
    end
end
