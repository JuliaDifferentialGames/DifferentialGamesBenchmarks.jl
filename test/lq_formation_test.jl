using Test
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

@testset "LQ Formation end-to-end" begin
    case = lookup_case(CaseID(:LQFormation, :Nominal, v"1"))
    g = game(case)
    @test DifferentialGamesBase.n_players(g) == 2
    @test DifferentialGamesBase.is_lq_game(g)
    @test DifferentialGamesBase.required_capabilities(g) == [:LQGame, :UnconstrainedGame, :DiscreteTime]

    sol = DifferentialGamesBase.solve(g, FNELQ(); verbose = false)
    @test sol.converged
    @test sol.equilibrium_type == :FeedbackNash

    ev = lq_formation_evaluator()
    result = evaluate(ev, g, sol)

    @test result.validity[:constraints_satisfied]
    @test result.quality[:formation_error_final] < 1e-3
    @test result.robustness[:dynamics_residual] < 1e-8
    for (_, r) in result.quality[:nash_residual]
        @test r < 1e-8
    end

    @testset "via Experiment/run" begin
        experiment = Experiment(case, FNELQ(); evaluator = ev)
        res = run(experiment)
        @test res isa ExperimentResult
        @test res.validity[:constraints_satisfied]
    end
end
