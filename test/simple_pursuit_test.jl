using Test
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

# ALGAMES, not iLQGames, is the mathematically compatible solver here: Simple
# Pursuit is built via the PD-GNEP path (SeparableDynamics), and only ALGAMES
# declares `:SeparableDynamics` in its `solver_capabilities` — iLQGames does
# not, and errors on this game's per-player dynamics assembly.
@testset "Simple Pursuit end-to-end" begin
    case = lookup_case(CaseID(:SimplePursuit, :Nominal, v"1"))
    g = game(case)
    @test DifferentialGamesBase.n_players(g) == 2

    sol = DifferentialGamesBase.solve(g, ALGAMES(); verbose = false)
    @test sol.converged

    ev = simple_pursuit_evaluator()
    result = evaluate(ev, g, sol)

    @test result.validity[:constraints_satisfied]
    @test result.quality[:capture_radius_min] >= 0.0
    @test result.quality[:time_to_capture] > 0.0
    @test result.robustness[:dynamics_residual] < 1e-6
    for (_, r) in result.quality[:nash_residual]
        @test r < 1e-4
    end

    @testset "via Experiment/run" begin
        experiment = Experiment(case, ALGAMES(); evaluator = ev)
        res = run(experiment)
        @test res isa ExperimentResult
        @test res.wall_time_s > 0.0
        @test res.validity[:constraints_satisfied]
    end
end
