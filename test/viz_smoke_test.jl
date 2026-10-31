using Test
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers
using CairoMakie

@testset "Viz extension smoke test" begin
    @test Base.get_extension(DifferentialGamesBenchmarks, :DifferentialGamesBenchmarksVizExt) !== nothing

    case = lookup_case(CaseID(:SimplePursuit, :Nominal, v"1"))
    g = game(case)
    sol = DifferentialGamesBase.solve(g, ALGAMES(); verbose = false)

    fig = gameplot(g, sol)
    @test fig isa Makie.Figure

    fig_case = gameplot(case, sol)
    @test fig_case isa Makie.Figure

    fig_debug = gameplot(case, sol; preset = :debug)
    @test fig_debug isa Makie.Figure

    fig_unknown = gameplot(case, sol; preset = :not_a_real_preset)
    @test fig_unknown isa Makie.Figure

    frame = renderframe(case, sol, 1.0)
    @test frame isa Makie.Figure

    experiment = Experiment(case, ALGAMES(); evaluator = simple_pursuit_evaluator())
    result = run(experiment)
    dash = dashboard(result)
    @test dash isa Makie.Figure
end

@testset "Viz extension — all Classical Games cases render" begin
    cases_and_evaluators = [
        (lookup_case(CaseID(:SimplePursuit, :Nominal, v"1")), simple_pursuit_evaluator()),
        (lookup_case(CaseID(:HomicidalChauffeur, :Nominal, v"1")), homicidal_chauffeur_evaluator()),
        (lookup_case(CaseID(:GameOfTwoCars, :Nominal, v"1")), game_of_two_cars_evaluator()),
        (lookup_case(CaseID(:LadyInTheLake, :Nominal, v"1")), lady_in_the_lake_evaluator()),
    ]
    for (case, ev) in cases_and_evaluators
        @testset "$(case.id)" begin
            g = game(case)
            sol = DifferentialGamesBase.solve(g, ALGAMES(); verbose = false)
            @test gameplot(case, sol) isa Makie.Figure
            @test renderframe(case, sol, g.time_horizon.tf / 2) isa Makie.Figure

            experiment = Experiment(case, ALGAMES(); evaluator = ev)
            result = run(experiment)
            @test dashboard(result) isa Makie.Figure
        end
    end
end

@testset "Viz extension — LQ Formation renders" begin
    case = lookup_case(CaseID(:LQFormation, :Nominal, v"1"))
    g = game(case)
    sol = DifferentialGamesBase.solve(g, FNELQ(); verbose = false)
    @test gameplot(case, sol) isa Makie.Figure
    @test renderframe(case, sol, g.time_horizon.tf / 2) isa Makie.Figure

    experiment = Experiment(case, FNELQ(); evaluator = lq_formation_evaluator())
    result = run(experiment)
    @test dashboard(result) isa Makie.Figure
end
