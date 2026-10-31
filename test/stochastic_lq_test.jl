using Test
using LinearAlgebra
using DifferentialGamesBenchmarks
using DifferentialGamesBase
using DifferentialGamesBaseSolvers

@testset "Stochastic LQ family" begin
    @testset "StochasticLQ/AdditiveNoise: SFNELQ expected cost vs simulation" begin
        case = lookup_case(CaseID(:StochasticLQ, :AdditiveNoise, v"1"))
        sg = game(case)
        @test sg isa DifferentialGamesBase.StochasticGameProblem
        sol = DifferentialGamesBase.solve(sg, SFNELQ())
        @test sol.converged && sol.solver_info[:status] === :solved
        z = stochastic_expected_cost_error(sg, sol; samples = 40000, seed = 11)
        @test all(abs.(z) .< 4)
    end

    @testset "ChanceConstrainedLQ/PositionBound: per-event vs trajectory-wide risk" begin
        case = lookup_case(CaseID(:ChanceConstrainedLQ, :PositionBound, v"1"))
        sg = game(case)
        sol = DifferentialGamesBase.solve(sg, CCiLQGames(mc_samples = 0))
        @test sol.converged
        rates = chance_violation_rates(sg, sol; samples = 40000, seed = 12)
        # the bound row is every second row (component 2); each must be ≈ ε/2 or below
        rows = 2:2:length(rates.per_event)
        ε_row = STOCHASTIC_LQ_CHANCE_RISK / 2
        @test maximum(rates.per_event[rows]) <= ε_row + 0.01
        @test maximum(rates.per_event[rows]) >= ε_row - 0.02      # active constraint
        # but per-event risk does not bound the trajectory-wide failure probability
        @test rates.any_violation > 2 * ε_row
        # lower-bound rows are never violated
        @test maximum(rates.per_event[1:2:end]) == 0
    end

    @testset "ChanceConstrainedLQ/JointBudget: union bound holds" begin
        case = lookup_case(CaseID(:ChanceConstrainedLQ, :JointBudget, v"1"))
        sg = game(case)
        sol = DifferentialGamesBase.solve(sg, CCiLQGames(mc_samples = 0))
        @test sol.converged
        @test sol.solver_info[:joint_risk_bound] ≈ STOCHASTIC_LQ_CHANCE_RISK
        rates = chance_violation_rates(sg, sol; samples = 40000, seed = 13)
        lo, hi = rates.any_violation_interval
        @test lo <= STOCHASTIC_LQ_CHANCE_RISK
        @test rates.any_violation <= STOCHASTIC_LQ_CHANCE_RISK + 0.01
    end
end
