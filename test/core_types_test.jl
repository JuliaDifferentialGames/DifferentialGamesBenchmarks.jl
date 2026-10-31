using Test
using LinearAlgebra
using DifferentialGamesBenchmarks
using DifferentialGamesBase

@testset "CaseID" begin
    id = CaseID(:SimplePursuit, :Nominal, v"1")
    @test string(id) == "SimplePursuit.Nominal_v1"
    @test id == CaseID(:SimplePursuit, :Nominal, v"1")
    @test id != CaseID(:SimplePursuit, :Nominal, v"2")
end

@testset "BenchmarkCase" begin
    case = SIMPLE_PURSUIT_NOMINAL_V1
    @test case.id == CaseID(:SimplePursuit, :Nominal, v"1")
    g1 = game(case)
    g2 = game(case)
    @test g1 !== g2  # fresh instance every call
    @test g1 isa DifferentialGamesBase.GameProblem{Float64}
end

@testset "ChallengeProfile" begin
    @test NOMINAL_CHALLENGE.name == :Nominal
    @test isempty(NOMINAL_CHALLENGE.transforms)
    g = game(SIMPLE_PURSUIT_NOMINAL_V1)
    @test apply(NOMINAL_CHALLENGE, g) === g

    noisy = ChallengeProfile(:Noisy, [g -> add_process_noise(g, Matrix{Float64}(I, 4, 4))])
    sg = apply(noisy, g)
    @test sg isa DifferentialGamesBase.StochasticGameProblem
end

@testset "AccessSpec" begin
    g = game(SIMPLE_PURSUIT_NOMINAL_V1)
    @test access(FullModelAccess(), g) === g
    perturbed = access(ApproximateModelAccess(identity), g)
    @test perturbed === g
    @test_throws ErrorException access(ModelFreeAccess(), g)
end

@testset "Registry" begin
    @test lookup_case(CaseID(:SimplePursuit, :Nominal, v"1")) === SIMPLE_PURSUIT_NOMINAL_V1
    @test_throws KeyError lookup_case(CaseID(:NotACase, :Nominal, v"1"))
end

@testset "Budget" begin
    b = Budget()
    @test b.max_wall_time_s == 60.0
    @test b.max_iterations === nothing
    b2 = Budget(max_wall_time_s = 10.0, max_iterations = 5)
    @test b2.max_wall_time_s == 10.0
    @test b2.max_iterations == 5
end
