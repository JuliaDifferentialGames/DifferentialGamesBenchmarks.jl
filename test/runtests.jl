using DifferentialGamesBenchmarks
using Test

@testset "DifferentialGamesBenchmarks.jl" begin
    include("core_types_test.jl")
    include("simple_pursuit_test.jl")
    include("homicidal_chauffeur_test.jl")
    include("two_cars_test.jl")
    include("lady_in_the_lake_test.jl")
    include("lq_formation_test.jl")
    include("stochastic_lq_test.jl")
    include("falcon_cases_test.jl")
    include("viz_smoke_test.jl")
end
