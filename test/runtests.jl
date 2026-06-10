using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @safetestset "Operator Splitting API" include("operator_splitting_api.jl")
    @safetestset "Aliasing" include("alias_u0.jl")
    @safetestset "Consistency" include("consistency.jl")
    @safetestset "Explicit Imports" include("explicit_imports.jl")
end

if GROUP == "QA"
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.instantiate()
    @testset "Quality Assurance" begin
        include("qa/qa.jl")
    end
end
