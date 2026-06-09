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
    @safetestset "Quality Assurance" include("qa/qa.jl")
end
