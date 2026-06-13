using Test
using SafeTestsets
using SciMLTesting

run_tests(;
    core = () -> begin
        @safetestset "Operator Splitting API" include("operator_splitting_api.jl")
        @safetestset "Aliasing" include("alias_u0.jl")
        @safetestset "Consistency" include("consistency.jl")
        @safetestset "Explicit Imports" include("explicit_imports.jl")
    end,
    groups = Dict(
        # Declared env -> runs only for GROUP=="QA", never under "All"
        # (matches the original `if GROUP == "QA"` branch). The qa env's
        # [sources] table already points at the repo root, so developing the
        # root resolves to the same local PR-branch code.
        "QA" => (;
            env = joinpath(@__DIR__, "qa"),
            body = () -> begin
                @testset "Quality Assurance" begin
                    include("qa/qa.jl")
                end
            end,
        ),
    ),
)
