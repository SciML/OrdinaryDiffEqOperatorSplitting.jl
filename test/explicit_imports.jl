using ExplicitImports
using OrdinaryDiffEqOperatorSplitting
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(OrdinaryDiffEqOperatorSplitting) === nothing
    @test check_no_stale_explicit_imports(OrdinaryDiffEqOperatorSplitting) === nothing
end
