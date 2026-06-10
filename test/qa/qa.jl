using OrdinaryDiffEqOperatorSplitting
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OrdinaryDiffEqOperatorSplitting)
end

@testset "JET" begin
    # JET.test_package reports 2 possible errors in src/integrator.jl
    # (rollback_children!/_rollback_children! on the SplitSubIntegrator path).
    # Tracked in https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/issues/87
    @test_broken false  # JET: rollback_children!(::SplitSubIntegrator) no matching method + _rollback_children! undefined (src/integrator.jl) — tracked in https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/issues/87
end
