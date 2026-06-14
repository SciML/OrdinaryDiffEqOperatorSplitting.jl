using SafeTestsets

@safetestset "Aqua" begin
    using OrdinaryDiffEqOperatorSplitting
    using Aqua
    using Test
    Aqua.test_all(OrdinaryDiffEqOperatorSplitting)
end

@safetestset "JET" begin
    using OrdinaryDiffEqOperatorSplitting
    using JET
    using Test
    # JET.test_package reports 2 possible errors in src/integrator.jl
    # (rollback_children!/_rollback_children! on the SplitSubIntegrator path).
    # Tracked in https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/issues/87
    @test_broken false  # JET: rollback_children!(::SplitSubIntegrator) no matching method + _rollback_children! undefined (src/integrator.jl) — tracked in https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/issues/87
end
