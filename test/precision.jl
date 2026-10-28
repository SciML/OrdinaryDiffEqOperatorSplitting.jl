using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqOperatorSplitting
using Test

@testset "Float32 time span" begin
    f(du, u, p, t) = @. du = -u
    split_f = GenericSplitFunction((ODEFunction(f),), (1:1,))
    prob = OperatorSplittingProblem(split_f, Float32[1.0], (0.0f0, 1.0f0))
    alg = LieTrotterGodunov((Euler(),))

    integrator = init(prob, alg; dt = 0.1f0)
    @test integrator.opts.dtmin === eps(Float32)
    @test integrator.opts.dtmax === 1.0f0
    step!(integrator)
    @test integrator.t === 0.1f0
    @test integrator.u ≈ Float32[0.9]

    # An explicitly given bound is converted to the time type.
    integrator = init(prob, alg; dt = 0.1f0, dtmin = 1.0e-3)
    @test integrator.opts.dtmin === 1.0f-3
end

@testset "inner_verbosity" begin
    @test OrdinaryDiffEqOperatorSplitting.inner_verbosity(true) isa OrdinaryDiffEqOperatorSplitting.DiffEqBase.DEVerbosity
    v = OrdinaryDiffEqOperatorSplitting.OperatorSplittingVerbosity(OrdinaryDiffEqOperatorSplitting.SciMLLogging.Standard())
    @test OrdinaryDiffEqOperatorSplitting.inner_verbosity(v) === v.inner_verbosity
    @test OrdinaryDiffEqOperatorSplitting.inner_verbosity(v.inner_verbosity) === v.inner_verbosity
end
