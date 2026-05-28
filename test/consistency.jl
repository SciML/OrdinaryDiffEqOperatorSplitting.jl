using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqOperatorSplitting
using Test

@testset "Baseline consistency" begin

    f(du, u, p, t) = @. du = -u
    tspan = (0.0, 2.0)
    u0 = [1.0]
    dt = 0.1

    cell_indices = 1:1
    split_f = GenericSplitFunction((ODEFunction(f),), (cell_indices,))
    prob1 = OperatorSplittingProblem(split_f, u0, tspan)
    prob2 = ODEProblem(f, u0, tspan)
    splitting_solver = LieTrotterGodunov((Euler(),))

    integrator1 = init(prob1, splitting_solver; dt = dt)
    integrator2 = init(prob2, Euler(); dt = dt)
    for ((u1, t1), (u2, t2)) in zip(TimeChoiceIterator(integrator1, tspan[1]:(2dt):tspan[2]), TimeChoiceIterator(integrator2, tspan[1]:(2dt):tspan[2]))
        @test u1 ≈ u2
        @test t1 ≈ t2
    end
    @test integrator1.iter == integrator2.iter

end

@testset "Respect FSAL (#79)" begin
    mutable struct CR{F}
        f::F; n::Int
    end
    (c::CR)(du, u, p, t) = (c.n += 1; c.f(du, u, p, t))
    a = CR((du, u, p, t) -> (du .= -0.1 .* A * u), 0)
    b = CR((du, u, p, t) -> (du .= -u .+ 0.01 / length(u) .* sum(abs2, u)), 0)

    fsplit = GenericSplitFunction((ODEFunction(a), ODEFunction(b)), (dofs, dofs))
    alg = StrangMarchuk((Euler(), Euler()))
    integ = init(
        OperatorSplittingProblem(fsplit, copy(u0), (0.0, 1.0e6)), alg;
        dt = dt, adaptive = false, alias_u0 = false, verbose = false
    )
    a.n = 0; b.n = 0
    step!(integ)
    @test a.n == 2 * 2 # 2 solves á 2 f calls
    @test b.n == 1 * 2 # 1 solve  á 2 f calls
    a.n = 0; b.n = 0
    step!(integ)
    @test a.n == 1 * 1 + 1 * 2 # 2 solves, where the first call respects FSAL
    @test b.n == 1 * 2 # 1 solve  á 2 f calls
end
