using OrdinaryDiffEqOperatorSplitting
using Test
using DiffEqBase
using OrdinaryDiffEqLowOrderRK

function ode1(du, u, p, t)
    @. du = -0.1u
end
f1 = ODEFunction(ode1)
f1dofs = [1, 2, 3]

function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
end
f2 = ODEFunction(ode2)
f2dofs = [1, 3]

f = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))

# Next we can define the split problem.
u0 = [-1.0, 1.0, 0.0]
tspan = (0.0, 1.0)
dt = 0.1
alg = LieTrotterGodunov((Euler(), Euler()))

@testset "alias_u0=true in init function" begin
    u0 = [-1.0, 1.0, 0.0]
    prob = OperatorSplittingProblem(f, u0, tspan, alias_u0 = true)
    integrator = init(prob, alg; dt = dt)

    # When alias_u0=true, integrator.u should be the same object as u0
    @test integrator.u === u0

    # Modify integrator.u and check if u0 is also modified
    original_u0 = copy(u0)
    integrator.u[1] = 999.0
    @test u0[1] == 999.0  # u0 should be modified since it's aliased
end

@testset "alias_u0=false in init function" begin
    u0 = [-1.0, 1.0, 0.0]
    prob = OperatorSplittingProblem(f, u0, tspan, alias_u0 = false)
    integrator = init(prob, alg; dt = dt)

    # When alias_u0=false, integrator.u should be a copy, not the same object
    @test integrator.u !== u0
    @test integrator.u == u0  # But values should be equal

    # Modify integrator.u and check if u0 remains unchanged
    original_u0 = copy(u0)
    integrator.u[1] = 999.0
    @test u0[1] == original_u0[1]  # u0 should remain unchanged
    @test u0 == original_u0  # Entire u0 should be unchanged
end

@testset "default alias_u0 behavior" begin
    # Test default behavior when alias_u0 is not specified
    u0 = [-1.0, 1.0, 0.0]
    prob = OperatorSplittingProblem(f, u0, tspan)
    integrator = init(prob, alg; dt = dt)

    # Default behavior should be alias_u0=false (based on init function signature)
    @test integrator.u !== u0
    @test integrator.u == u0

    # Modify integrator.u and check if u0 remains unchanged
    original_u0 = copy(u0)
    integrator.u[1] = 999.0
    @test u0[1] == original_u0[1]
    @test u0 == original_u0
end

@testset "alias_u0 comparison test" begin
    # Test comparing both behaviors with same initial condition
    u0 = [-1.0, 1.0, 0.0]
    prob = OperatorSplittingProblem(f, u0, tspan)

    # Test alias_u0=true in init
    integrator_alias = init(prob, alg; dt = dt, alias_u0 = true)
    @test integrator_alias.u === u0

    # Reset u0 and test alias_u0=false
    u0[:] = [-1.0, 1.0, 0.0]
    integrator_copy = init(prob, alg; dt = dt, alias_u0 = false)
    @test integrator_copy.u !== u0
    @test integrator_copy.u == u0
end
