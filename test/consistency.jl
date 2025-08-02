using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqOperatorSplitting
using Test

f(du, u, p, t) = @. du = -u
tspan = (0.0, 2.0)
u0 = [1.0]
dt = 0.1

cell_indices = 1:1
split_f = GenericSplitFunction((ODEFunction(f),), (cell_indices,))
prob1 = OperatorSplittingProblem(split_f, u0, tspan)
prob2 = ODEProblem(f, u0, tspan)
splitting_solver = LieTrotterGodunov((Euler(),))

integrator1 = init(prob1, splitting_solver; dt=dt)
integrator2 = init(prob2, Euler(); dt=dt)
for ((u1, t1), (u2, t2)) in zip(TimeChoiceIterator(integrator1, tspan[1]:(2dt):tspan[2]), TimeChoiceIterator(integrator2, tspan[1]:(2dt):tspan[2]))
    @test u1 ≈ u2
    @test t1 ≈ t2
end
@test integrator1.iter == integrator2.iter
