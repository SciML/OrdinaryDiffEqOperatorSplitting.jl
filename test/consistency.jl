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

integrator1 = init(prob1, splitting_solver; dt = dt)
integrator2 = init(prob2, Euler(); dt = dt)

# Compare solutions at every 2*dt time point.
# Original test used TimeChoiceIterator (removed in DiffEqBase v7).
nsteps = Int(round((tspan[2] - tspan[1]) / dt))
@test integrator1.u ≈ integrator2.u
@test integrator1.t ≈ integrator2.t
for _ in 1:(nsteps ÷ 2)
    step!(integrator1)
    step!(integrator1)
    step!(integrator2)
    step!(integrator2)
    @test integrator1.u ≈ integrator2.u
    @test integrator1.t ≈ integrator2.t
end
@test integrator1.iter == integrator2.iter
