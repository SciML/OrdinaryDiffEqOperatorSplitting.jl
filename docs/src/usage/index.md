# Usage

## Minimal Example

For example, we can solve a simple split problem using the `Euler()` algorithm
for each subproblem with the `LieTrotterGodunov` algorithm, by defining a problem tree and an analogue solver tree via tuples:

```julia
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqOperatorSplitting
# This is the true, full ODE.
function ode_true(du, u, p, t)
    du .-= 0.1u
    du[1] -= 0.01u[3]
    du[3] -= 0.01u[1]
end

# This is the first operator of the ODE.
function ode1(du, u, p, t)
    @. du = -0.1u
end
f1 = ODEFunction(ode1)
f1dofs = [1, 2, 3]

# This is the second operator of the ODE.
function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
end
f2 = ODEFunction(ode2)
f2dofs = [1, 3]

# This defines the split of the ODE.
f = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))

# Next we can define the split problem.
u0 = [-1.0, 1.0, 0.0]
tspan = (0.0, 1.0)
prob = OperatorSplittingProblem(f, u0, tspan)

# And the time integration algorithm.
alg = LieTrotterGodunov(
    (Euler(), Euler())
)

# Right now OrdinaryDiffEqOperatorSplitting.jl does not implement the SciML solution interface,
# but we can only intermediate solutions via the iterator interface.
integrator = init(prob, alg, dt = 0.1)
for (u, t) in TimeChoiceIterator(integrator, 0.0:0.5:1.0)
    @show t, u
end
```
