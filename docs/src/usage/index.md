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
    return du[3] -= 0.01u[1]
end

# This is the first operator of the ODE.
function ode1(du, u, p, t)
    return @. du = -0.1u
end
f1 = ODEFunction(ode1)
f1dofs = [1, 2, 3]

# This is the second operator of the ODE.
function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    return du[2] = -0.01u[1]
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
# but we can obtain intermediate solutions via the iterator interface.
integrator = init(prob, alg, dt = 0.1)
for (u, t) in TimeChoiceIterator(integrator, 0.0:0.5:1.0)
    @show t, u
end
```

For second-order accuracy, use the `StrangMarchuk` algorithm instead.
It performs the symmetric palindromic splitting
A₁(Δt/2) → … → Aₙ(Δt) → … → A₁(Δt/2):

```julia
alg = StrangMarchuk(
    (Euler(), Euler())
)

integrator = init(prob, alg, dt = 0.1)
for (u, t) in TimeChoiceIterator(integrator, 0.0:0.5:1.0)
    @show t, u
end
```

## Multirate splitting

Operators with different stiffness or cost can use different substep sizes via
the `inner_dts` keyword on `init`, `solve`, and `reinit!`. Each entry sets the
corresponding child sub-integrator's `dt`; `nothing` at an index falls back to
the outer `dt`.

```julia
# Reuse f, prob from the minimal example above. Suppose ode1 is cheap but
# CFL-limited at a small dt, while ode2 is stable at a large dt.
alg = StrangMarchuk((Euler(), Euler()))

dt_outer = 0.1
dt_inner = 0.02       # ode1 subcycles 5 substeps per outer Strang half-step

sol = solve(
    prob, alg;
    dt = dt_outer,
    inner_dts = (dt_inner, dt_outer),
    adaptive = false,
)
```

Default `inner_dts = nothing` preserves the existing single-rate behaviour (the
outer `dt` is broadcast to every child). A `Tuple` activates multirate. For
non-adaptive inner algorithms the entry becomes the fixed substep size; for
adaptive inner algorithms (e.g. `Tsit5`) it acts as a starting hint via
`set_proposed_dt!`. The same kwarg works on `reinit!`: re-pass the Tuple to
preserve the multirate configuration. Nested split children (a
`GenericSplitFunction` as a child) are explicitly rejected with a clear
`ArgumentError`.
