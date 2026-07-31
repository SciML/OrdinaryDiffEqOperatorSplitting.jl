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

## Configuring individual subintegrators

`init` takes one value per keyword, which is not enough when the operators want
different treatment -- a stiff reaction term needs small steps while a diffusion term
is happy with large ones, or one operator should be integrated adaptively and another
at a fixed step size.

A [`TreeOption`](@ref) carries one value per node of the splitting tree instead. It is
built for a given splitting function, so it knows the shape of the tree and rejects
addresses that do not exist:

```julia
f_reaction = GenericSplitFunction((f_r1, f_r2), (r1dofs, r2dofs))
f = GenericSplitFunction((f_diffusion, f_reaction), (ddofs, rdofs))

dt = TreeOption(f, 1.0e-2)   # every node starts with the same value
```

Nodes are addressed either by their path or by a [`SplitNode`](@ref) minted from the
splitting function, where `f[2, 1]` is the first operator of the second operator of
`f`. Plain assignment sets a single node, broadcast assignment sets a node and
everything below it:

```julia
dt[2]      = 1.0e-4    # the reaction split node alone
dt[2]     .= 1.0e-4    # the reaction split node and both of its operators
dt[f[2]]  .= 1.0e-4    # the same, addressed by node
dt[f[2, 1]] = 1.0e-5   # just the first reaction operator

integrator = init(prob, alg; dt)
```

Assignments are applied in order, so a broadcast over a subtree overwrites anything
more specific written before it.

### Multi-rate integration

A node's `dt` is the step size it uses to traverse the interval its parent hands it.
Giving the reaction subtree a smaller `dt` therefore makes it subcycle: with an outer
step of `1.0e-2` and a reaction step of `1.0e-4`, the reaction operators take a hundred
steps per splitting step and still land exactly on the synchronization point.

Two things to keep in mind:

- Under `StrangMarchuk` a child is handed intervals of `Δt/2`, `Δt` and `Δt/2`. A step
  size that does not divide them leaves a short final sub-step in each interval, so the
  sub-steps are not all the same length. Step sizes that are exactly representable
  (powers of two, say) avoid this.
- A node's `dt` larger than the interval it is handed is clipped to that interval.

For an adaptive node the configured `dt` is only the initial step size.

### Mixing adaptive and fixed-step operators

Adaptivity is configured the same way:

```julia
adaptive = TreeOption(f, false)
adaptive[f[2, 1]] = true
adaptive[f[2, 2]] = true

integrator = init(prob, alg; dt = 1.0e-2, adaptive)
```

Note that the splitting nodes themselves stay non-adaptive here. Broadcasting
`adaptive[2] .= true` would also mark the reaction *split node* as adaptive, and since
`LieTrotterGodunov` is not an adaptive algorithm that produces a warning.

Any other keyword accepted by the inner integrators can be given per node as well and
is passed down to the leaves, while keywords a splitting node understands (`dtmin`,
`dtmax`, `failfactor`) are applied at every level:

```julia
reltol = TreeOption(f, 1.0e-3)
reltol[f[2, 1]] = 1.0e-9

integrator = init(prob, alg; dt = 1.0e-2, adaptive, reltol)
```

### Reading the tree back

A `SplitNode` addresses the same position in every tree that mirrors the splitting
function, so it resolves against the algorithm and the integrator too:

```julia
f[f[2, 1]]            # the sub function
alg[f[2, 1]]          # the inner algorithm
integrator[f[2, 1]]   # the sub integrator
```

`integrator[i]` is *not* available for this: SciMLBase already gives integer indexing
of an integrator the meaning "the `i`-th state component".

Calling `reinit!` without a `dt` restores every node to its configured step size, so a
multi-rate setup survives. Passing a `dt` reconfigures the tree exactly as at `init`:
a single value applies to every node, a `TreeOption` node by node.
