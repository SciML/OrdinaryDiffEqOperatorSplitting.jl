# Developer documentation

!!! warning
    This page specifies developer extension APIs for packages implementing
    operator-splitting solvers. They are versioned for the OrdinaryDiffEq ecosystem,
    but are not supported end-user APIs. Application code should use the APIs in the
    API Reference instead.

## Synchronizers API

A key part of operator splitting algorithms is the synchronization logic. Parameters of one subproblem might need to be kept in sync with the solution of other subproblems and vice versa. To handle this efficiently OrdinaryDiffEqOperatorSplitting.jl provides a small set of utils.

```@docs
OrdinaryDiffEqOperatorSplitting.NoExternalSynchronization
OrdinaryDiffEqOperatorSplitting.forward_sync_subintegrator!
OrdinaryDiffEqOperatorSplitting.backward_sync_subintegrator!
OrdinaryDiffEqOperatorSplitting.forward_sync_external!
OrdinaryDiffEqOperatorSplitting.backward_sync_external!
OrdinaryDiffEqOperatorSplitting.need_sync
OrdinaryDiffEqOperatorSplitting.sync_vectors!
```

## Adding Synchronizers

!!! warning
    
    The API is not stable yet and subject to breaking changes.

You need to provide dispatches for

```@docs; canonical=false
OrdinaryDiffEqOperatorSplitting.forward_sync_subintegrator!
OrdinaryDiffEqOperatorSplitting.backward_sync_subintegrator!
```

with your custom synchronizer object and add it to the split function construction as follows:

```julia
f1, f2 = generate_individual_functions() # assuming 3 unknowns each
i1, i2 = generate_solution_indices()     # e.g. ([1,2,3], Int[])
synchronizer_tree = generate_my_synchronizer_tree() # e.g. (MySynchronizer([1,2,3]), NoExternalSynchronization())
f = GenericSplitFunction((f1, f2), (i1, i2), synchronizer_tree)
u0 = [-1.0, 1.0, 0.0]
tspan = (0.0, 1.0)
prob = OperatorSplittingProblem(f, u0, tspan)
```

## Solver extension API

```@docs
OrdinaryDiffEqOperatorSplitting.AbstractOperatorSplittingAlgorithm
OrdinaryDiffEqOperatorSplitting.AbstractOperatorSplittingCache
OrdinaryDiffEqOperatorSplitting.init_cache
OrdinaryDiffEqOperatorSplitting._perform_step!
OrdinaryDiffEqOperatorSplitting.advance_solution_by!
OrdinaryDiffEqOperatorSplitting.child_failed
OrdinaryDiffEqOperatorSplitting.alg_adaptive_order
OrdinaryDiffEqOperatorSplitting.splitting_interpolant
OrdinaryDiffEqOperatorSplitting.splitting_interpolant!
```

## Adding solvers

!!! warning
    
    This is a developer extension API. It may change independently of the end-user
    API and should not be used in application code.

To add a new solver, define two structs -- one describing the algorithm, one for its
cache -- and dispatch the developer extension functions on them:

- `init_cache(f, alg; uprev, u)` builds the cache for one node of the splitting tree.
- `_perform_step!(parent, children, cache, dt)` advances that node by `dt`.
- `alg_adaptive_order(alg)`, only if the algorithm is adaptive (see below).

The algorithm struct has to carry the inner algorithms of the problem sequence in a
field named `inner_algs`, because that is how the tree of integrators is built
alongside the tree of split functions. The cache is where a scheme keeps the buffers
it needs beyond `u`/`uprev`.

```julia
using SciMLBase, OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS

struct MySimpleFirstOrderAlgorithm{InnerAlgorithmTypes} <:
    OS.AbstractOperatorSplittingAlgorithm
    inner_algs::InnerAlgorithmTypes # Tuple of solvers for the problem sequence
end

struct MySimpleFirstOrderCache{uType, uprevType} <: OS.AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
end

function OS.init_cache(
        f::GenericSplitFunction, alg::MySimpleFirstOrderAlgorithm;
        uprev::AbstractArray, u::AbstractVector
    )
    # `u` and `uprev` are the buffers of *this* node; the integrator owns them and
    # the cache only keeps references. Allocate additional buffers with
    # `similar(u)` if the scheme needs them.
    return MySimpleFirstOrderCache(u, uprev)
end
```

The stepping function receives the node whose step is being performed (`parent`) and
the tuple of its child integrators, which may be leaf `DEIntegrator`s or nested
`SplitSubIntegrator`s -- the same code handles both. Everything else a step needs is
reachable from `parent`: `parent.child_solution_indices[i]` are the indices of the
`i`-th child in this node's solution vector, and `parent.child_synchronizers[i]` is its
synchronizer.

Advancing one child means synchronizing into it, stepping it, and synchronizing back:

```julia
function advance_one_child!(parent, child, i, dt)
    idxs = parent.child_solution_indices[i]
    sync = parent.child_synchronizers[i]

    OS.forward_sync_subintegrator!(parent, child, idxs, sync)
    OS.advance_solution_by!(parent, child, dt)
    if OS.child_failed(child)
        # A failed child must stop the remaining stages of this step.
        parent.force_stepfail = true
        return
    end
    OS.backward_sync_subintegrator!(parent, child, idxs, sync)
    return
end

function OS._perform_step!(
        parent, children::Tuple, cache::MySimpleFirstOrderCache, dt
    )
    advance_one_child!(parent, children[1], 1, dt)
    parent.force_stepfail && return

    advance_one_child!(parent, children[2], 2, dt)
    parent.force_stepfail && return

    # Done :) The solution of the step is in `parent.u`; `parent.uprev` holds the
    # state at the beginning of the step and must be left untouched, as rollback
    # after a rejected step restores from it.
    return
end
```

This example is written for exactly two operators. A scheme that works for any number
of them loops over `children` with `Unrolled.@unroll`, as the built-in algorithms in
`src/solver.jl` do; a plain `for` loop over the heterogeneously typed tuple would be
type unstable.

### Adaptive algorithms

A splitting node runs a step size controller only if its algorithm both declares
itself adaptive and produces an error estimate. Such an algorithm has to

1. define `SciMLBase.isadaptive(::MyAlgorithm) = true`,
2. define [`OrdinaryDiffEqOperatorSplitting.alg_adaptive_order`](@ref), the order of
   its error estimator, and
3. write the tolerance-scaled error estimate to `parent.EEst` at the end of
    `_perform_step!`, following the OrdinaryDiffEq convention that a step is accepted
    when `EEst <= 1`.

Only do the last step when `parent.controller_cache !== nothing`: the node may have
been configured non-adaptive, in which case no controller consumes the estimate. The
tolerances and the norm to scale with live in `parent.opts`:

```julia
if parent.controller_cache !== nothing
    (; abstol, reltol, internalnorm) = parent.opts
    @. residual = error_of_the_step / (abstol + max(abs(parent.u), abs(parent.uprev)) * reltol)
    parent.EEst = internalnorm(residual, parent.t + dt)
end
```

See [`PalindromicPairLieTrotterGodunov`](@ref) in `src/solver.jl` for a complete
example, and [Adaptive time stepping](@ref) for how the two layers of adaptivity
interact.

## Dense output

Saving, `saveat` and continuous callback root-finding all go through a single hook,
so an algorithm only has to describe how to interpolate *within one of its steps*:

```julia
function OrdinaryDiffEqOperatorSplitting.splitting_interpolant(
        integrator, cache::MySimpleFirstOrderCache, Θ, dt, y₀, y₁, idxs, ::Type{Val{D}}
) where {D}
    # Θ = (t - tprev) / dt is the step-local coordinate, y₀ = u(tprev), y₁ = u(t).
    # ...
end

function OrdinaryDiffEqOperatorSplitting.splitting_interpolant!(
        out, integrator, cache::MySimpleFirstOrderCache, Θ, dt, y₀, y₁, idxs,
        ::Type{Val{D}}
) where {D}
    # In-place variant; must return `out`.
end
```

Both fall back to linear interpolation for any `AbstractOperatorSplittingCache`, so
implementing them is optional -- but with the fallback, saved output is first order
even for a second-order scheme, and continuous callback event times are the exact
roots of a straight chord across the step. One method improves both.

The fallback is linear for a structural reason, and it is the same reason saving and
callbacks live on the outer integrator alone: a splitting step advances its children
sequentially over staggered subintervals, so a child's own interpolant describes a
different sub-problem over a different interval, and they do not compose into an
approximation of the split solution. Only the step endpoints, which the outer
integrator owns, are states of the full split system -- an inner split is a stage,
not a step.

When implementing this, note that `Θ` is derived from
`integrator.t - integrator.tprev`, **not** from `integrator.dt`: once a step is
accepted, `step_accept_controller!` has already replaced `dt` with the step size
proposed for the *next* step.
