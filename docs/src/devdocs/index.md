# Developer documentation

## Synchronizers API

A key part of operator splitting algorithms is the synchronization logic. Parameters of one subproblem might need to be kept in sync with the solution of other subproblems and vice versa. To handle this efficiently OrdinaryDiffEqOperatorSplitting.jl provides a small set of utils.

```@docs
OrdinaryDiffEqOperatorSplitting.NoExternalSynchronization
OrdinaryDiffEqOperatorSplitting.forward_sync_subintegrator!
OrdinaryDiffEqOperatorSplitting.backward_sync_subintegrator!
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

## Adding Solvers

!!! warning
    
    The API is not stable yet and subject to breaking changes.

To add a new solver, define two structs -- one describing the algorithm, one for its
cache -- and dispatch three internal functions on them:

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
    if OS._child_failed(child)
        # Signal the failure and return immediately. `step_footer!` runs the failure
        # escalation protocol from here: retry at this node if it is adaptive,
        # otherwise escalate to the parent. Never keep stepping after a failure --
        # the state of the failed child is meaningless.
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

```@docs
OrdinaryDiffEqOperatorSplitting.alg_adaptive_order
```
