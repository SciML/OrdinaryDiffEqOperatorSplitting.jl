# Developer Documentation

!!! warning "Developer extension interface v1"
    These interfaces are for packages implementing operator-splitting solvers
    or synchronizers. They are not end-user APIs. Version 1 may change only in
    a breaking release of OrdinaryDiffEqOperatorSplitting.jl.

## Solver Extensions

A solver extension defines an algorithm type with an `inner_algs` tuple, a
concrete cache, `init_cache`, and `_perform_step!`. The tuple shape must mirror
the corresponding `GenericSplitFunction`. A step implementation synchronizes
each child before and after advancing it, and sets `parent.force_stepfail` if a
child reports failure.

```@docs
OrdinaryDiffEqOperatorSplitting.AbstractOperatorSplittingAlgorithm
OrdinaryDiffEqOperatorSplitting.AbstractOperatorSplittingCache
OrdinaryDiffEqOperatorSplitting.init_cache
OrdinaryDiffEqOperatorSplitting._perform_step!
```

The following sweeps the operators in reverse order, which is a complete
first-order splitting in its own right (it coincides with `LieTrotterGodunov`
exactly when the operators commute):

```julia
import OrdinaryDiffEqOperatorSplitting as OS

struct ReverseLieTrotterGodunov{AlgTupleType} <: OS.AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType
end

struct ReverseLieTrotterGodunovCache{uType, uprevType} <: OS.AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
end

function OS.init_cache(
        f::GenericSplitFunction, alg::ReverseLieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
    )
    return ReverseLieTrotterGodunovCache(u, uprev)
end

function OS._perform_step!(
        parent, children::Tuple, cache::ReverseLieTrotterGodunovCache, dt
    )
    for i in reverse(eachindex(children))
        child = children[i]
        idxs = parent.child_solution_indices[i]
        sync = parent.child_synchronizers[i]

        OS.forward_sync_subintegrator!(parent, child, idxs, sync)
        OS.advance_solution_by!(parent, child, dt)
        if OS._child_failed(child)
            parent.force_stepfail = true
            return nothing
        end
        OS.backward_sync_subintegrator!(parent, child, idxs, sync)
    end
    return nothing
end
```

The algorithm is then used like any built-in one, with one inner algorithm per
leaf operator:

```julia
alg = ReverseLieTrotterGodunov((Euler(), Euler()))
sol = solve(OperatorSplittingProblem(f, u0, tspan), alg; dt = 0.01)
```

## Synchronizer Extensions

Synchronizers communicate parameters and solution buffers between parent and
child integrators. Define the external synchronization methods for the custom
synchronizer type, then supply one synchronizer for every operator in a
`GenericSplitFunction`.

```@docs
OrdinaryDiffEqOperatorSplitting.NoExternalSynchronization
OrdinaryDiffEqOperatorSplitting.forward_sync_subintegrator!
OrdinaryDiffEqOperatorSplitting.backward_sync_subintegrator!
OrdinaryDiffEqOperatorSplitting.need_sync
OrdinaryDiffEqOperatorSplitting.sync_vectors!
```
