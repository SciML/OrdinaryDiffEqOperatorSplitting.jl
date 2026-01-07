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

To add a new solver just define two new structs, one for the algorithm description and one for the algorithm cache and dispatch internal functions, as follows:

```julia
using SciMLBase, OrdinaryDiffEqOperatorSplitting
struct MySimpleFirstOrderAlgorithm{InnerAlgorithmTypes} <:
       OrdinaryDiffEqOperatorSplitting.AbstractOperatorSplittingAlgorithm
    inner_algs::InnerAlgorithmTypes # Tuple of solver for the problem sequence
end

struct MySimpleFirstOrderCache{uType, uprevType, iiType} <:
       OrdinaryDiffEqOperatorSplitting.AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    inner_caches::iiType
end

function OrdinaryDiffEqOperatorSplitting.init_cache(
        f::GenericSplitFunction, alg::MySimpleFirstOrderAlgorithm;
        uprev::AbstractArray, u::AbstractVector,
        inner_caches,
        alias_uprev = true,
        alias_u = false
)
    @assert length(inner_caches) == 2
    _uprev = alias_uprev ? uprev : SciMLBase.recursivecopy(uprev)
    _u = alias_u ? u : SciMLBase.recursivecopy(u)
    return MySimpleFirstOrderAlgorithmCache(_u, _uprev, inner_caches)
end

@inline function OrdinaryDiffEqOperatorSplitting.advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator, subintegrators::Tuple,
        solution_indices::Tuple, synchronizers::Tuple,
        cache::MySimpleFirstOrderAlgorithmCache, tnext)
    # We assume that the integrators are already synced
    (;inner_caches) = cache

    # Advance first subproblem
    OrdinaryDiffEqOperatorSplitting.forward_sync_subintegrator!(
        outer_integrator, subintegrators[1], solution_indices[1], synchronizers[1])
    OrdinaryDiffEqOperatorSplitting.advance_solution_to!(
        outer_integrator, subintegrators[1], solution_indices[1],
        synchronizers[1], inner_caches[1], tnext)
    if subintegrators[1].sol.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return
    end
    OrdinaryDiffEqOperatorSplitting.backward_sync_subintegrator!(
        outer_integrator, subintegrators[1], solution_indices[1], synchronizers[1])

    # Advance second subproblem
    OrdinaryDiffEqOperatorSplitting.forward_sync_subintegrator!(
        outer_integrator, subintegrators[2], solution_indices[2], synchronizers[2])
    OrdinaryDiffEqOperatorSplitting.advance_solution_to!(
        outer_integrator, subintegrators[2], solution_indices[2],
        synchronizers[2], inner_caches[2], tnext)
    if subintegrators[2].sol.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return
    end
    OrdinaryDiffEqOperatorSplitting.backward_sync_subintegrator!(
        outer_integrator, subintegrators[2], solution_indices[2], synchronizers[2])

    # Done :)
end
```
