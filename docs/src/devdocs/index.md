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
implementing them is optional. Note the accuracy consequences: with the linear
fallback, interpolated output is first order even for a second-order scheme, and
continuous callback event times are the exact roots of a straight chord across the
step. Implementing a higher-order interpolant improves saved output *and* event
location at once.

The reason the fallback is only linear is structural rather than incidental. A
splitting step advances its children sequentially over staggered subintervals, so at
any time strictly inside the step the children's own interpolants describe different
sub-problems evaluated over different intervals, and they do not compose into an
approximation of the split solution. Only the step endpoints, which the outer
integrator owns, are states of the full split system.

The same structural argument is why saving and callbacks live on the outer integrator
alone: an inner split is a stage, not a step, so there is no time point at which its
state is a meaningful approximation of the split solution for a saved point to record
or for a callback condition to act on.

Two invariants matter when implementing this:

  - `Θ` is derived from `integrator.t - integrator.tprev`, **not** from
    `integrator.dt`. Once a step is accepted, `step_accept_controller!` has already
    replaced `dt` with the step size proposed for the *next* step.
  - Interpolation must not mutate integrator state. `change_t_via_interpolation!`
    relies on being able to evaluate the interpolant repeatedly (the callback
    root-finder does so many times per step) before committing to a time.
