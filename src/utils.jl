# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? DataStructures.FasterForward : DataStructures.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = DataStructures.BinaryHeap{FT, ordering}(tstops)

    if isnothing(saveat)
        saveat = [t0, tf]
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        saveat = tf > t0 ? saveat : -saveat
        saveat = [t0:saveat:tf..., tf]
    else
        saveat = collect(saveat)
    end
    saveat = DataStructures.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
end

"""
    need_sync(a, b)

Determines whether it is necessary to synchronize two objects with any
solution information.
"""
need_sync

need_sync(a::AbstractVector, b::AbstractVector) = true
need_sync(a::SubArray, b::AbstractVector) = a.parent !== b
need_sync(a::AbstractVector, b::SubArray) = a !== b.parent
need_sync(a::SubArray, b::SubArray) = a.parent !== b.parent

"""
    sync_vectors!(a, b)

Copies the information in object `b` into object `a`, if synchronization is necessary.
"""
function sync_vectors!(a, b)
    return if need_sync(a, b) && a !== b
        a .= b
    end
end

# ---------------------------------------------------------------------------
# forward_sync_subintegrator!
# ---------------------------------------------------------------------------
"""
    forward_sync_subintegrator!(outer_integrator, inner, solution_indices, sync)

Copy state from the outer integrator into the inner integrator before a
sub-step, and apply any external parameter synchronisation via `sync`.
"""
function forward_sync_subintegrator!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, solution_indices, sync
    )
    forward_sync_internal!(outer_integrator, inner_integrator, solution_indices)
    return forward_sync_external!(outer_integrator, inner_integrator, sync)
end

"""
    forward_sync_subintegrator! for SplitSubIntegrator

When the inner node is a `SplitSubIntegrator` we only need to copy the master
solution vector slice into its `u` (the `SplitSubIntegrator.u` is already a
view, but on a different device or after a rollback it may need refreshing).
"""
function forward_sync_subintegrator!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, solution_indices, sync
    )
    # Sync the view: master → sub.u  (noop if they already alias)
    @views uouter = outer_integrator.u[solution_indices]
    sync_vectors!(sub.u, uouter)
    sync_vectors!(sub.uprev, uouter)
    return forward_sync_external!(outer_integrator, sub, sync)
end

# ---------------------------------------------------------------------------
# backward_sync_subintegrator!
# ---------------------------------------------------------------------------
"""
    backward_sync_subintegrator!(outer_integrator, inner, solution_indices, sync)

Copy state from the inner integrator back into the outer integrator after a
sub-step, and apply any external parameter synchronisation via `sync`.
"""
function backward_sync_subintegrator!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, solution_indices, sync
    )
    backward_sync_internal!(outer_integrator, inner_integrator, solution_indices)
    return backward_sync_external!(outer_integrator, inner_integrator, sync)
end

function backward_sync_subintegrator!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, solution_indices, sync
    )
    @views uouter = outer_integrator.u[solution_indices]
    sync_vectors!(uouter, sub.u)
    return backward_sync_external!(outer_integrator, sub, sync)
end

# ---------------------------------------------------------------------------
# forward_sync_internal! / backward_sync_internal!
# ---------------------------------------------------------------------------
function forward_sync_internal!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, solution_indices
    )
    return nothing
end
function backward_sync_internal!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, solution_indices
    )
    return nothing
end

function forward_sync_internal!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, solution_indices
    )
    @views uouter = outer_integrator.u[solution_indices]
    sync_vectors!(inner_integrator.uprev, uouter)
    sync_vectors!(inner_integrator.u, uouter)
    return SciMLBase.u_modified!(inner_integrator, true)
end
function backward_sync_internal!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, solution_indices
    )
    @views uouter = outer_integrator.u[solution_indices]
    return sync_vectors!(uouter, inner_integrator.u)
end

# ---------------------------------------------------------------------------
# forward_sync_external! / backward_sync_external!
# ---------------------------------------------------------------------------
function forward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, sync::NoExternalSynchronization
    )
    return nothing
end
function forward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, sync::NoExternalSynchronization
    )
    return nothing
end
# SplitSubIntegrator has no parameters for now → no-op
function forward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, sync::NoExternalSynchronization
    )
    return nothing
end
function forward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, sync
    )
    return synchronize_solution_with_parameters!(outer_integrator, inner_integrator.p, sync)
end
function forward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, sync
    )
    # SplitSubIntegrator does not carry p for now; dispatch on sync type if needed
    return nothing
end

function backward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, sync::NoExternalSynchronization
    )
    return nothing
end
function backward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, sync::NoExternalSynchronization
    )
    return nothing
end
function backward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, sync::NoExternalSynchronization
    )
    return nothing
end
function backward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DEIntegrator, sync
    )
    return synchronize_solution_with_parameters!(outer_integrator, inner_integrator.p, sync)
end
function backward_sync_external!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, sync
    )
    return nothing
end

function synchronize_solution_with_parameters!(
        outer_integrator::OperatorSplittingIntegrator, p, sync
    )
    @warn "Outer synchronizer not dispatched for parameter type $(typeof(p)) with synchronizer type $(typeof(sync))." maxlog = 1
    return nothing
end
function synchronize_solution_with_parameters!(
        outer_integrator::OperatorSplittingIntegrator, p::NullParameters, sync
    )
    return nothing
end

# ---------------------------------------------------------------------------
# NOTE: build_solution_index_tree and build_synchronizer_tree are NO LONGER
# needed as standalone functions — the information is now embedded directly
# into each SplitSubIntegrator during build_subintegrator_tree_with_cache.
# They are kept here (no-ops returning nothing) only so that any external
# code that might call them does not hard-error.
# ---------------------------------------------------------------------------
function build_solution_index_tree(f::GenericSplitFunction)
    # Deprecated: solution index trees now live inside SplitSubIntegrator.
    return nothing
end

function build_synchronizer_tree(f::GenericSplitFunction)
    # Deprecated: synchronizer trees now live inside SplitSubIntegrator.
    return nothing
end
