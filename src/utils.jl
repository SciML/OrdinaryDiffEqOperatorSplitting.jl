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
        # We do not need to filter saveat like tstops because the saving
        # callback will ignore any times that are not between t0 and tf.
        saveat = collect(saveat)
    end
    saveat = DataStructures.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
end

"""
    need_sync(a, b)

Determines whether it is necessary to synchronize two objects with any
solution information. A possible reason when no synchronization is necessary
might be that the vectors alias each other in memory.
"""
need_sync

need_sync(a::AbstractVector, b::AbstractVector) = true
need_sync(a::SubArray, b::AbstractVector) = a.parent !== b
need_sync(a::AbstractVector, b::SubArray) = a !== b.parent
need_sync(a::SubArray, b::SubArray) = a.parent !== b.parent

"""
    sync_vectors!(a, b)

Copies the information in `b` into `a` if synchronization is necessary.
"""
function sync_vectors!(a, b)
    if need_sync(a, b) && a !== b
        a .= b
    end
    return nothing
end

"""
    forward_sync_subintegrator!(parent_integrator::OperatorSplittingIntegrator, inner_integrator::DEIntegrator, solution_indices, sync)

This function is responsible of copying the solution and parameters of the parent integrator and the synchronized subintegrators with the information given into the inner integrator.
If the inner integrator is synchronized with other inner integrators using `sync`, the function `forward_sync_external!` shall be dispatched for `sync`.
The `sync` object is passed from the outside and is the main entry point to dispatch custom types on for parameter synchronization.
The `solution_indices` are indices into the parent integrators solution vectors.
"""

function forward_sync_subintegrator!(
        parent::AnySplitIntegrator,
        child::DEIntegrator,
        solution_indices,
        sync
    )
    forward_sync_internal!(parent.u, child, solution_indices)
    forward_sync_external!(parent, child, sync)
    return nothing
end

# Shared internal helper: copy master u slice → leaf DEIntegrator u/uprev
function forward_sync_internal!(u_source, child::DEIntegrator, solution_indices)
    @views usrc = u_source[solution_indices]
    sync_vectors!(child.u, usrc)
    sync_vectors!(child.uprev, child.u)
    SciMLBase.u_modified!(child, true)
    return nothing
end


"""
    backward_sync_subintegrator!(parent_integrator::OperatorSplittingIntegrator, inner_integrator::DEIntegrator, solution_indices, sync)

This function is responsible of copying the solution of the inner integrator back into parent integrator and the synchronized subintegrators.
If the inner integrator is synchronized with other inner integrators using `sync`, the function `backward_sync_external!` shall be dispatched for `sync`.
The `sync` object is passed from the outside and is the main entry point to dispatch custom types on for parameter synchronization.
The `solution_indices` are indices in the parent integrators solution vectors.
"""

function backward_sync_subintegrator!(
        parent::AnySplitIntegrator,
        child::DEIntegrator,
        solution_indices,
        sync
    )
    @views udst = parent.u[solution_indices]
    sync_vectors!(udst, child.u)
    backward_sync_external!(parent, child, sync)
    return nothing
end

# ---------------------------------------------------------------------------
# forward_sync_external! / backward_sync_external!
# These handle parameter synchronisation via the `sync` object.
# ---------------------------------------------------------------------------

# NoExternalSynchronization: no-op for all parent/child combinations
forward_sync_external!(parent::DEIntegrator, child::DEIntegrator, ::NoExternalSynchronization) = nothing
backward_sync_external!(parent::DEIntegrator, child::DEIntegrator, ::NoExternalSynchronization) = nothing
forward_sync_external!(parent::OperatorSplittingIntegrator, child::DEIntegrator, ::NoExternalSynchronization) = nothing
backward_sync_external!(parent::OperatorSplittingIntegrator, child::DEIntegrator, ::NoExternalSynchronization) = nothing

# OperatorSplittingIntegrator parent with DEIntegrator child: parameter sync
function forward_sync_external!(
        parent::OperatorSplittingIntegrator,
        child::DEIntegrator,
        sync
    )
    return synchronize_solution_with_parameters!(parent, child.p, sync)
end
function backward_sync_external!(
        parent::OperatorSplittingIntegrator,
        child::DEIntegrator,
        sync
    )
    return synchronize_solution_with_parameters!(parent, child.p, sync)
end


function synchronize_solution_with_parameters!(
        parent::OperatorSplittingIntegrator, p, sync
    )
    @warn "Outer synchronizer not dispatched for parameter type $(typeof(p)) with synchronizer type $(typeof(sync))." maxlog = 1
    return nothing
end
function synchronize_solution_with_parameters!(
        parent::OperatorSplittingIntegrator, ::NullParameters, sync
    )
    return nothing
end

# Time stuff
function OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator::AnySplitIntegrator)
    if tdir(integrator) > 0
        integrator.dt = min(integrator.opts.dtmax, integrator.dt)
    else
        integrator.dt = max(integrator.opts.dtmax, integrator.dt)
    end
    dtmin = OrdinaryDiffEqCore.timedepentdtmin(integrator)
    if tdir(integrator) > 0
        integrator.dt = max(integrator.dt, dtmin)
    else
        integrator.dt = min(integrator.dt, dtmin)
    end
    return nothing
end

# Check time-step information consistency
validate_time_point(integrator::AnySplitIntegrator) = validate_time_point(integrator, integrator.child_subintegrators)
function validate_time_point(parent, child::SplitSubIntegrator)
    @assert parent.t == child.t "(parent.t=$(parent.t) != child.t=$(child.t))"
    return validate_time_point(child, child.child_subintegrators)
end

@unroll function validate_time_point(parent, children::Tuple)
    @unroll for child in children
        validate_time_point(parent, child)
    end
end

function validate_time_point(parent, child::DEIntegrator)
    return @assert child.t == parent.t "(parent.t=$(parent.t) != child.t=$(child.t))"
end

# ---------------------------------------------------------------------------
# _child_failed: check whether a child reported a failure
# ---------------------------------------------------------------------------
_child_failed(child::DEIntegrator) =
    child.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)

_child_failed(child::SplitSubIntegrator) =
    child.status.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
