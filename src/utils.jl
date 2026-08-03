# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? BinaryHeaps.FasterForward : BinaryHeaps.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = BinaryHeaps.BinaryHeap{FT, ordering}(tstops)

    # `t0` is excluded and `tf` included, matching OrdinaryDiffEqCore's
    # `initialize_saveat`: the initial point is owned by `save_start` and the final
    # one by `save_end`, so leaving them in the heap would duplicate them.
    tdir = tf > t0 ? one(FT) : -one(FT)
    if isnothing(saveat)
        saveat = FT[]
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        step = tdir * saveat
        # `tf` is not appended: the range hits it when it divides the span evenly,
        # and `save_end` owns the final point otherwise.
        saveat = collect((t0 + step):step:tf)
    else
        saveat = collect(FT, saveat)
    end
    saveat = filter(t -> tdir * t0 < tdir * t <= tdir * tf, saveat)
    saveat = BinaryHeaps.BinaryHeap{FT, ordering}(saveat)

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
    # Skip if we are calling the same subintegrator twice in a row (as in e.g. palindromic methods)
    if is_next_sync_continuous(parent)
        reset_next_sync_continuous(parent)
        return nothing
    end
    forward_sync_internal!(parent.u, parent.uprev, child, solution_indices)
    @timeit_debug "external sync" forward_sync_external!(parent, child, sync)
    return nothing
end

# Tell a leaf integrator that its state was changed from the outside so it discards
# FSAL information. SciMLBase v3 renamed `u_modified!` → `derivative_discontinuity!`;
# call the appropriate name based on which SciMLBase is loaded.
function mark_state_modified!(child::DEIntegrator)
    @static if isdefined(SciMLBase, :derivative_discontinuity!)
        SciMLBase.derivative_discontinuity!(child, true)
    else
        SciMLBase.u_modified!(child, true)
    end
    return nothing
end

# Shared internal helper: copy the parent u slice → child DEIntegrator u/uprev.
# `uprev_parent` is the parent's rollback buffer: the refresh of the child's own
# rollback anchor must never write through a child `uprev` that aliases it, because
# that buffer has to survive the whole step untouched (rejection and the palindromic
# mid-step rewind restore from it). When they alias, the slice already holds the
# interval start state and no copy is needed.
function forward_sync_internal!(u_source, uprev_parent, child::DEIntegrator, solution_indices)
    @views usrc = u_source[solution_indices]
    @timeit_debug "sync vectors" begin
        sync_vectors!(child.u, usrc)
        if need_sync(child.uprev, uprev_parent)
            sync_vectors!(child.uprev, child.u)
        end
    end
    mark_state_modified!(child)
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
    backward_sync_internal!(parent.u, child, solution_indices)
    @timeit_debug "external sync" backward_sync_external!(parent, child, sync)
    return nothing
end

function backward_sync_internal!(u_dest, child::DEIntegrator, solution_indices)
    @views udst = u_dest[solution_indices]
    @timeit_debug "sync vectors" begin
        sync_vectors!(udst, child.u)
    end
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
    # dtmin/dtmax are magnitudes; clamp |dt| and restore the direction. dtmin wins
    # over dtmax if the two conflict.
    dtmax = abs(integrator.opts.dtmax)
    dtmin = abs(OrdinaryDiffEqCore.timedepentdtmin(integrator))
    integrator.dt = tdir(integrator) * max(min(abs(integrator.dt), dtmax), dtmin)
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
# _child_failed: check whether a child failed
#
# Leaves are checked through `SciMLBase.check_error`, not their stored retcode:
# fixed-step leaf integrators complete `step!` with NaN state without flagging it
# themselves, and the failure has to be caught *before* the surrounding step is
# accepted so that the escalation protocol can retry from a clean `uprev`.
# ---------------------------------------------------------------------------
_child_failed(child::DEIntegrator) =
    SciMLBase.check_error(child) ∉ (ReturnCode.Default, ReturnCode.Success)

_child_failed(child::SplitSubIntegrator) =
    child.status.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
