# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? BinaryHeaps.FasterForward : BinaryHeaps.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = BinaryHeaps.BinaryHeap{FT, ordering}(tstops)

    # Keep `t0 < t <= tf` in tdir-space: `save_start` owns the initial point and
    # `save_end` the final one, so leaving either in the heap would duplicate it.
    tdir = tf > t0 ? one(FT) : -one(FT)
    saveat = if isnothing(saveat)
        FT[]
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        step = tdir * saveat
        collect((t0 + step):step:tf)
    else
        filter(t -> tdir * t0 < tdir * t <= tdir * tf, collect(FT, saveat))
    end
    saveat = BinaryHeaps.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
end

"""
    need_sync(a, b)

Return whether copying solution information from `b` into `a` is necessary.

# Arguments
- `a`: Destination vector or view.
- `b`: Source vector or view.

# Returns
`false` when both arguments share the same backing storage and copying would be
redundant; `true` otherwise. Extend this function for custom array wrappers whose
aliasing relationship cannot be determined by the built-in vector methods.
"""
need_sync

need_sync(a::AbstractVector, b::AbstractVector) = true
need_sync(a::SubArray, b::AbstractVector) = a.parent !== b
need_sync(a::AbstractVector, b::SubArray) = a !== b.parent
need_sync(a::SubArray, b::SubArray) = a.parent !== b.parent

"""
    sync_vectors!(a, b)

Copy solution information from `b` into `a` when [`need_sync`](@ref) determines that
their storage does not alias.

# Arguments
- `a`: Destination vector or view.
- `b`: Source vector or view.

# Returns
`nothing`. If no copy is required, `a` is left unchanged.
"""
function sync_vectors!(a, b)
    if need_sync(a, b) && a !== b
        a .= b
    end
    return nothing
end

"""
    forward_sync_subintegrator!(parent_integrator::AnySplitIntegrator, inner_integrator::DEIntegrator, solution_indices, sync)

Synchronize one child with its parent immediately before advancing that child.

# Arguments
- `parent_integrator`: Splitting node that owns the current full solution.
- `inner_integrator`: Direct leaf child to receive its local state.
- `solution_indices`: Indices selecting the child state in the parent's solution.
- `sync`: Synchronizer object for any parameter or cross-child synchronization.

# Interface requirements
The built-in method copies the parent's selected state into `inner_integrator` and
marks its derivative cache stale. To synchronize parameters or coupled child state,
implement `forward_sync_external!(parent_integrator, inner_integrator, sync)` for the
concrete `sync` type.

# Returns
`nothing`.
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
# FSAL information.
function mark_state_modified!(child::DEIntegrator)
    SciMLBase.derivative_discontinuity!(child, true)
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
    backward_sync_subintegrator!(parent_integrator::AnySplitIntegrator, inner_integrator::DEIntegrator, solution_indices, sync)

Synchronize one advanced child back into its parent.

# Arguments
- `parent_integrator`: Splitting node that owns the full solution.
- `inner_integrator`: Direct leaf child whose state is copied back.
- `solution_indices`: Indices selecting the child state in the parent's solution.
- `sync`: Synchronizer object for parameter or cross-child synchronization.

# Interface requirements
The built-in method copies the child state into the parent's selected state. To
synchronize parameters or coupled child state, implement
`backward_sync_external!(parent_integrator, inner_integrator, sync)` for the concrete
`sync` type.

# Returns
`nothing`.
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

"""
    forward_sync_external!(parent_integrator, inner_integrator, sync)

Synchronize external state into a child immediately before that child advances.

# Arguments
- `parent_integrator`: Splitting parent that owns the full state.
- `inner_integrator`: Direct child receiving the synchronized state.
- `sync`: Synchronizer object selected in the [`GenericSplitFunction`](@ref) tree.

# Interface requirements
Implement this method for a concrete `sync` type when parameters or coupled child
state must be refreshed before the child solves. The default
[`NoExternalSynchronization`](@ref) implementation is a no-op.

# Returns
`nothing`.

This is a developer extension API, not a supported end-user API.
"""
function forward_sync_external! end

"""
    backward_sync_external!(parent_integrator, inner_integrator, sync)

Synchronize external state after a child has advanced.

# Arguments
- `parent_integrator`: Splitting parent that owns the full state.
- `inner_integrator`: Direct child whose result may update coupled state.
- `sync`: Synchronizer object selected in the [`GenericSplitFunction`](@ref) tree.

# Interface requirements
Implement this method for a concrete `sync` type when parameters or coupled child
state must be refreshed after the child solves. The default
[`NoExternalSynchronization`](@ref) implementation is a no-op.

# Returns
`nothing`.

This is a developer extension API, not a supported end-user API.
"""
function backward_sync_external! end

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
function _fix_dt_at_bounds!(integrator::AnySplitIntegrator)
    # dtmin/dtmax are magnitudes; clamp |dt| and restore the direction. dtmin wins
    # over dtmax if the two conflict.
    dtmax = abs(integrator.opts.dtmax)
    dtmin = abs(DiffEqBase.timedepentdtmin(integrator))
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
# child_failed: check whether a child failed
#
# Leaves are checked through `SciMLBase.check_error`, not their stored retcode:
# fixed-step leaf integrators complete `step!` with NaN state without flagging it
# themselves, and the failure has to be caught *before* the surrounding step is
# accepted so that the escalation protocol can retry from a clean `uprev`.
# ---------------------------------------------------------------------------
child_failed(child::DEIntegrator) =
    SciMLBase.check_error(child) ∉ (ReturnCode.Default, ReturnCode.Success)

child_failed(child::SplitSubIntegrator) =
    child.status.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
