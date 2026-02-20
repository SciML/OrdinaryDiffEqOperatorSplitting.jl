mutable struct IntegratorStats
    naccept::Int64
    nreject::Int64
    # TODO inner solver stats
end

IntegratorStats() = IntegratorStats(0, 0)

Base.@kwdef mutable struct IntegratorOptions{tType, fType, F3}
    adaptive::Bool
    dtmin::tType = eps(Float64)
    dtmax::tType = Inf
    failfactor::fType = 4.0
    verbose::Bool = false
    isoutofdomain::F3 = DiffEqBase.ODE_DEFAULT_ISOUTOFDOMAIN
end

# ---------------------------------------------------------------------------
# SplitSubIntegratorStatus
# ---------------------------------------------------------------------------
"""
    SplitSubIntegratorStatus

Minimal error-communication object carried by a [`SplitSubIntegrator`](@ref).
Holds only `retcode` so that failure can be propagated up the operator-splitting
tree without carrying an actual solution vector.
"""
mutable struct SplitSubIntegratorStatus
    retcode::ReturnCode.T
end

SplitSubIntegratorStatus() = SplitSubIntegratorStatus(ReturnCode.Default)

# ---------------------------------------------------------------------------
# SplitSubIntegrator
# ---------------------------------------------------------------------------
"""
    SplitSubIntegrator

An intermediate node in the operator-splitting subintegrator tree.

Each `SplitSubIntegrator` is self-contained: it knows its own solution indices,
its children's synchronizers, solution indices, and sub-integrators.  It does
**not** carry an `f` field (operator information lives in the cache/algorithm).

## Fields
- `alg`                    — `AbstractOperatorSplittingAlgorithm` at this level
- `u`                      — local solution buffer for this sub-problem (may be a
                              view *or* an independent array, e.g. for GPU sub-problems)
- `uprev`                  — copy of `u` at the start of a step (for rollback)
- `u_master`               — reference to the full master solution vector of the
                              outermost `OperatorSplittingIntegrator` (needed for sync)
- `t`, `dt`, `dtcache`     — time tracking
- `iter`                   — step counter at this level
- `EEst`                   — error estimate (`NaN` for non-adaptive, `1.0` default
                              for adaptive)
- `controller`             — step-size controller (or `nothing` for non-adaptive)
- `force_stepfail`         — flag: current step must be retried
- `last_step_failed`       — flag: previous step failed (double-failure detection)
- `status`                 — [`SplitSubIntegratorStatus`](@ref) for retcode communication
- `cache`                  — `AbstractOperatorSplittingCache` for the algorithm at
                              this level
- `child_subintegrators`   — tuple of direct children (`SplitSubIntegrator` or
                              `DEIntegrator`)
- `solution_indices`       — global indices (into master `u`) **owned by this node**
- `child_solution_indices` — tuple of per-child global solution indices
- `child_synchronizers`    — tuple of per-child synchronizer objects
"""
mutable struct SplitSubIntegrator{
        algType,
        uType,
        tType,
        EEstType,
        controllerType,
        cacheType,
        childSubintType,
        solidxType,
        childSolidxType,
        childSyncType,
    }
    alg::algType
    u::uType                        # local solution buffer
    uprev::uType                    # local rollback buffer
    u_master::uType                 # reference to outermost master u
    t::tType
    dt::tType
    dtcache::tType
    iter::Int
    EEst::EEstType
    controller::controllerType
    force_stepfail::Bool
    last_step_failed::Bool
    status::SplitSubIntegratorStatus
    cache::cacheType
    child_subintegrators::childSubintType   # Tuple
    solution_indices::solidxType
    child_solution_indices::childSolidxType # Tuple
    child_synchronizers::childSyncType      # Tuple
end

# --- SplitSubIntegrator interface ---

@inline SciMLBase.isadaptive(sub::SplitSubIntegrator) = isadaptive(sub.alg)

# proposed-dt interface (mirrors ODEIntegrator)
function SciMLBase.set_proposed_dt!(sub::SplitSubIntegrator, dt)
    if sub.dtcache != dt  # only touch if actually changing
        sub.dtcache = dt
        if !isadaptive(sub)
            sub.dt = dt
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# OperatorSplittingIntegrator
# ---------------------------------------------------------------------------
"""
    OperatorSplittingIntegrator <: AbstractODEIntegrator

A variant of [`ODEIntegrator`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123)
to perform operator splitting.

Derived from https://github.com/CliMA/ClimaTimeSteppers.jl/blob/ef3023747606d2750e674d321413f80638136632/src/integrators.jl.

Note: `solution_index_tree` and `synchronizer_tree` have been removed; this
information now lives inside each [`SplitSubIntegrator`](@ref) child node.
"""
mutable struct OperatorSplittingIntegrator{
        fType,
        algType,
        uType,
        tType,
        pType,
        heapType,
        tstopsType,
        saveatType,
        callbackType,
        cacheType,
        solType,
        subintTreeType,
        childSolidxType,
        childSyncType,
        controllerType,
        optionsType,
    } <: SciMLBase.AbstractODEIntegrator{algType, true, uType, tType}
    const f::fType
    const alg::algType
    u::uType             # Master solution
    uprev::uType         # Master solution previous step
    tmp::uType           # Interpolation buffer
    p::pType
    t::tType             # Current time
    tprev::tType
    dt::tType            # Time step length used during time marching
    dtcache::tType       # Proposed time step length
    const dtchangeable::Bool
    tstops::heapType
    _tstops::tstopsType
    saveat::heapType
    _saveat::saveatType
    callback::callbackType
    advance_to_tstop::Bool
    last_step_failed::Bool
    force_stepfail::Bool
    isout::Bool
    u_modified::Bool
    cache::cacheType
    sol::solType
    # Tuple of SplitSubIntegrator nodes (one per top-level operator).
    child_subintegrators::subintTreeType
    child_solution_indices::childSolidxType # Tuple
    child_synchronizers::childSyncType      # Tuple
    iter::Int
    controller::controllerType
    opts::optionsType
    stats::IntegratorStats
    tdir::tType
end

# Convenience: the old field name `subintegrator_tree` was used in tests and
# docs; alias it so external code still compiles during the transition.
# (Remove in a future breaking release.)
@inline Base.getproperty(i::OperatorSplittingIntegrator, s::Symbol) =
    s === :subintegrator_tree ? getfield(i, :child_subintegrators) : getfield(i, s)

# ---------------------------------------------------------------------------
# __init
# ---------------------------------------------------------------------------
function SciMLBase.__init(
        prob::OperatorSplittingProblem,
        alg::AbstractOperatorSplittingAlgorithm,
        args...;
        dt,
        tstops = (),
        saveat = (),
        d_discontinuities = (),
        save_everystep = false,
        callback = nothing,
        advance_to_tstop = false,
        adaptive = isadaptive(alg),
        controller = nothing,
        alias_u0 = false,
        verbose = true,
        kwargs...
    )
    (; u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    dtcache = dt
    dt = tf > t0 ? dt : -dt
    tType = typeof(dt)

    (!isadaptive(alg) && adaptive && verbose) &&
        @warn("The algorithm $alg is not adaptive.")

    dtchangeable = true

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(
        tType, tstops, d_discontinuities, prob.tspan
    )
    saveat_internal  = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, prob.tspan)
    d_discontinuities_internal = OrdinaryDiffEqCore.initialize_d_discontinuities(
        tType, d_discontinuities, prob.tspan
    )

    u     = setup_u(prob, alg, alias_u0)
    uprev = setup_u(prob, alg, false)
    tmp   = setup_u(prob, alg, false)
    uType = typeof(u)

    sol      = SciMLBase.build_solution(prob, alg, tType[], uType[])
    callback = DiffEqBase.CallbackSet(callback)

    child_subintegrators, cache = build_subintegrators(
        prob, alg,
        uprev, u,
        u,            # u_master == u at the outermost level
        1:length(u),
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose
    )

    child_solution_indices  = ntuple(i -> prob.f.solution_indices[i],    length(prob.f.functions))
    child_synchronizers     = ntuple(i -> prob.f.synchronizers[i],       length(prob.f.functions))

    integrator = OperatorSplittingIntegrator(
        prob.f,
        alg,
        u, uprev, tmp,
        p,
        t0, copy(dt),
        dt, dtcache,
        dtchangeable,
        tstops_internal, tstops,
        saveat_internal,  saveat,
        callback,
        advance_to_tstop,
        false, false, false, false,
        cache, sol,
        child_subintegrators,
        child_solution_indices,
        child_synchronizers,
        0,
        controller,
        IntegratorOptions(; verbose, adaptive),
        IntegratorStats(),
        tType(tstops_internal.ordering isa DataStructures.FasterForward ? 1 : -1)
    )
    DiffEqBase.initialize!(callback, u0, t0, integrator)
    return integrator
end

# ---------------------------------------------------------------------------
# reinit!
# ---------------------------------------------------------------------------
SciMLBase.has_reinit(integrator::OperatorSplittingIntegrator) = true

function DiffEqBase.reinit!(
        integrator::OperatorSplittingIntegrator,
        u0 = integrator.sol.prob.u0;
        t0 = integrator.sol.prob.tspan[1],
        tf = integrator.sol.prob.tspan[2],
        dt = isadaptive(integrator) ? nothing : integrator.dtcache,
        erase_sol = false,
        tstops = integrator._tstops,
        saveat = integrator._saveat,
        reinit_callbacks = true,
        reinit_retcode = true
    )
    integrator.u    .= u0
    integrator.uprev .= u0
    integrator.t     = t0
    integrator.tprev = t0
    if dt !== nothing
        integrator.dt = dt
    end
    integrator.tstops, integrator.saveat =
        tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    integrator.iter = 0
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        DiffEqBase.initialize!(integrator.callback, u0, t0, integrator)
    else
        saving_callback = integrator.callback.discrete_callbacks[end]
        DiffEqBase.initialize!(saving_callback, u0, t0, integrator)
    end
    if reinit_retcode
        integrator.sol = SciMLBase.solution_new_retcode(
            integrator.sol, ReturnCode.Default
        )
    end

    _subreinit_tuple!(
        integrator.f,
        u0,
        integrator.child_subintegrators;
        t0, tf, dt,
        erase_sol, tstops, saveat,
        reinit_callbacks, reinit_retcode
    )
    return nothing
end

# --- subreinit! helpers ---

# Iterate over a tuple of children (outermost call from reinit!)
@unroll function _subreinit_tuple!(
        f,
        u0,
        children::Tuple;
        kwargs...
    )
    i = 1
    @unroll for child in children
        _subreinit_child!(get_operator(f, i), u0, child; kwargs...)
        i += 1
    end
end

# Reinitialise a leaf DEIntegrator child
function _subreinit_child!(
        f_child,
        u0,
        child::DEIntegrator;
        dt,
        kwargs...
    )
    if dt !== nothing && child.dtchangeable
        SciMLBase.set_proposed_dt!(child, dt)
    end
    # solution_indices live on the parent SplitSubIntegrator (or on the outer
    # integrator for top-level children) — they were baked into child at init.
    # reinit! on an ODEIntegrator resets u from its prob.u0; we need to pass
    # the correct slice here.  The parent calls us with the correct f_child
    # but not the indices — those are embedded in child.sol.prob.u0 already
    # because we constructed child with a view/copy of the right slice.
    return DiffEqBase.reinit!(child; kwargs...)
end

# Reinitialise an intermediate SplitSubIntegrator child
function _subreinit_child!(
        f_child,
        u0,
        sub::SplitSubIntegrator;
        t0,
        tf,
        dt,
        kwargs...
    )
    idxs = sub.solution_indices
    sub.u    .= @view u0[idxs]
    sub.uprev .= @view u0[idxs]
    sub.t     = t0
    if dt !== nothing
        SciMLBase.set_proposed_dt!(sub, dt)
    end
    sub.iter             = 0
    sub.force_stepfail   = false
    sub.last_step_failed = false
    sub.status           = SplitSubIntegratorStatus(ReturnCode.Default)
    # Reset EEst to its appropriate default
    if isadaptive(sub)
        sub.EEst = one(sub.EEst)
    end
    # Recurse into this node's children
    _subreinit_tuple!(
        f_child,
        u0,
        sub.child_subintegrators;
        t0, tf, dt, kwargs...
    )
    return nothing
end

# ---------------------------------------------------------------------------
# handle_tstop!
# ---------------------------------------------------------------------------
function OrdinaryDiffEqCore.handle_tstop!(integrator::OperatorSplittingIntegrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t    = tdir(integrator) * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        if tdir_t == tdir_tstop
            while tdir_t == tdir_tstop
                SciMLBase.pop_tstop!(integrator)
                SciMLBase.has_tstop(integrator) ?
                    (tdir_tstop = SciMLBase.first_tstop(integrator)) : break
            end
            notify_integrator_hit_tstop!(integrator)
        elseif tdir_t > tdir_tstop
            if !integrator.dtchangeable
                SciMLBase.change_t_via_interpolation!(
                    integrator,
                    tdir(integrator) * SciMLBase.pop_tstop!(integrator),
                    Val{true}
                )
                notify_integrator_hit_tstop!(integrator)
            else
                error("Something went wrong. Integrator stepped past tstops but the algorithm was dtchangeable. Please report this error.")
            end
        end
    end
    return nothing
end

notify_integrator_hit_tstop!(integrator::OperatorSplittingIntegrator) = nothing

is_first_iteration(integrator::OperatorSplittingIntegrator)  = integrator.iter == 0
increment_iteration(integrator::OperatorSplittingIntegrator) = integrator.iter += 1

# ---------------------------------------------------------------------------
# Step accept/reject — outermost integrator
# ---------------------------------------------------------------------------
function reject_step!(integrator::OperatorSplittingIntegrator)
    OrdinaryDiffEqCore.increment_reject!(integrator.stats)
    return reject_step!(integrator, integrator.cache, integrator.controller)
end
function reject_step!(integrator::OperatorSplittingIntegrator, cache, controller)
    integrator.u .= integrator.uprev
    # TODO: roll back sub-integrators
    return nothing
end
function reject_step!(integrator::OperatorSplittingIntegrator, cache, ::Nothing)
    if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
    return nothing
end

function should_accept_step(integrator::OperatorSplittingIntegrator)
    integrator.force_stepfail || integrator.isout && return false
    return should_accept_step(integrator, integrator.cache, integrator.controller)
end
function should_accept_step(integrator::OperatorSplittingIntegrator, cache, ::Nothing)
    return !(integrator.force_stepfail)
end

function accept_step!(integrator::OperatorSplittingIntegrator)
    OrdinaryDiffEqCore.increment_accept!(integrator.stats)
    return accept_step!(integrator, integrator.cache, integrator.controller)
end
function accept_step!(integrator::OperatorSplittingIntegrator, cache, controller)
    return store_previous_info!(integrator)
end
function store_previous_info!(integrator::OperatorSplittingIntegrator)
    if length(integrator.uprev) > 0
        update_uprev!(integrator)
    end
    return nothing
end
function update_uprev!(integrator::OperatorSplittingIntegrator)
    RecursiveArrayTools.recursivecopy!(integrator.uprev, integrator.u)
    return nothing
end

# Step accept/reject — SplitSubIntegrator
function accept_step!(sub::SplitSubIntegrator)
    RecursiveArrayTools.recursivecopy!(sub.uprev, sub.u)
    return nothing
end
function reject_step!(sub::SplitSubIntegrator)
    sub.u .= sub.uprev
    _rollback_children!(sub.child_subintegrators, sub.u_master)
    return nothing
end

# Roll back each child's local buffer to match master u.
# For DEIntegrators the leaf will be re-synced via forward_sync before the
# next attempt, so there is nothing to do here.
@unroll function _rollback_children!(children::Tuple, u_master)
    @unroll for child in children
        _rollback_child!(child, u_master)
    end
end
function _rollback_child!(child::SplitSubIntegrator, u_master)
    child.u .= @view u_master[child.solution_indices]
    RecursiveArrayTools.recursivecopy!(child.uprev, child.u)
    _rollback_children!(child.child_subintegrators, u_master)
    return nothing
end
function _rollback_child!(child::DEIntegrator, u_master)
    # forward_sync before the next sub-step will restore this correctly.
    return nothing
end

# ---------------------------------------------------------------------------
# step_header! / step_footer! — outermost integrator
# ---------------------------------------------------------------------------
function step_header!(integrator::OperatorSplittingIntegrator)
    if !is_first_iteration(integrator)
        if should_accept_step(integrator)
            accept_step!(integrator)
        else
            reject_step!(integrator)
        end
    elseif integrator.u_modified
        update_uprev!(integrator)
    end
    increment_iteration(integrator)
    OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator)
    OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator)
    return integrator.force_stepfail = false
end

function footer_reset_flags!(integrator)
    return integrator.u_modified = false
end
function setup_validity_flags!(integrator, t_next)
    return integrator.isout = false
end
function fix_solution_buffer_sizes!(integrator, sol)
    resize!(integrator.sol.t, integrator.saveiter)
    resize!(integrator.sol.u, integrator.saveiter)
    if !(integrator.sol isa SciMLBase.DAESolution)
        resize!(integrator.sol.k, integrator.saveiter_dense)
    end
    return nothing
end

function step_footer!(integrator::OperatorSplittingIntegrator)
    ttmp = integrator.t + tdir(integrator) * integrator.dt
    footer_reset_flags!(integrator)
    setup_validity_flags!(integrator, ttmp)
    if should_accept_step(integrator)
        integrator.last_step_failed = false
        integrator.tprev = integrator.t
        integrator.t     = ttmp
        step_accept_controller!(integrator)
    elseif integrator.force_stepfail
        if isadaptive(integrator)
            step_reject_controller!(integrator)
            OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
        elseif integrator.dtchangeable
            integrator.dt /= integrator.opts.failfactor
        elseif integrator.last_step_failed
            return
        end
        integrator.last_step_failed = true
    end
    return nothing
end

# ---------------------------------------------------------------------------
# __solve / solve! / step!
# ---------------------------------------------------------------------------
function SciMLBase.__solve(
        prob::OperatorSplittingProblem,
        alg::AbstractOperatorSplittingAlgorithm, args...; kwargs...
    )
    integrator = SciMLBase.__init(prob, alg, args...; kwargs...)
    return DiffEqBase.solve!(integrator)
end

function DiffEqBase.solve!(integrator::OperatorSplittingIntegrator)
    while !isempty(integrator.tstops)
        while tdir(integrator) * integrator.t < SciMLBase.first_tstop(integrator)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return integrator.sol
            __step!(integrator)
            step_footer!(integrator)
            SciMLBase.has_tstop(integrator) || break
        end
        OrdinaryDiffEqCore.handle_tstop!(integrator)
    end
    SciMLBase.postamble!(integrator)
    integrator.sol.retcode != ReturnCode.Default && return integrator.sol
    return integrator.sol = SciMLBase.solution_new_retcode(
        integrator.sol, ReturnCode.Success
    )
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator)
    @timeit_debug "step!" if integrator.advance_to_tstop
        tstop = SciMLBase.first_tstop(integrator)
        while !reached_tstop(integrator, tstop)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return
            __step!(integrator)
            step_footer!(integrator)
            SciMLBase.has_tstop(integrator) || break
        end
    else
        step_header!(integrator)
        @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
            ReturnCode.Success, ReturnCode.Default,
        ) && return
        __step!(integrator)
        step_footer!(integrator)
        while !should_accept_step(integrator)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return
            __step!(integrator)
            step_footer!(integrator)
        end
    end
    OrdinaryDiffEqCore.handle_tstop!(integrator)
    return
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator, dt, stop_at_tdt = false)
    @timeit_debug "step!" begin
        dt <= zero(dt) && error("dt must be positive")
        stop_at_tdt && !integrator.dtchangeable &&
            error("Cannot stop at t + dt if dtchangeable is false")
        tnext = integrator.t + tdir(integrator) * dt
        stop_at_tdt && DiffEqBase.add_tstop!(integrator, tnext)
        while !reached_tstop(integrator, tnext, stop_at_tdt)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return
            __step!(integrator)
            step_footer!(integrator)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# check_error
# ---------------------------------------------------------------------------
function SciMLBase.check_error(integrator::OperatorSplittingIntegrator)
    if !SciMLBase.successful_retcode(integrator.sol) &&
            integrator.sol.retcode != ReturnCode.Default
        return integrator.sol.retcode
    end
    if DiffEqBase.NAN_CHECK(integrator.dtcache) || DiffEqBase.NAN_CHECK(integrator.dt)
        integrator.opts.verbose &&
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        return ReturnCode.DtNaN
    end
    return _check_error_children(integrator.sol.retcode, integrator.child_subintegrators)
end

@unroll function _check_error_children(current_retcode, children::Tuple)
    @unroll for child in children
        rc = _child_retcode(child)
        if !SciMLBase.successful_retcode(rc) && rc != ReturnCode.Default
            return rc
        end
    end
    return current_retcode
end

_child_retcode(child::DEIntegrator)       = SciMLBase.check_error(child)
_child_retcode(child::SplitSubIntegrator) = child.status.retcode

# ---------------------------------------------------------------------------
# Internal step
# ---------------------------------------------------------------------------
function setup_u(prob::OperatorSplittingProblem, solver, alias_u0)
    alias_u0 ? prob.u0 : RecursiveArrayTools.recursivecopy(prob.u0)
end

@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator)
    return (integrator.tmp,)
end

function linear_interpolation!(y, t, y1, y2, t1, t2)
    return y .= y1 + (t - t1) * (y2 - y1) / (t2 - t1)
end
function (integrator::OperatorSplittingIntegrator)(tmp, t)
    return linear_interpolation!(
        tmp, t, integrator.uprev, integrator.u, integrator.tprev, integrator.t
    )
end

# Stepsize controller hooks — outermost integrator
@inline function stepsize_controller!(integrator::OperatorSplittingIntegrator)
    isadaptive(integrator.alg) || return nothing
    return stepsize_controller!(integrator, integrator.alg)
end
@inline function step_accept_controller!(integrator::OperatorSplittingIntegrator)
    isadaptive(integrator.alg) || return nothing
    return step_accept_controller!(integrator, integrator.alg, nothing)
end
@inline function step_reject_controller!(integrator::OperatorSplittingIntegrator)
    isadaptive(integrator.alg) || return nothing
    return step_reject_controller!(integrator, integrator.alg, nothing)
end

# Time helpers
tdir(integrator) =
    integrator.tstops.ordering isa DataStructures.FasterForward ? 1 : -1
is_past_t(integrator, t) =
    tdir(integrator) * (t - integrator.t) ≤ zero(integrator.t)
function reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable)
    if stop_at_tstop
        integrator.t > tstop &&
            error("Integrator missed stop at $tstop (current time=$(integrator.t)). Aborting.")
        return integrator.t == tstop
    else
        return is_past_t(integrator, tstop)
    end
end

# SciMLBase integrator interface
function SciMLBase.done(integrator::OperatorSplittingIntegrator)
    integrator.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success) && return true
    if isempty(integrator.tstops)
        SciMLBase.postamble!(integrator)
        return true
    end
    return false
end

function SciMLBase.postamble!(integrator::OperatorSplittingIntegrator)
    return DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
end

function __step!(integrator::OperatorSplittingIntegrator)
    tnext = integrator.t + integrator.dt
    _sync_children!(integrator)
    advance_solution_to!(integrator, tnext)
    stepsize_controller!(integrator)
    return nothing
end

# Sync all direct children of the outermost integrator
function _sync_children!(integrator::OperatorSplittingIntegrator)
    _sync_children_tuple!(integrator.child_subintegrators, integrator)
end

@unroll function _sync_children_tuple!(
        children::Tuple,
        parent::OperatorSplittingIntegrator
    )
    @unroll for child in children
        _sync_child_to_parent!(child, parent)
    end
end

function _sync_child_to_parent!(child::DEIntegrator, parent::OperatorSplittingIntegrator)
    @assert child.t == parent.t "($(child.t) != $(parent.t))"
    if !isadaptive(child) && child.dtchangeable
        SciMLBase.set_proposed_dt!(child, parent.dt)
    end
end

function _sync_child_to_parent!(
        child::SplitSubIntegrator, parent::OperatorSplittingIntegrator
    )
    @assert child.t == parent.t "($(child.t) != $(parent.t))"
    if !isadaptive(child)
        SciMLBase.set_proposed_dt!(child, parent.dt)
    end
end

# Entry point: dispatch to the algorithm's advance_solution_to!
function advance_solution_to!(integrator::OperatorSplittingIntegrator, tnext)
    return advance_solution_to!(integrator, integrator.cache, tnext)
end

# Algorithm-level dispatch (implemented in solver.jl per algorithm)
function advance_solution_to!(
        integrator::OperatorSplittingIntegrator,
        cache::AbstractOperatorSplittingCache, tnext::Number
    )
    return advance_solution_to!(
        integrator, integrator.child_subintegrators, cache, tnext
    )
end

# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

# Top-level builder: called from __init with the full problem.
# Returns (child_subintegrators::Tuple, cache::AbstractOperatorSplittingCache)
function build_subintegrators(
        prob::OperatorSplittingProblem,
        alg::AbstractOperatorSplittingAlgorithm,
        uprevouter::AbstractVector,
        uouter::AbstractVector,
        u_master::AbstractVector,
        solution_indices,
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose
    )
    (; f, p) = prob

    results = ntuple(
        i -> _build_child(
            prob,
            alg.inner_algs[i],
            get_operator(f, i),
            p[i],
            uprevouter, uouter, u_master,
            f.solution_indices[i],
            t0, dt, tf,
            tstops, saveat, d_discontinuities, callback,
            adaptive, verbose
        ),
        length(f.functions)
    )

    child_subintegrators = ntuple(i -> results[i][1], length(f.functions))
    child_caches         = ntuple(i -> results[i][2], length(f.functions))

    cache = init_cache(
        f, alg;
        uprev = uprevouter, u = uouter, alias_u = true,
        inner_caches = child_caches
    )

    return child_subintegrators, cache
end

# Intermediate node: inner alg is an AbstractOperatorSplittingAlgorithm and
# f is a GenericSplitFunction  →  produce a SplitSubIntegrator
function _build_child(
        prob::OperatorSplittingProblem,
        alg::AbstractOperatorSplittingAlgorithm,
        f::GenericSplitFunction,
        p::Tuple,
        uprevouter::AbstractVector,
        uouter::AbstractVector,
        u_master::AbstractVector,
        solution_indices,
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose,
        save_end = false,
        controller = nothing
    )
    tType = typeof(dt)

    # Recurse: build each grandchild
    grandchild_results = ntuple(
        i -> _build_child(
            prob,
            alg.inner_algs[i],
            get_operator(f, i),
            p[i],
            uprevouter, uouter, u_master,
            f.solution_indices[i],
            t0, dt, tf,
            tstops, saveat, d_discontinuities, callback,
            adaptive, verbose
        ),
        length(f.functions)
    )

    child_subintegrators    = ntuple(i -> grandchild_results[i][1], length(f.functions))
    child_caches            = ntuple(i -> grandchild_results[i][2], length(f.functions))
    child_solution_indices  = ntuple(i -> f.solution_indices[i],    length(f.functions))
    child_synchronizers     = ntuple(i -> f.synchronizers[i],       length(f.functions))

    u_sub    = @view uouter[solution_indices]
    uprev_sub = @view uprevouter[solution_indices]

    level_cache = init_cache(
        f, alg;
        uprev = uprev_sub, u = u_sub,
        inner_caches = child_caches
    )

    EEst_val = isadaptive(alg) ? one(tType) : tType(NaN)

    sub = SplitSubIntegrator(
        alg,
        # u and uprev: independent copies so that rollback works even when
        # u_sub is a view into a device-local buffer.
        RecursiveArrayTools.recursivecopy(Array(u_sub)),
        RecursiveArrayTools.recursivecopy(Array(u_sub)),
        u_master,
        t0, dt, dt,     # t, dt, dtcache
        0,              # iter
        EEst_val,
        controller,
        false, false,   # force_stepfail, last_step_failed
        SplitSubIntegratorStatus(),
        level_cache,
        child_subintegrators,
        solution_indices,
        child_solution_indices,
        child_synchronizers
    )

    return sub, level_cache
end

# Leaf node: inner alg is a plain SciMLBase.AbstractODEAlgorithm
# → produce an ODEIntegrator (existing behaviour)
function _build_child(
        prob::OperatorSplittingProblem,
        alg::SciMLBase.AbstractODEAlgorithm,
        f::F, p::P,
        uprevouter::S, uouter::S,
        u_master::S,
        solution_indices,
        t0::T, dt::T, tf::T,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose,
        save_end = false,
        controller = nothing
    ) where {S, T, P, F}
    u    = @view uouter[solution_indices]
    prob2 = if p isa NullParameters
        SciMLBase.ODEProblem(f, u, (t0, min(t0 + dt, tf)))
    else
        SciMLBase.ODEProblem(f, u, (t0, min(t0 + dt, tf)), p)
    end

    integrator = SciMLBase.__init(
        prob2, alg;
        dt,
        saveat = (),
        d_discontinuities,
        save_everystep = false,
        advance_to_tstop = false,
        adaptive, controller, verbose
    )
    return integrator, integrator.cache
end

# ---------------------------------------------------------------------------
# SciMLBase API
# ---------------------------------------------------------------------------
SciMLBase.has_stats(::OperatorSplittingIntegrator) = true

SciMLBase.has_tstop(i::OperatorSplittingIntegrator)    = !isempty(i.tstops)
SciMLBase.first_tstop(i::OperatorSplittingIntegrator)  = first(i.tstops)
SciMLBase.pop_tstop!(i::OperatorSplittingIntegrator)   = pop!(i.tstops)

DiffEqBase.get_dt(i::OperatorSplittingIntegrator) = i.dt
function set_dt!(i::OperatorSplittingIntegrator, dt)
    dt <= zero(dt) && error("dt must be positive")
    return i.dt = dt
end

function DiffEqBase.add_tstop!(i::OperatorSplittingIntegrator, t)
    is_past_t(i, t) &&
        error("Cannot add a tstop at $t because that is behind the current \
               integrator time $(i.t)")
    return push!(i.tstops, t)
end

function DiffEqBase.add_saveat!(i::OperatorSplittingIntegrator, t)
    is_past_t(i, t) &&
        error("Cannot add a saveat point at $t because that is behind the \
               current integrator time $(i.t)")
    return push!(i.saveat, t)
end

DiffEqBase.u_modified!(i::OperatorSplittingIntegrator, bool) = nothing
