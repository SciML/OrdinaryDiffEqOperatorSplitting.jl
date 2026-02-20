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
It contains only the `retcode` so that failure can be propagated up the
operator-splitting tree without carrying an actual solution vector.
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

An intermediate node in the operator-splitting subintegrator tree.  It is
self-contained: it knows its own solution indices, its child synchronizers,
and the child solution-index tree.  It does **not** carry an `f` field
(operator information lives in the cache / algorithm).

Fields
------
- `alg`                — the `AbstractOperatorSplittingAlgorithm` at this level
- `u`                  — view into the *master* solution vector for this sub-problem
- `uprev`              — copy of `u` at the start of a step (for rollback)
- `u_master`           — reference to the full master solution vector of the
                          outermost `OperatorSplittingIntegrator` (needed during sync)
- `t`, `dt`, `dtcache` — time tracking
- `iter`               — step counter
- `EEst`               — error estimate (`NaN` for non-adaptive, `1.0` default for adaptive)
- `controller`         — step-size controller (or `nothing` for non-adaptive)
- `force_stepfail`     — flag set when a step must be re-tried
- `last_step_failed`   — flag set after a failed step to detect double-failure
- `status`             — [`SplitSubIntegratorStatus`](@ref) for retcode communication
- `cache`              — `AbstractOperatorSplittingCache` for the algorithm at this level
- `subintegrator_tree` — tuple of child integrators (`SplitSubIntegrator` or `DEIntegrator`)
- `solution_indices`   — global indices (into master `u`) owned by this sub-integrator
- `solution_index_tree`— per-child global solution indices
- `synchronizer_tree`  — per-child synchronizer objects
"""
mutable struct SplitSubIntegrator{
        algType,
        uType,
        tType,
        EEstType,
        controllerType,
        cacheType,
        subintTreeType,
        solidxType,
        solidxTreeType,
        syncTreeType,
    }
    alg::algType
    u::uType                  # view into master u for this sub-problem
    uprev::uType              # local copy for rollback  (same element type, plain Array)
    u_master::uType           # reference to the outermost master u
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
    subintegrator_tree::subintTreeType   # Tuple
    solution_indices::solidxType
    solution_index_tree::solidxTreeType  # Tuple
    synchronizer_tree::syncTreeType      # Tuple
end

# Convenience predicate
@inline SciMLBase.isadaptive(sub::SplitSubIntegrator) = isadaptive(sub.alg)

# ---------------------------------------------------------------------------
# OperatorSplittingIntegrator
# ---------------------------------------------------------------------------
"""
    OperatorSplittingIntegrator <: AbstractODEIntegrator

A variant of [`ODEIntegrator`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123) to perform operator splitting.

Derived from https://github.com/CliMA/ClimaTimeSteppers.jl/blob/ef3023747606d2750e674d321413f80638136632/src/integrators.jl.
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
        controllerType,
        optionsType,
    } <: SciMLBase.AbstractODEIntegrator{algType, true, uType, tType}
    const f::fType
    const alg::algType
    u::uType                 # Master Solution
    uprev::uType             # Master Solution
    tmp::uType               # Interpolation buffer
    p::pType
    t::tType                 # Current time
    tprev::tType
    dt::tType                # Time step length used during time marching
    dtcache::tType           # Proposed time step length
    const dtchangeable::Bool # Indicator whether dtcache can be changed
    tstops::heapType
    _tstops::tstopsType      # argument to __init used as default argument to reinit!
    saveat::heapType
    _saveat::saveatType      # argument to __init used as default argument to reinit!
    callback::callbackType
    advance_to_tstop::Bool
    # TODO group these into some internal flag struct
    last_step_failed::Bool
    force_stepfail::Bool
    isout::Bool
    u_modified::Bool
    # DiffEqBase.initialize! and DiffEqBase.finalize!
    cache::cacheType
    sol::solType
    # NOTE: solution_index_tree and synchronizer_tree have been moved into
    # the SplitSubIntegrator nodes.  The flat subintegrator_tree here is a
    # Tuple of SplitSubIntegrator (or DEIntegrator for the degenerate
    # single-level case).
    subintegrator_tree::subintTreeType
    iter::Int
    controller::controllerType
    opts::optionsType
    stats::IntegratorStats
    tdir::tType
end

# ---------------------------------------------------------------------------
# __init
# ---------------------------------------------------------------------------
# called by DiffEqBase.init and DiffEqBase.solve
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

    # Warn if the algorithm is non-adaptive but the user tries to make it adaptive.
    (!isadaptive(alg) && adaptive && verbose) && warn("The algorithm $alg is not adaptive.")

    dtchangeable = true # isadaptive(alg)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(
        tType, tstops, d_discontinuities, prob.tspan
    )
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, prob.tspan)
    d_discontinuities_internal = OrdinaryDiffEqCore.initialize_d_discontinuities(
        tType, d_discontinuities, prob.tspan
    )

    u = setup_u(prob, alg, alias_u0)
    uprev = setup_u(prob, alg, false)
    tmp = setup_u(prob, alg, false)
    uType = typeof(u)

    sol = SciMLBase.build_solution(prob, alg, tType[], uType[])

    callback = DiffEqBase.CallbackSet(callback)

    # Build the subintegrator tree.  Each SplitSubIntegrator is now
    # self-contained: it holds its own solution_indices, solution_index_tree,
    # and synchronizer_tree.
    subintegrator_tree, cache = build_subintegrator_tree_with_cache(
        prob, alg,
        uprev, u,
        u,          # u_master  == u at the outermost level
        1:length(u),
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose
    )

    integrator = OperatorSplittingIntegrator(
        prob.f,
        alg,
        u,
        uprev,
        tmp,
        p,
        t0,
        copy(t0),
        dt,
        dtcache,
        dtchangeable,
        tstops_internal,
        tstops,
        saveat_internal,
        saveat,
        callback,
        advance_to_tstop,
        false,
        false,
        false,
        false,
        cache,
        sol,
        subintegrator_tree,
        0,
        controller,
        IntegratorOptions(; verbose, adaptive),
        IntegratorStats(),
        tType(tstops_internal.ordering isa DataStructures.FasterForward ? 1 : -1)
    )
    DiffEqBase.initialize!(callback, u0, t0, integrator)
    return integrator
end

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
    integrator.u .= u0
    integrator.uprev .= u0
    integrator.t = t0
    integrator.tprev = t0
    if dt !== nothing
        integrator.dt = dt
    end
    integrator.tstops, integrator.saveat = tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    integrator.iter = 0
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        DiffEqBase.initialize!(integrator.callback, u0, t0, integrator)
    else # always reinit the saving callback so that t0 can be saved if needed
        saving_callback = integrator.callback.discrete_callbacks[end]
        DiffEqBase.initialize!(saving_callback, u0, t0, integrator)
    end
    if reinit_retcode
        integrator.sol = SciMLBase.solution_new_retcode(
            integrator.sol, ReturnCode.Default
        )
    end

    return subreinit!(
        integrator.f,
        u0,
        integrator.subintegrator_tree;
        t0, tf, dt,
        erase_sol,
        tstops,
        saveat,
        reinit_callbacks,
        reinit_retcode
    )
end

# subreinit! for a leaf DEIntegrator
function subreinit!(
        f,
        u0,
        subintegrator::DEIntegrator;
        dt,
        kwargs...
    )
    # dt is not reset as expected in reinit!
    if dt !== nothing
        subintegrator.dt = dt
    end
    # solution_indices are carried by the parent SplitSubIntegrator
    error("subreinit! called directly on a DEIntegrator — should be reached only via SplitSubIntegrator")
end

# subreinit! for an intermediate SplitSubIntegrator
function subreinit!(
        f,
        u0,
        sub::SplitSubIntegrator;
        t0,
        tf,
        dt,
        kwargs...
    )
    idxs = sub.solution_indices
    sub.u .= @view u0[idxs]
    sub.uprev .= @view u0[idxs]
    sub.t = t0
    if dt !== nothing
        sub.dt = dt
        sub.dtcache = dt
    end
    sub.iter = 0
    sub.force_stepfail = false
    sub.last_step_failed = false
    sub.status = SplitSubIntegratorStatus(ReturnCode.Default)
    # Reset EEst to the appropriate default
    if isadaptive(sub)
        sub.EEst = one(sub.EEst)
    else
        sub.EEst = sub.EEst  # keep NaN sentinel
    end
    return subreinit_children!(f, u0, sub; t0, tf, dt, kwargs...)
end

@unroll function subreinit_children!(
        f,
        u0,
        sub::SplitSubIntegrator;
        kwargs...
    )
    i = 1
    @unroll for child in sub.subintegrator_tree
        _subreinit_child!(get_operator(f, i), u0, child, sub.solution_index_tree[i]; kwargs...)
        i += 1
    end
end

# Dispatch for leaf DEIntegrator children
function _subreinit_child!(
        f_child,
        u0,
        child::DEIntegrator,
        child_solution_indices;
        dt,
        kwargs...
    )
    if dt !== nothing
        child.dt = dt
    end
    return DiffEqBase.reinit!(child, @view(u0[child_solution_indices]); kwargs...)
end

# Dispatch for nested SplitSubIntegrator children
function _subreinit_child!(
        f_child,
        u0,
        child::SplitSubIntegrator,
        _child_solution_indices;  # ignored — child carries its own
        kwargs...
    )
    return subreinit!(f_child, u0, child; kwargs...)
end

# Top-level subreinit! over a tuple of subintegrators (called from reinit!)
@unroll function subreinit!(
        f,
        u0,
        subintegrators::Tuple;
        kwargs...
    )
    i = 1
    @unroll for sub in subintegrators
        subreinit!(get_operator(f, i), u0, sub; kwargs...)
        i += 1
    end
end

# ---------------------------------------------------------------------------
# handle_tstop!
# ---------------------------------------------------------------------------
function OrdinaryDiffEqCore.handle_tstop!(integrator::OperatorSplittingIntegrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t = tdir(integrator) * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        if tdir_t == tdir_tstop
            while tdir_t == tdir_tstop #remove all redundant copies
                res = SciMLBase.pop_tstop!(integrator)
                SciMLBase.has_tstop(integrator) ?
                    (tdir_tstop = SciMLBase.first_tstop(integrator)) : break
            end
            notify_integrator_hit_tstop!(integrator)
        elseif tdir_t > tdir_tstop
            if !integrator.dtchangeable
                SciMLBase.change_t_via_interpolation!(
                    integrator,
                    tdir(integrator) *
                        SciMLBase.pop_tstop!(integrator), Val{true}
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

is_first_iteration(integrator::OperatorSplittingIntegrator) = integrator.iter == 0
increment_iteration(integrator::OperatorSplittingIntegrator) = integrator.iter += 1

# ---------------------------------------------------------------------------
# Controller interface — outermost integrator
# ---------------------------------------------------------------------------
function reject_step!(integrator::OperatorSplittingIntegrator)
    OrdinaryDiffEqCore.increment_reject!(integrator.stats)
    return reject_step!(integrator, integrator.cache, integrator.controller)
end
function reject_step!(integrator::OperatorSplittingIntegrator, cache, controller)
    return integrator.u .= integrator.uprev
    # TODO what do we need to do with the subintegrators?
end
function reject_step!(integrator::OperatorSplittingIntegrator, cache, ::Nothing)
    return if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
end

# Solution looping interface
function should_accept_step(integrator::OperatorSplittingIntegrator)
    if integrator.force_stepfail || integrator.isout
        return false
    end
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
    return if length(integrator.uprev) > 0 # Integrator can rollback
        update_uprev!(integrator)
    end
end

function update_uprev!(integrator::OperatorSplittingIntegrator)
    RecursiveArrayTools.recursivecopy!(integrator.uprev, integrator.u)
    return nothing
end

# ---------------------------------------------------------------------------
# Controller interface — SplitSubIntegrator
# ---------------------------------------------------------------------------
function reject_step!(sub::SplitSubIntegrator)
    sub.u .= sub.uprev
    # Propagate rollback to all leaf DEIntegrators within this subtree so
    # their state is consistent before the next attempt.
    _rollback_subintegrator_tree!(sub.subintegrator_tree, sub.u_master)
end

function _rollback_subintegrator_tree!(subintegrators::Tuple, u_master)
    @unroll for child in subintegrators
        _rollback_child!(child, u_master)
    end
end

function _rollback_child!(child::SplitSubIntegrator, u_master)
    child.u .= child.uprev
    _rollback_subintegrator_tree!(child.subintegrator_tree, u_master)
end

function _rollback_child!(child::DEIntegrator, u_master)
    # The leaf integrator's uprev already holds the correct state because
    # forward_sync_internal! copies u_master into it before each sub-step.
    # Nothing to do here beyond letting the view aliasing keep things consistent.
    return nothing
end

function accept_step!(sub::SplitSubIntegrator)
    RecursiveArrayTools.recursivecopy!(sub.uprev, sub.u)
end

# ---------------------------------------------------------------------------
# step_header! / step_footer! — outermost integrator
# ---------------------------------------------------------------------------
function step_header!(integrator::OperatorSplittingIntegrator)
    # Accept or reject the step
    if !is_first_iteration(integrator)
        if should_accept_step(integrator)
            accept_step!(integrator)
        else # Step should be rejected and hence repeated
            reject_step!(integrator)
        end
    elseif integrator.u_modified # && integrator.iter == 0
        update_uprev!(integrator)
    end

    # Before stepping we might need to adjust the dt
    increment_iteration(integrator)
    # OrdinaryDiffEqCore.choose_algorithm!(integrator, integrator.cache)
    OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator)
    OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator)
    return integrator.force_stepfail = false
end

function footer_reset_flags!(integrator)
    return integrator.u_modified = false
end
function setup_validity_flags!(integrator, t_next)
    return integrator.isout = false #integrator.opts.isoutofdomain(integrator.u, integrator.p, t_next)
end
function fix_solution_buffer_sizes!(integrator, sol)
    resize!(integrator.sol.t, integrator.saveiter)
    resize!(integrator.sol.u, integrator.saveiter)
    return if !(integrator.sol isa SciMLBase.DAESolution)
        resize!(integrator.sol.k, integrator.saveiter_dense)
    end
end

function step_footer!(integrator::OperatorSplittingIntegrator)
    ttmp = integrator.t + tdir(integrator) * integrator.dt

    footer_reset_flags!(integrator)
    setup_validity_flags!(integrator, ttmp)

    if should_accept_step(integrator)
        integrator.last_step_failed = false
        integrator.tprev = integrator.t
        integrator.t = ttmp
        step_accept_controller!(integrator) # Noop for non-adaptive algorithms
    elseif integrator.force_stepfail
        if isadaptive(integrator)
            step_reject_controller!(integrator)
            OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
        elseif integrator.dtchangeable # Non-adaptive but can change dt
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
# called by DiffEqBase.solve
function SciMLBase.__solve(
        prob::OperatorSplittingProblem,
        alg::AbstractOperatorSplittingAlgorithm, args...; kwargs...
    )
    integrator = SciMLBase.__init(prob, alg, args...; kwargs...)
    return DiffEqBase.solve!(integrator)
end

# either called directly (after init), or by DiffEqBase.solve (via __solve)
function DiffEqBase.solve!(integrator::OperatorSplittingIntegrator)
    while !isempty(integrator.tstops)
        while tdir(integrator) * integrator.t < SciMLBase.first_tstop(integrator)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return
            __step!(integrator)
            step_footer!(integrator)
            if !SciMLBase.has_tstop(integrator)
                break
            end
        end
        OrdinaryDiffEqCore.handle_tstop!(integrator)
    end
    SciMLBase.postamble!(integrator)
    if integrator.sol.retcode != ReturnCode.Default
        return integrator.sol
    end
    return integrator.sol = SciMLBase.solution_new_retcode(
        integrator.sol, ReturnCode.Success
    )
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator)
    @timeit_debug "step!" if integrator.advance_to_tstop
        tstop = first_tstop(integrator)
        while !reached_tstop(integrator, tstop)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return
            __step!(integrator)
            step_footer!(integrator)
            if !SciMLBase.has_tstop(integrator)
                break
            end
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
    return OrdinaryDiffEqCore.handle_tstop!(integrator)
end

function SciMLBase.check_error(integrator::OperatorSplittingIntegrator)
    if !SciMLBase.successful_retcode(integrator.sol) &&
            integrator.sol.retcode != ReturnCode.Default
        return integrator.sol.retcode
    end

    verbose = true # integrator.opts.verbose

    if DiffEqBase.NAN_CHECK(integrator.dtcache) || DiffEqBase.NAN_CHECK(integrator.dt)
        if verbose
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        end
        return ReturnCode.DtNaN
    end

    return check_error_subintegrators(integrator, integrator.subintegrator_tree)
end

# Recurse over a tuple of children
function check_error_subintegrators(integrator, subintegrator_tree::Tuple)
    for sub in subintegrator_tree
        retcode = check_error_subintegrators(integrator, sub)
        if !SciMLBase.successful_retcode(retcode) && retcode != ReturnCode.Default
            return retcode
        end
    end
    return integrator.sol.retcode
end

# Leaf: read retcode from the DEIntegrator's solution
function check_error_subintegrators(integrator, sub::DEIntegrator)
    return SciMLBase.check_error(sub)
end

# Intermediate node: read retcode from the SplitSubIntegrator status object
function check_error_subintegrators(integrator, sub::SplitSubIntegrator)
    rc = sub.status.retcode
    if !SciMLBase.successful_retcode(rc) && rc != ReturnCode.Default
        return rc
    end
    # Also recurse into children
    return check_error_subintegrators(integrator, sub.subintegrator_tree)
end

function DiffEqBase.step!(integrator::OperatorSplittingIntegrator, dt, stop_at_tdt = false)
    return @timeit_debug "step!" begin
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
end

function setup_u(prob::OperatorSplittingProblem, solver, alias_u0)
    if alias_u0
        return prob.u0
    else
        return RecursiveArrayTools.recursivecopy(prob.u0)
    end
end

# TimeChoiceIterator API
@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator)
    return (integrator.tmp,)
end

# Interpolation
function linear_interpolation!(y, t, y1, y2, t1, t2)
    return y .= y1 + (t - t1) * (y2 - y1) / (t2 - t1)
end
function (integrator::OperatorSplittingIntegrator)(tmp, t)
    return linear_interpolation!(
        tmp, t, integrator.uprev, integrator.u, integrator.tprev, integrator.t
    )
end

# ---------------------------------------------------------------------------
# Stepsize controller hooks — outermost integrator
# ---------------------------------------------------------------------------
"""
    stepsize_controller!(::OperatorSplittingIntegrator)

Updates the controller using the current state of the integrator if the operator splitting algorithm is adaptive.
"""
@inline function stepsize_controller!(integrator::OperatorSplittingIntegrator)
    algorithm = integrator.alg
    isadaptive(algorithm) || return nothing
    return stepsize_controller!(integrator, algorithm)
end

"""
    step_accept_controller!(::OperatorSplittingIntegrator)

Updates `dtcache` of the integrator if the step is accepted and the operator splitting algorithm is adaptive.
"""
@inline function step_accept_controller!(integrator::OperatorSplittingIntegrator)
    algorithm = integrator.alg
    isadaptive(algorithm) || return nothing
    return step_accept_controller!(integrator, algorithm, nothing)
end

"""
    step_reject_controller!(::OperatorSplittingIntegrator)

Updates `dtcache` of the integrator if the step is rejected and the the operator splitting algorithm is adaptive.
"""
@inline function step_reject_controller!(integrator::OperatorSplittingIntegrator)
    algorithm = integrator.alg
    isadaptive(algorithm) || return nothing
    return step_reject_controller!(integrator, algorithm, nothing)
end

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
tdir(integrator) = integrator.tstops.ordering isa DataStructures.FasterForward ? 1 : -1
is_past_t(integrator, t) = tdir(integrator) * (t - integrator.t) ≤ zero(integrator.t)
function reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable)
    if stop_at_tstop
        integrator.t > tstop &&
            error("Integrator missed stop at $tstop (current time=$(integrator.t)). Aborting.")
        return integrator.t == tstop
    else
        return is_past_t(integrator, tstop)
    end
end

# ---------------------------------------------------------------------------
# SciMLBase integrator interface
# ---------------------------------------------------------------------------
function SciMLBase.done(integrator::OperatorSplittingIntegrator)
    if !(
            integrator.sol.retcode in (
                ReturnCode.Default, ReturnCode.Success,
            )
        )
        return true
    elseif isempty(integrator.tstops)
        SciMLBase.postamble!(integrator)
        return true
    end
    return false
end

function SciMLBase.postamble!(integrator::OperatorSplittingIntegrator)
    return DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
end

function __step!(integrator)
    tnext = integrator.t + integrator.dt
    synchronize_subintegrator_tree!(integrator)
    advance_solution_to!(integrator, tnext)
    return stepsize_controller!(integrator)
end

# solvers need to define this interface
function advance_solution_to!(integrator::OperatorSplittingIntegrator, tnext)
    return advance_solution_to!(integrator, integrator.cache, tnext)
end

function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, sync, cache, tend
    )
    # Advance a SplitSubIntegrator node using its own advance_solution_to! dispatch
    dt = tend - sub.t
    sub.dt = dt
    return advance_solution_to!(outer_integrator, sub, tend)
end

function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        integrator::DEIntegrator, solution_indices, sync, cache, tend
    )
    dt = tend - integrator.t
    return SciMLBase.step!(integrator, dt, true)
end

# ---------------------------------------------------------------------------
# SciMLBase.jl integrator interface
# ---------------------------------------------------------------------------
SciMLBase.has_stats(::OperatorSplittingIntegrator) = true

SciMLBase.has_tstop(integrator::OperatorSplittingIntegrator) = !isempty(integrator.tstops)
SciMLBase.first_tstop(integrator::OperatorSplittingIntegrator) = first(integrator.tstops)
SciMLBase.pop_tstop!(integrator::OperatorSplittingIntegrator) = pop!(integrator.tstops)

DiffEqBase.get_dt(integrator::OperatorSplittingIntegrator) = integrator.dt
function set_dt!(integrator::OperatorSplittingIntegrator, dt)
    dt <= zero(dt) && error("dt must be positive")
    return integrator.dt = dt
end

function DiffEqBase.add_tstop!(integrator::OperatorSplittingIntegrator, t)
    is_past_t(integrator, t) &&
        error("Cannot add a tstop at $t because that is behind the current \
               integrator time $(integrator.t)")
    return push!(integrator.tstops, t)
end

function DiffEqBase.add_saveat!(integrator::OperatorSplittingIntegrator, t)
    is_past_t(integrator, t) &&
        error("Cannot add a saveat point at $t because that is behind the \
               current integrator time $(integrator.t)")
    return push!(integrator.saveat, t)
end

DiffEqBase.u_modified!(i::OperatorSplittingIntegrator, bool) = nothing

# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------
function synchronize_subintegrator_tree!(integrator::OperatorSplittingIntegrator)
    return synchronize_subintegrator!(integrator.subintegrator_tree, integrator)
end

@unroll function synchronize_subintegrator!(
        subintegrator_tree::Tuple, integrator::OperatorSplittingIntegrator
    )
    @unroll for sub in subintegrator_tree
        synchronize_subintegrator!(sub, integrator)
    end
end

# Sync a SplitSubIntegrator node: update its t/dt then recurse into children
function synchronize_subintegrator!(
        sub::SplitSubIntegrator, integrator::OperatorSplittingIntegrator
    )
    (; t, dt) = integrator
    @assert sub.t == t "SplitSubIntegrator time $(sub.t) out of sync with outer integrator time $t"
    if !isadaptive(sub)
        sub.dt = dt
        sub.dtcache = dt
    end
    # Recurse: sync children against the *sub-integrator* (not outer) time
    @unroll for child in sub.subintegrator_tree
        synchronize_subintegrator_child!(child, sub)
    end
end

function synchronize_subintegrator_child!(
        child::DEIntegrator, parent::SplitSubIntegrator
    )
    @assert child.t == parent.t "Child integrator time $(child.t) out of sync with parent time $(parent.t)"
    if !isadaptive(child)
        SciMLBase.set_proposed_dt!(child, parent.dt)
    end
end

function synchronize_subintegrator_child!(
        child::SplitSubIntegrator, parent::SplitSubIntegrator
    )
    @assert child.t == parent.t "Nested SplitSubIntegrator time $(child.t) out of sync with parent time $(parent.t)"
    if !isadaptive(child)
        child.dt = parent.dt
        child.dtcache = parent.dt
    end
end

# ---------------------------------------------------------------------------
# advance_solution_to! for AbstractOperatorSplittingCache
# (dispatches into the algorithm-specific method in solver.jl)
# ---------------------------------------------------------------------------
function advance_solution_to!(
        integrator::OperatorSplittingIntegrator,
        cache::AbstractOperatorSplittingCache, tnext::Number
    )
    return advance_solution_to!(
        integrator, integrator.subintegrator_tree, cache, tnext
    )
end

# advance_solution_to! for a SplitSubIntegrator node
# (the algorithm-specific method in solver.jl calls this signature)
function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator, tend
    )
    return advance_solution_to!(
        outer_integrator, sub, sub.subintegrator_tree,
        sub.solution_index_tree, sub.synchronizer_tree,
        sub.cache, tend
    )
end

# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------
# Top-level dispatch: builds a Tuple of SplitSubIntegrators
function build_subintegrator_tree_with_cache(
        prob::OperatorSplittingProblem, alg::AbstractOperatorSplittingAlgorithm,
        uprevouter::AbstractVector, uouter::AbstractVector,
        u_master::AbstractVector,
        solution_indices,
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose
    )
    (; f, p) = prob

    subintegrator_tree_with_caches = ntuple(
        i -> build_subintegrator_tree_with_cache(
            prob,
            alg.inner_algs[i],
            get_operator(f, i),
            p[i],
            uprevouter, uouter,
            u_master,
            f.solution_indices[i],
            t0, dt, tf,
            tstops, saveat, d_discontinuities, callback,
            adaptive, verbose
        ),
        length(f.functions)
    )

    subintegrator_tree = ntuple(
        i -> subintegrator_tree_with_caches[i][1], length(f.functions)
    )
    caches = ntuple(i -> subintegrator_tree_with_caches[i][2], length(f.functions))

    return subintegrator_tree,
        init_cache(
            f, alg;
            uprev = uprevouter, u = uouter, alias_u = true,
            inner_caches = caches
        )
end

# Intermediate node: inner algorithm is also an AbstractOperatorSplittingAlgorithm
# wrapping a GenericSplitFunction → produce a SplitSubIntegrator
function build_subintegrator_tree_with_cache(
        prob::OperatorSplittingProblem, alg::AbstractOperatorSplittingAlgorithm,
        f::GenericSplitFunction, p::Tuple,
        uprevouter::AbstractVector, uouter::AbstractVector,
        u_master::AbstractVector,
        solution_indices,
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose,
        save_end = false,
        controller = nothing
    )
    tType = typeof(dt)

    # Build children recursively
    child_results = ntuple(
        i -> build_subintegrator_tree_with_cache(
            prob,
            alg.inner_algs[i],
            get_operator(f, i),
            p[i],
            uprevouter, uouter,
            u_master,
            f.solution_indices[i],
            t0, dt, tf,
            tstops, saveat, d_discontinuities, callback,
            adaptive, verbose
        ),
        length(f.functions)
    )

    child_subintegrators = ntuple(i -> child_results[i][1], length(f.functions))
    child_caches         = ntuple(i -> child_results[i][2], length(f.functions))

    # Build per-child solution_index_tree and synchronizer_tree
    child_solution_indices = ntuple(i -> f.solution_indices[i], length(f.functions))
    child_synchronizers    = ntuple(i -> f.synchronizers[i], length(f.functions))

    # Cache for *this* level
    u_sub    = @view uouter[solution_indices]
    uprev_sub = @view uprevouter[solution_indices]
    level_cache = init_cache(
        f, alg;
        uprev = uprev_sub, u = u_sub,
        inner_caches = child_caches
    )

    # EEst default
    EEst_val = isadaptive(alg) ? one(tType) : tType(NaN)

    sub = SplitSubIntegrator(
        alg,
        u_sub,
        RecursiveArrayTools.recursivecopy(u_sub),   # uprev: local copy for rollback
        u_master,
        t0,
        dt,
        dt,         # dtcache
        0,          # iter
        EEst_val,
        controller,
        false,      # force_stepfail
        false,      # last_step_failed
        SplitSubIntegratorStatus(),
        level_cache,
        child_subintegrators,
        solution_indices,
        child_solution_indices,
        child_synchronizers
    )

    return sub, level_cache
end

# Leaf node: inner algorithm is a plain SciMLBase.AbstractODEAlgorithm
# → produce an ODEIntegrator (existing behaviour)
function build_subintegrator_tree_with_cache(
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
    uprev = @view uprevouter[solution_indices]
    u = @view uouter[solution_indices]

    prob2 = if p isa NullParameters
        SciMLBase.ODEProblem(f, u, (t0, min(t0 + dt, tf)))
    else
        SciMLBase.ODEProblem(f, u, (t0, min(t0 + dt, tf)), p)
    end

    integrator = SciMLBase.__init(
        prob2,
        alg;
        dt,
        saveat = (),
        d_discontinuities,
        save_everystep = false,
        advance_to_tstop = false,
        adaptive,
        controller,
        verbose
    )

    return integrator, integrator.cache
end

# forward/backward sync no-ops for tuple nodes (handled inside SplitSubIntegrator)
function forward_sync_subintegrator!(
        outer_integrator::OperatorSplittingIntegrator, subintegrator_tree::Tuple,
        solution_indices::Tuple, synchronizers::Tuple
    )
    return nothing
end
function backward_sync_subintegrator!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrator_tree::Tuple, solution_indices::Tuple, synchronizer::Tuple
    )
    return nothing
end
