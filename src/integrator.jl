mutable struct IntegratorStats
    naccept::Int64
    nreject::Int64
    # TODO inner solver stats
end

IntegratorStats() = IntegratorStats(0, 0)

Base.@kwdef mutable struct IntegratorOptions{tType, fType, vbType, F3, atolType, rtolType, normType}
    adaptive::Bool
    dtmin::tType = eps(Float64)
    dtmax::tType = Inf
    failfactor::fType = 4.0
    verbose::vbType = DEFAULT_VERBOSITY
    isoutofdomain::F3 = DiffEqBase.ODE_DEFAULT_ISOUTOFDOMAIN
    # Error control, used when the node's algorithm provides an error estimate. The
    # step size controller knobs (qmin, qmax, gamma, ...) are not held here: they
    # live in the node's controller cache, resolved by OrdinaryDiffEqCore.
    abstol::atolType = 1.0e-6
    reltol::rtolType = 1.0e-3
    internalnorm::normType = DiffEqBase.ODE_DEFAULT_NORM
end


"""
    SaveOptions

The saving settings of the outermost [`OperatorSplittingIntegrator`](@ref).

Saving is a property of the outer integrator only: the inner splits are stages
rather than steps, so there is no time point at which their state is a meaningful
approximation of the split solution. These settings are therefore consumed by
`__init` directly and never enter the [`ConfigTree`](@ref), which keeps them from
travelling down to the leaf integrators.

`saveat` itself is not held here -- it lives in the integrator's `saveat` heap,
because it is consumed as the integration proceeds.
"""
struct SaveOptions
    save_on::Bool
    save_everystep::Bool
    save_start::Bool
    save_end::Bool
end


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


"""
    SplitSubIntegrator <: AbstractODEIntegrator

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
  `dtchangeable`, `stops`
- `iter`                   — step counter at this level
- `success_iter`           — accepted-step counter at this level
- `EEst`                   — error estimate (`NaN` for non-adaptive, `1.0` default
                              for adaptive)
- `controller_cache`       — `OrdinaryDiffEqCore.AbstractControllerCache` holding the
                              step-size controller and its state (or `nothing` for
                              non-adaptive)
- `force_stepfail`         — flag: current step must be retried
- `last_step_failed`       — flag: previous step failed (double-failure detection)
- `status`                 — [`SplitSubIntegratorStatus`](@ref) for retcode communication
- `cache`                  — `AbstractOperatorSplittingCache` for the algorithm at
                              this level
- `child_subintegrators`   — tuple of direct children (`SplitSubIntegrator` or
                              `DEIntegrator`)
- `solution_indices`       — indices into the *parent's* solution vector **owned by
                              this node** (indices are parent-relative at every level)
- `child_solution_indices` — tuple of per-child indices into *this node's* `u`
- `child_synchronizers`    — tuple of per-child synchronizer objects
"""
mutable struct SplitSubIntegrator{
        algType,
        uType,
        tType,
        tstopsType,
        EEstType,
        controllerType,
        cacheType,
        childSubintType,
        solidxType,
        childSolidxType,
        childSyncType,
        optionsType,
    } <: SciMLBase.AbstractODEIntegrator{algType, true, uType, tType}
    alg::algType
    u::uType                        # local solution buffer
    uprev::uType                    # local rollback buffer
    u_master::uType                 # reference to outermost master u
    t::tType
    tprev::tType
    dt::tType
    dtcache::tType
    const dtchangeable::Bool
    tstops::tstopsType
    iter::Int
    success_iter::Int
    EEst::EEstType
    controller_cache::controllerType
    force_stepfail::Bool
    last_step_failed::Bool
    # Named after the field DiffEqBase's callback machinery reads directly.
    derivative_discontinuity::Bool
    status::SplitSubIntegratorStatus
    stats::IntegratorStats
    cache::cacheType
    child_subintegrators::childSubintType   # Tuple
    solution_indices::solidxType
    child_solution_indices::childSolidxType # Tuple
    child_synchronizers::childSyncType      # Tuple
    opts::optionsType
    tdir::tType
end

# --- SplitSubIntegrator interface ---

tdir(integrator::SplitSubIntegrator) = sign(integrator.dt)

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


"""
    OperatorSplittingIntegrator <: AbstractODEIntegrator

A variant of [`ODEIntegrator`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123)
to perform operator splitting.
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
        callbackCacheType,
        eventErrType,
        cacheType,
        solType,
        subintTreeType,
        childSolidxType,
        childSyncType,
        controllerType,
        optionsType,
        configType,
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
    # Scratch buffers a `VectorContinuousCallback` needs; `nothing` when the callback
    # set contains none.
    const callback_cache::callbackCacheType
    # Continuous-callback bookkeeping. DiffEqBase's root finder reads
    # `event_last_time`/`last_event_error` to avoid re-detecting the event it just
    # handled, and `handle_callbacks!` is responsible for writing them.
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::eventErrType
    advance_to_tstop::Bool
    last_step_failed::Bool
    force_stepfail::Bool
    isout::Bool
    # DiffEqBase's callback machinery reads this as a literal field (not through
    # `derivative_discontinuity!`), so the name has to match exactly.
    derivative_discontinuity::Bool
    just_hit_tstop::Bool
    cache::cacheType
    sol::solType
    # Saving state. `saveiter` is the authoritative length of the saved prefix of
    # `sol.t`/`sol.u`: `savevalues!` writes index `saveiter` and `postamble!`
    # truncates both vectors to it.
    const save_opts::SaveOptions
    saveiter::Int
    postamble_done::Bool
    # Tuple of SplitSubIntegrator nodes (one per top-level operator).
    child_subintegrators::subintTreeType
    child_solution_indices::childSolidxType # Tuple
    child_synchronizers::childSyncType      # Tuple
    iter::Int
    success_iter::Int
    # Step-size controller plus its state, or `nothing` when this node is not adaptive.
    controller_cache::controllerType
    EEst::tType  # error estimate of the last attempted step (NaN when no controller runs)
    opts::optionsType
    stats::IntegratorStats
    tdir::tType
    next_sync_is_continuous::Bool
    # Per-node settings, kept so that `reinit!` can restore them.
    config::configType
end

is_next_sync_continuous(integrator) = false
is_next_sync_continuous(integrator::OperatorSplittingIntegrator) = integrator.next_sync_is_continuous
mark_next_sync_continuous(integrator) = nothing
mark_next_sync_continuous(integrator::OperatorSplittingIntegrator) = integrator.next_sync_is_continuous = true
reset_next_sync_continuous(integrator) = nothing
reset_next_sync_continuous(integrator::OperatorSplittingIntegrator) = integrator.next_sync_is_continuous = false

const AnySplitIntegrator = Union{SplitSubIntegrator, OperatorSplittingIntegrator}

# DiffEqBase promotes the problem's `tspan` against the `dt` keyword before `__init`
# is ever called. A TreeOption is not a number, so hand it the root value: the step
# size the outermost integrator runs with.
function SciMLBase.promote_tspan(u0, p, tspan, prob::OperatorSplittingProblem, kwargs)
    dt = get(kwargs, :dt, nothing)
    dt === nothing && return tspan
    tspan1, tspan2, _ = promote(tspan..., _root_value(dt))
    return (tspan1, tspan2)
end
_root_value(opt::TreeOption) = opt.value
_root_value(v) = v

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
        save_on = true,
        save_everystep = false,
        save_start = true,
        save_end = true,
        dense = false,
        callback = nothing,
        advance_to_tstop = false,
        adaptive = nothing,
        controller = nothing,
        alias_u0 = false,
        verbose = true,
        kwargs...
    )
    (; u0, p) = prob
    t0, tf = prob.tspan

    # Dense output would need per-step interpolation data on top of the saved
    # points. The interpolant of a splitting step is linear (see the interpolation
    # section below), so `sol(t)` already reproduces it exactly from `sol.t`/`sol.u`
    # via SciMLBase's `LinearInterpolation` and storing `k` would buy nothing.
    dense && throw(
        ArgumentError(
            "dense output is not supported by operator splitting integrators. The \
            interpolant is linear, so `sol(t)` between saved points is already exact \
            for it; use `saveat` or `save_everystep = true` to control the saved points."
        )
    )
    save_opts = SaveOptions(save_on, save_everystep, save_start, save_end)

    # By default every node adapts exactly if its own algorithm does; a scalar or a
    # TreeOption overrides the whole tree explicitly.
    if adaptive === nothing
        adaptive = default_adaptive_option(prob.f, alg)
    end

    # Every setting is either one value for the whole tree or a TreeOption carrying a
    # value per node. Beyond the four this integrator handles itself, whatever the
    # caller passes travels down to the leaf integrators.
    config = build_config_tree(prob.f, (; dt, adaptive, verbose, controller, kwargs...))
    validate_dt_tree(config)
    tType = typeof(config.values.dt)
    config = signed_dt_tree(config, tf > t0 ? one(tType) : -one(tType), tType)
    warn_non_adaptive(alg, config)

    dt = config.values.dt
    dtcache = abs(dt)

    dtchangeable = isdtchangeable(alg)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Heaps store raw times and carry the integration direction in their ordering.
    # (OrdinaryDiffEqCore's initialize_tstops stores tdir-scaled times instead, which
    # is incompatible with the heaps reinit! rebuilds and breaks backward tspans.)
    tstops_internal, saveat_internal = tstops_and_saveat_heaps(
        t0, tf, (tstops..., d_discontinuities...), saveat
    )

    u = setup_u(prob, alg, alias_u0)
    uprev = setup_u(prob, alg, false)
    tmp = setup_u(prob, alg, false)
    uType = typeof(u)

    # `build_solution` defaults `interp` to a `LinearInterpolation` over the very
    # vectors passed in here, so `sol(t)` sees every later `savevalues!` push and
    # agrees exactly with the integrator's own (linear) interpolant. `stats` has to
    # be a real `DEStats`: the callback machinery increments `sol.stats.ncondition`.
    sol = SciMLBase.build_solution(
        prob, alg, tType[], uType[];
        stats = SciMLBase.DEStats(0),
        calculate_error = false,
    )
    callback = DiffEqBase.CallbackSet(callback)

    # Only a VectorContinuousCallback needs the scratch buffers, and they have to be
    # wide enough for the widest one in the set.
    max_len_cb = DiffEqBase.max_vector_callback_length_int(callback)
    eventErrType = real(eltype(u))
    callback_cache = if max_len_cb === nothing
        nothing
    else
        DiffEqBase.CallbackCache(u, max_len_cb, eventErrType, eventErrType)
    end

    child_subintegrators = build_subintegrators(
        prob, alg,
        uprev, u,
        u,            # u_master == u at the outermost level
        1:length(u),
        t0, tf,
        tstops, saveat, d_discontinuities, callback,
        config
    )

    cache = init_cache(
        prob.f, alg;
        uprev = uprev, u = u,
    )

    child_solution_indices = ntuple(i -> prob.f.solution_indices[i], length(prob.f.functions))
    child_synchronizers = ntuple(i -> prob.f.synchronizers[i], length(prob.f.functions))

    root_controller_cache = _node_controller_cache(alg, cache, config.values, tType)
    EEst = root_controller_cache === nothing ? tType(NaN) : one(tType)

    integrator = OperatorSplittingIntegrator(
        prob.f,
        alg,
        u, uprev, tmp,
        p,
        t0, t0,
        dt, dtcache,
        dtchangeable,
        tstops_internal, tstops,
        saveat_internal, saveat,
        callback,
        callback_cache,
        0, 1, zero(eventErrType),   # event_last_time, vector_event_last_time, last_event_error
        advance_to_tstop,
        false, false, false, false, false,
        cache, sol,
        save_opts, 0, false,
        child_subintegrators,
        child_solution_indices,
        child_synchronizers,
        0, 0,               # iter, success_iter
        root_controller_cache,
        EEst,
        split_integrator_options(config.values),
        IntegratorStats(),
        tType(tf > t0 ? 1 : -1),
        false,
        config,
    )
    # The initial save happens *after* the callbacks are initialized, so that an
    # initializer which modifies `u` is reflected in the first saved point.
    initialize_callbacks!(integrator)
    save_initial_value!(integrator)
    return integrator
end

"""
    initialize_callbacks!(integrator::OperatorSplittingIntegrator)

Run the callbacks' `initialize` hooks, mirroring OrdinaryDiffEqCore's
`initialize_callbacks!`.

The flag starts out set so that an `initialize` hook can clear it -- the default
initializer does exactly that -- and a hook that did modify `u` has its change
propagated to `uprev` and to the subintegrator tree before the first step.
"""
function initialize_callbacks!(integrator::OperatorSplittingIntegrator)
    integrator.derivative_discontinuity = true
    # Pass the integrator's own `u`, not the problem's `u0`: with `alias_u0 = false`
    # they are different arrays and a hook has to be able to modify the live state.
    modified = DiffEqBase.initialize!(
        integrator.callback, integrator.u, integrator.t, integrator
    )
    if modified
        update_uprev!(integrator)
        resync_children_after_modification!(integrator)
    end
    integrator.derivative_discontinuity = false
    return nothing
end

# A callback that changed `u` invalidates every child's copy of its slice.
# `rollback_children!` refills the children from the parent's `u`, re-anchors their
# clocks and marks their state modified so leaf integrators drop FSAL data. It also
# restores the `parent.u == child.u` invariant that lets the palindromic schemes skip
# the forward sync of their first child (`next_sync_is_continuous`).
function resync_children_after_modification!(integrator::OperatorSplittingIntegrator)
    rollback_children!(integrator)
    return nothing
end

function SciMLBase.reeval_internals_due_to_modification!(
        integrator::OperatorSplittingIntegrator, continuous_modification = true;
        callback_initializealg = nothing
    )
    resync_children_after_modification!(integrator)
    integrator.derivative_discontinuity = false
    return nothing
end

# `save_start` puts u(t0) at index 1, so the first `savevalues!` writes index 2.
function save_initial_value!(integrator::OperatorSplittingIntegrator)
    if integrator.save_opts.save_on && integrator.save_opts.save_start
        integrator.saveiter = 1
        RecursiveArrayTools.copyat_or_push!(integrator.sol.t, 1, integrator.t)
        RecursiveArrayTools.copyat_or_push!(integrator.sol.u, 1, integrator.u)
    else
        integrator.saveiter = 0
    end
    return nothing
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
        dt = nothing,
        erase_sol = false,
        tstops = integrator._tstops,
        saveat = integrator._saveat,
        reinit_callbacks = true,
        reinit_retcode = true
    )
    # The heap ordering types and every node's tdir are fixed at init, so a reinit!
    # cannot flip the integration direction.
    (tf > t0) == (integrator.tdir > 0) ||
        error("reinit! cannot change the direction of integration. Build a new integrator instead.")

    # Without a `dt` every node is restored to the step size it was configured with,
    # so a multi-rate setup survives a reinit!. Passing one is equivalent to passing
    # it to `init`: a scalar reconfigures the whole tree, a TreeOption node by node.
    if dt !== nothing
        integrator.config = _reconfigure_dt(integrator, dt, t0, tf)
    end
    config = integrator.config

    integrator.u .= u0
    integrator.uprev .= u0
    integrator.t = t0
    integrator.tprev = t0
    integrator.dt = config.values.dt
    integrator.dtcache = abs(config.values.dt)
    integrator.tstops, integrator.saveat =
        tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    integrator.iter = 0
    integrator.success_iter = 0
    reinit_node_controller!(integrator)
    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    integrator.event_last_time = 0
    integrator.vector_event_last_time = 1
    integrator.last_event_error = zero(integrator.last_event_error)
    if reinit_retcode
        integrator.sol = SciMLBase.solution_new_retcode(
            integrator.sol, ReturnCode.Default
        )
    end

    _subreinit_tuple!(
        integrator.f,
        u0,
        integrator.child_subintegrators,
        config;
        t0, tf,
        erase_sol, tstops, saveat,
        reinit_callbacks, reinit_retcode
    )
    # A leaf's `reinit!` restores it to the `u0` slice captured when it was built, not
    # to the `u0` handed to this call, so the new state has to be pushed down
    # explicitly. Relying on the forward sync of the next step is not enough: the
    # palindromic schemes skip it for their first child.
    resync_children_after_modification!(integrator)

    # After the children, so that an initializer which modifies `u` (and therefore
    # resyncs the tree) is not undone by the child reinit, and so that the first saved
    # point is the state the initializers left behind.
    if reinit_callbacks
        initialize_callbacks!(integrator)
    end
    # Saving is built into the integrator rather than provided by a saving callback,
    # so the initial save is redone regardless of `reinit_callbacks`. Stale entries
    # past the new `saveiter` are truncated by `postamble!`.
    integrator.postamble_done = false
    save_initial_value!(integrator)
    return nothing
end

# Rebuild the step sizes of the stored configuration from a `dt` given to reinit!.
function _reconfigure_dt(integrator::OperatorSplittingIntegrator, dt, t0, tf)
    tType = typeof(integrator.dt)
    dt_config = build_config_tree(integrator.f, (; dt))
    validate_dt_tree(dt_config)
    dt_config = signed_dt_tree(dt_config, tf > t0 ? one(tType) : -one(tType), tType)
    return _replace_dt(integrator.config, dt_config)
end

_replace_dt(config::ConfigTree, dt_config::ConfigTree) = ConfigTree(
    merge(config.values, (; dt = dt_config.values.dt)),
    ntuple(
        i -> _replace_dt(config.children[i], dt_config.children[i]),
        length(config.children)
    )
)

# --- subreinit! helpers ---

# Iterate over a tuple of children (outermost call from reinit!)
@unroll function _subreinit_tuple!(
        f,
        u0,
        children::Tuple,
        config::ConfigTree;
        kwargs...
    )
    i = 1
    @unroll for child in children
        _subreinit_child!(get_operator(f, i), u0, child, config.children[i]; kwargs...)
        i += 1
    end
end

# Reinitialise a leaf DEIntegrator child
function _subreinit_child!(
        f_child,
        u0,
        child::DEIntegrator,
        config::ConfigTree;
        kwargs...
    )
    if child.dtchangeable
        SciMLBase.set_proposed_dt!(child, config.values.dt)
        # Reinit does not touch this, so we reset it manually.
        set_dt!(child, config.values.dt)
    end
    DiffEqBase.reinit!(child; kwargs...)
    return nothing
end

# Reinitialise an intermediate SplitSubIntegrator child
function _subreinit_child!(
        f_child,
        u0,
        sub::SplitSubIntegrator,
        config::ConfigTree;
        t0,
        tf,
        kwargs...
    )
    sub.t = t0
    SciMLBase.set_proposed_dt!(sub, config.values.dt)
    set_dt!(sub, config.values.dt)
    sub.iter = 0
    sub.success_iter = 0
    sub.force_stepfail = false
    sub.last_step_failed = false
    sub.status = SplitSubIntegratorStatus(ReturnCode.Default)
    reinit_node_controller!(sub)
    # Recurse into this node's children
    _subreinit_tuple!(
        f_child,
        u0,
        sub.child_subintegrators,
        config;
        t0, tf, kwargs...
    )
    return nothing
end

# ---------------------------------------------------------------------------
# handle_tstop!
# ---------------------------------------------------------------------------
function OrdinaryDiffEqCore.handle_tstop!(integrator::AnySplitIntegrator)
    if SciMLBase.has_tstop(integrator)
        # The heaps store raw times; comparisons happen in tdir-space so that
        # "ahead"/"behind" is direction independent.
        tdir_t = tdir(integrator) * integrator.t
        tdir_tstop = tdir(integrator) * SciMLBase.first_tstop(integrator)
        if tdir_t == tdir_tstop
            while tdir_t == tdir_tstop
                SciMLBase.pop_tstop!(integrator)
                SciMLBase.has_tstop(integrator) ?
                    (tdir_tstop = tdir(integrator) * SciMLBase.first_tstop(integrator)) : break
            end
            notify_integrator_hit_tstop!(integrator)
        elseif tdir_t > tdir_tstop
            if !integrator.dtchangeable
                SciMLBase.change_t_via_interpolation!(
                    integrator,
                    SciMLBase.pop_tstop!(integrator),
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

notify_integrator_hit_tstop!(integrator::SplitSubIntegrator) = nothing
function notify_integrator_hit_tstop!(integrator::OperatorSplittingIntegrator)
    integrator.just_hit_tstop = true
    return nothing
end


# ---------------------------------------------------------------------------
# Step accept/reject
# ---------------------------------------------------------------------------
function reject_step!(integrator::AnySplitIntegrator)
    OrdinaryDiffEqCore.increment_reject!(integrator.stats)
    if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
    integrator.u .= integrator.uprev
    rollback_children!(integrator)
    return nothing
end

function should_accept_step(integrator::OperatorSplittingIntegrator)
    (integrator.force_stepfail || integrator.isout) && return false
    return should_accept_step(integrator, integrator.cache, integrator.controller_cache)
end
function should_accept_step(integrator::SplitSubIntegrator)
    integrator.force_stepfail && return false
    return should_accept_step(integrator, integrator.cache, integrator.controller_cache)
end
function should_accept_step(integrator::AnySplitIntegrator, cache, ::Nothing)
    return !(integrator.force_stepfail)
end
# An active controller additionally requires the error estimate to pass.
function should_accept_step(
        integrator::AnySplitIntegrator, cache,
        controller_cache::OrdinaryDiffEqCore.AbstractControllerCache
    )
    return accept_step_controller(integrator, controller_cache, integrator.alg)
end

# `stats.naccept` is counted in `step_footer!` (which sees every accepted attempt,
# including the final one); this header-side bookkeeping only prepares the next step.
function accept_step!(integrator::AnySplitIntegrator)
    integrator.success_iter += 1
    return accept_step!(integrator, integrator.cache, integrator.controller_cache)
end
function accept_step!(integrator::AnySplitIntegrator, cache, controller_cache)
    return store_previous_info!(integrator)
end
function store_previous_info!(integrator::AnySplitIntegrator)
    if length(integrator.uprev) > 0
        update_uprev!(integrator)
    end
    return nothing
end
function update_uprev!(integrator::AnySplitIntegrator)
    RecursiveArrayTools.recursivecopy!(integrator.uprev, integrator.u)
    return nothing
end

# Roll a node's children back to the state of the node itself: the local solution
# buffers are refilled from the parent's (already restored) `u` and the child clocks
# are moved back to the parent's time. Leaves get their `u` restored here as well,
# not only by the forward sync before the next solve: palindromic algorithms skip
# that sync for the first child (`next_sync_is_continuous`), so a rollback that left
# the leaf state stale would silently resume from the failed attempt.
rollback_children!(parent::AnySplitIntegrator) = _rollback_children!(
    parent.child_subintegrators, parent.child_solution_indices, parent.u, parent.t
)
@unroll function _rollback_children!(children::Tuple, solution_indices::Tuple, u_parent, t)
    i = 0
    @unroll for child in children
        i += 1
        rollback_child!(child, u_parent, solution_indices[i], t)
    end
end
function rollback_child!(child::SplitSubIntegrator, u_parent, solution_indices, t)
    child.u .= @view u_parent[solution_indices]
    RecursiveArrayTools.recursivecopy!(child.uprev, child.u)
    child.t = t
    child.tprev = t
    _reset_child_failure!(child)
    rollback_children!(child)
    return nothing
end
function rollback_child!(child::DEIntegrator, u_parent, solution_indices, t)
    child.u .= @view u_parent[solution_indices]
    RecursiveArrayTools.recursivecopy!(child.uprev, child.u)
    mark_state_modified!(child)
    child.t = t
    child.tprev = t
    _reset_child_failure!(child)
    return nothing
end

# Clear a failed child's retcode so the retry of the nearest adaptive ancestor can
# re-run it -- a transiently failed non-adaptive child would otherwise stay failed
# forever and turn every escalated failure fatal.
function _reset_child_failure!(child::SplitSubIntegrator)
    _child_failed(child) && (child.status.retcode = ReturnCode.Default)
    child.last_step_failed = false
    child.force_stepfail = false
    return nothing
end
function _reset_child_failure!(child::DEIntegrator)
    # Based on the *stored* retcode (the sticky part), not on `check_error`: the
    # rollback just restored this child's state, so a state-based check would
    # already come back clean while the stored retcode still blocks re-stepping.
    if child.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
        child.sol = SciMLBase.solution_new_retcode(child.sol, ReturnCode.Default)
    end
    # Resetting the retcode is not sufficient: SciMLBase's generic check_error
    # re-derives ConvergenceFailure from a sticky `last_stepfail` on non-adaptive
    # leaves (e.g. after an inner Newton failure), which would turn the retry the
    # escalation protocol just set up into an immediate failure again.
    if hasfield(typeof(child), :last_stepfail)
        child.last_stepfail = false
    end
    if hasfield(typeof(child), :force_stepfail)
        child.force_stepfail = false
    end
    return nothing
end

# ---------------------------------------------------------------------------
# step_header! / step_footer!
# ---------------------------------------------------------------------------
function step_header!(integrator::AnySplitIntegrator)
    if !is_first_iteration(integrator)
        if should_accept_step(integrator)
            accept_step!(integrator)
        else
            reject_step!(integrator)
        end
    elseif integrator.derivative_discontinuity
        update_uprev!(integrator)
    end
    increment_iteration(integrator)
    OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator)
    modify_dt_for_tstops!(integrator)
    integrator.force_stepfail = false
    return nothing
end

function modify_dt_for_tstops!(integrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = integrator.tdir * SciMLBase.first_tstop(integrator)
        if integrator.opts.adaptive
            integrator.dt = integrator.tdir *
                min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif iszero(integrator.dtcache) && integrator.dtchangeable
            integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
        elseif integrator.dtchangeable && !integrator.force_stepfail
            # always try to step! with dtcache, but lower if a tstop
            # however, if force_stepfail then don't set to dtcache, and no tstop worry
            integrator.dt = integrator.tdir *
                min(abs(integrator.dtcache), abs(tdir_tstop - tdir_t)) # step! to the end
        end
    end
    return
end

is_first_iteration(integrator::AnySplitIntegrator) = integrator.iter == 0
increment_iteration(integrator::AnySplitIntegrator) = integrator.iter += 1

function footer_reset_flags!(integrator)
    integrator.derivative_discontinuity = false
    integrator.just_hit_tstop = false
    # Re-arm the postamble: stepping on after a `solve!`/`done` (e.g. after pushing a
    # further tstop) has to be able to close out the solution again.
    integrator.postamble_done = false
    return
end
footer_reset_flags!(::SplitSubIntegrator) = nothing
function setup_validity_flags!(integrator, t_next)
    integrator.isout = false
    return
end
setup_validity_flags!(::SplitSubIntegrator, _) = nothing

# Window for absorbing floating point drift when landing on a time point. Scaled
# by the local time scale *and* the step size: near t = 0 (e.g. integrating
# backward to zero) a purely value-relative window collapses below the ulp drift
# the subdivided child steps accumulate.
_snap_window(t, tstop, dt) =
    100 * eps(float(max(abs(t), abs(tstop), abs(dt)) / oneunit(t))) * oneunit(t)

function fixed_t_for_floatingpoint_error!(integrator::AnySplitIntegrator, ttmp)
    return if DiffEqBase.has_tstop(integrator)
        tstop = DiffEqBase.first_tstop(integrator)
        if abs(ttmp - tstop) < _snap_window(integrator.t, tstop, integrator.dt)
            try_snap_children_to_tstop!.(integrator.child_subintegrators, tstop)
            tstop
        else
            ttmp
        end
    else
        ttmp
    end
end
function try_snap_children_to_tstop!(integrator::SplitSubIntegrator, tstop)
    if abs(tstop - integrator.t) < _snap_window(integrator.t, tstop, integrator.dt)
        integrator.t = tstop
    else
        @warn "Failed to snap timestep for integrator $(integrator.t) with parent integrator hitting the tstop $(tstop)."
    end
    return try_snap_children_to_tstop!.(integrator.child_subintegrators, tstop)
end
function try_snap_children_to_tstop!(integrator::DEIntegrator, tstop)
    return if abs(tstop - integrator.t) < _snap_window(integrator.t, tstop, integrator.dt)
        integrator.t = tstop
    else
        @warn "Failed to snap timestep for integrator $(integrator.t) with parent integrator hitting the tstop $(tstop)."
    end
end

function step_footer!(integrator::AnySplitIntegrator)
    ttmp = integrator.t + integrator.dt # dt is signed by the integration direction
    footer_reset_flags!(integrator)
    setup_validity_flags!(integrator, ttmp)
    if should_accept_step(integrator)
        OrdinaryDiffEqCore.increment_accept!(integrator.stats)
        integrator.last_step_failed = false
        integrator.tprev = integrator.t
        integrator.t = fixed_t_for_floatingpoint_error!(integrator, ttmp)
        # Children that step with subdivided dt (e.g. StrangMarchuk's `dt/2`
        # halves) accumulate ulp-level drift from the parent's exact `t`.
        # Re-anchor children to the parent's canonical time here so the
        # drift cannot accumulate across outer steps.
        try_snap_children_to_tstop!.(integrator.child_subintegrators, integrator.t)
        step_accept_controller!(integrator)
        validate_time_point(integrator)
        # Callbacks run here (outer integrator only) and also perform the saving.
        handle_callbacks!(integrator)
    elseif integrator.force_stepfail
        # Failure escalation protocol: the failing node's own adaptivity decides.
        fatal_rc = _fatal_child_retcode(integrator.child_subintegrators)
        if fatal_rc !== ReturnCode.Default
            # An *adaptive* child failed: it already exhausted its own step size
            # adaptation, so retrying on a smaller interval cannot help. Propagate
            # its diagnosis and stop.
            _set_retcode!(integrator, fatal_rc)
        elseif integrator.controller_cache !== nothing
            # A non-adaptive descendant failed and this is the nearest adaptive
            # ancestor: retry the step on a failfactor-shrunken interval, which
            # shrinks the effective dt of the whole subtree. The failed inner solve
            # tells us nothing about the error, so no step size law here. The
            # header-side reject_step! resets the subtree, including retcodes.
            OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
            integrator.dtcache = abs(integrator.dt)
            abort_below_dtmin!(integrator)
        else
            # Non-adaptive node: escalate the failure to this node's parent. At a
            # non-adaptive root this stops the time integration.
            _set_retcode!(
                integrator,
                _first_failed_child_retcode(integrator.child_subintegrators)
            )
        end
        integrator.last_step_failed = true
    else
        # The controller rejected the step (EEst > 1): shrink dt and retry.
        step_reject_controller!(integrator)
        abort_below_dtmin!(integrator)
        integrator.last_step_failed = true
    end
    return nothing
end

function abort_below_dtmin!(integrator::AnySplitIntegrator)
    abs(integrator.dt) > abs(OrdinaryDiffEqCore.timedepentdtmin(integrator)) && return nothing
    _is_verbose(integrator.opts.verbose) &&
        @warn("dt <= dtmin. Aborting. There is either an error in your model specification or the true solution is unstable.")
    _set_retcode!(integrator, ReturnCode.DtLessThanMin)
    return nothing
end

_set_retcode!(integrator::OperatorSplittingIntegrator, code) =
    integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, code)
_set_retcode!(integrator::SplitSubIntegrator, code) = integrator.status.retcode = code

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
        while tdir(integrator) * integrator.t <
                tdir(integrator) * SciMLBase.first_tstop(integrator)
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

function DiffEqBase.step!(integrator::AnySplitIntegrator)
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

# SciML convention: `dt` is signed and its sign has to match the direction of
# integration.
function DiffEqBase.step!(integrator::AnySplitIntegrator, dt, stop_at_tdt = false)
    @timeit_debug "step!" begin
        tdir(integrator) * dt < zero(dt) && error("Cannot step backward.")
        stop_at_tdt && !integrator.dtchangeable &&
            error("Cannot stop at t + dt if dtchangeable is false")
        tnext = integrator.t + dt
        stop_at_tdt && DiffEqBase.add_tstop!(integrator, tnext)
        while !reached_tstop(integrator, tnext, stop_at_tdt)
            step_header!(integrator)
            @timeit_debug "check_error" SciMLBase.check_error!(integrator) ∉ (
                ReturnCode.Success, ReturnCode.Default,
            ) && return
            __step!(integrator)
            step_footer!(integrator)
            # Pop every tstop as soon as it is reached, exactly like the solve!
            # loop does. Intermediate stops before `tnext` do occur -- the stale
            # `tnext` of an earlier failed attempt, or stops pushed down from the
            # root -- and leaving one in the heap once we sit exactly on it makes
            # the next header compute a zero step-to-tstop gap: the loop would
            # then spin at fixed `t` forever, growing the child tstop heaps.
            OrdinaryDiffEqCore.handle_tstop!(integrator)
        end
    end
    OrdinaryDiffEqCore.handle_tstop!(integrator)
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
        _is_verbose(integrator.opts.verbose) &&
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        return ReturnCode.DtNaN
    end
    return _check_error_children(integrator.sol.retcode, integrator.child_subintegrators)
end

function SciMLBase.check_error(integrator::SplitSubIntegrator)
    if !SciMLBase.successful_retcode(integrator.status.retcode) &&
            integrator.status.retcode != ReturnCode.Default
        return integrator.status.retcode
    end
    if DiffEqBase.NAN_CHECK(integrator.dtcache) || DiffEqBase.NAN_CHECK(integrator.dt)
        _is_verbose(integrator.opts.verbose) &&
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        return ReturnCode.DtNaN
    end
    return _check_error_children(integrator.status.retcode, integrator.child_subintegrators)
end

function SciMLBase.check_error!(integrator::SplitSubIntegrator)
    code = SciMLBase.check_error(integrator)
    integrator.status.retcode = code
    return code
end

# SciMLBase's generic `check_error!` finalizes the integrator whenever the code is
# anything but `Success`, but a mid-solve node legitimately reports `Default`: the
# stepping loops call this before every step, so the generic version would run the
# postamble -- callback finalizers, the endpoint save -- once per step. Only a real
# failure finalizes here.
function SciMLBase.check_error!(integrator::OperatorSplittingIntegrator)
    code = SciMLBase.check_error(integrator)
    # Rebuilding the solution allocates, and this runs before every step, so only
    # touch it when the code actually changed.
    if integrator.sol.retcode !== code
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, code)
    end
    if code ∉ (ReturnCode.Success, ReturnCode.Default)
        SciMLBase.postamble!(integrator)
    end
    return code
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

_child_retcode(child::DEIntegrator) = SciMLBase.check_error(child)
_child_retcode(child::SplitSubIntegrator) = child.status.retcode

function setup_u(prob::OperatorSplittingProblem, solver, alias_u0)
    return alias_u0 ? prob.u0 : RecursiveArrayTools.recursivecopy(prob.u0)
end

@inline function DiffEqBase.get_tmp_cache(integrator::OperatorSplittingIntegrator)
    return (integrator.tmp,)
end

# ---------------------------------------------------------------------------
# Interpolation
#
# A splitting step advances its children sequentially over staggered subintervals,
# so the children's own interpolants do not compose into an approximation of the
# split solution: the only dense output well defined at this level is the one built
# from the endpoints the outer integrator owns. The generic fallback is therefore
# linear, which is exact for the state a `LieTrotterGodunov` step produces and first
# order for the second order schemes. An algorithm that can do better implements
# `splitting_interpolant`/`splitting_interpolant!` for its own cache type; nothing
# else in the saving or callback machinery has to change.
# ---------------------------------------------------------------------------

"""
    splitting_interpolant(integrator, cache, Θ, dt, y₀, y₁, idxs, ::Type{Val{D}})
    splitting_interpolant!(out, integrator, cache, Θ, dt, y₀, y₁, idxs, ::Type{Val{D}})

Dense output of a single operator splitting step, evaluated in the step-local
coordinate `Θ = (t - tprev) / dt`, where `y₀`/`y₁` are the state at `tprev`/`t` and
`dt` is the step that was taken. `D` is the requested derivative order and `idxs`
selects components (`nothing` for all of them).

This is the extension point for higher order dense output: dispatch on the
algorithm's cache type. The fallback for any
[`AbstractOperatorSplittingCache`](@ref) is linear interpolation.
"""
splitting_interpolant, splitting_interpolant!

splitting_interpolant(
    integrator, ::AbstractOperatorSplittingCache, Θ, dt, y₀, y₁, idxs, ::Type{Val{D}}
) where {D} = SciMLBase.linear_interpolant(Θ, dt, y₀, y₁, idxs, Val{D})

function splitting_interpolant!(
        out, integrator, ::AbstractOperatorSplittingCache, Θ, dt, y₀, y₁, idxs,
        ::Type{Val{D}}
    ) where {D}
    SciMLBase.linear_interpolant!(out, Θ, dt, y₀, y₁, idxs, Val{D})
    return out
end

# The step that was just taken. `integrator.dt` must not be used here: by the time a
# step is finished `step_accept_controller!` has already overwritten it with the
# *next* proposal, so it is not the length of the interval `uprev`/`u` span.
_step_dt(integrator::OperatorSplittingIntegrator) = integrator.t - integrator.tprev

function _interp_theta(integrator::OperatorSplittingIntegrator, t)
    dt = _step_dt(integrator)
    # A zero-length interval only happens before the first step, where `uprev == u`
    # and every Θ gives the same value.
    return iszero(dt) ? zero(t / oneunit(dt)) : (t - integrator.tprev) / dt
end

function (integrator::OperatorSplittingIntegrator)(
        t::Number, ::Type{deriv} = Val{0}; idxs = nothing
    ) where {deriv}
    return splitting_interpolant(
        integrator, integrator.cache,
        _interp_theta(integrator, t), _step_dt(integrator),
        integrator.uprev, integrator.u, idxs, deriv
    )
end

function (integrator::OperatorSplittingIntegrator)(
        val::AbstractArray, t::Number, ::Type{deriv} = Val{0}; idxs = nothing
    ) where {deriv}
    return splitting_interpolant!(
        val, integrator, integrator.cache,
        _interp_theta(integrator, t), _step_dt(integrator),
        integrator.uprev, integrator.u, idxs, deriv
    )
end

# ---------------------------------------------------------------------------
# change_t_via_interpolation!
# ---------------------------------------------------------------------------

# DiffEqBase's continuous callback path passes the literal `Val{:false}` -- a `Val` of
# the *Symbol*, not of `false` -- so the flag has to be decoded by dispatch. Testing
# `if T` the way OrdinaryDiffEqCore does would throw on a Symbol.
_modify_save_endpoint(::Type{Val{true}}) = true
_modify_save_endpoint(::Type{Val{T}}) where {T} = false

"""
    change_t_via_interpolation!(integrator::OperatorSplittingIntegrator, t, modify_save_endpoint = Val{false}, reinitialize_alg = nothing)

Move the integrator back to a time `t` inside the step that was just taken.

`u` is refilled from the step's interpolant and the whole subintegrator tree is
re-anchored to `t` through [`rollback_children!`](@ref), so every child's state and
clock stay consistent with the parent (`validate_time_point` asserts the latter).

`integrator.dt` is deliberately left alone: it holds the step size proposed for the
next step, and the interpolation coordinate is derived from `t - tprev` rather than
from `dt`.
"""
function SciMLBase.change_t_via_interpolation!(
        integrator::OperatorSplittingIntegrator, t,
        modify_save_endpoint::Type{Val{T}} = Val{false},
        reinitialize_alg = nothing
    ) where {T}
    if integrator.tdir * t < integrator.tdir * integrator.tprev
        error("Current interpolant only works between tprev and t")
    elseif t != integrator.t
        integrator(integrator.u, t)
        integrator.t = t
        # Push the interpolated slices into the children and move their clocks back.
        rollback_children!(integrator)
        if _modify_save_endpoint(modify_save_endpoint)
            solution_endpoint_match_cur_integrator!(integrator)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Saving
#
# Only the outermost integrator saves: the inner splits are stages rather than
# steps, so their intermediate states are not approximations of the split solution
# at any time point.
#
# `saveiter` is the authoritative length of the saved prefix of `sol.t`/`sol.u`.
# `sol.interp` is a `LinearInterpolation` aliasing those same vectors, so `sol(t)`
# reproduces this integrator's own interpolant between saved points.
# ---------------------------------------------------------------------------

_saved_at_current_t(integrator::OperatorSplittingIntegrator) =
    integrator.saveiter > 0 &&
    integrator.sol.t[integrator.saveiter] == integrator.t

function _save_current!(integrator::OperatorSplittingIntegrator)
    integrator.saveiter += 1
    RecursiveArrayTools.copyat_or_push!(
        integrator.sol.t, integrator.saveiter, integrator.t
    )
    RecursiveArrayTools.copyat_or_push!(
        integrator.sol.u, integrator.saveiter, integrator.u
    )
    return nothing
end

"""
    savevalues!(integrator::OperatorSplittingIntegrator, force_save = false, reduce_size = true)

Save the state at the current time point and at any pending `saveat` points that the
step just taken has passed, returning `(saved, savedexactly)`.

`saveat` points strictly inside the step are filled from the step's interpolant
(see [`splitting_interpolant`](@ref)), so asking for output never changes the
sequence of steps and therefore never changes the splitting error. Because the
generic interpolant is linear, interpolated points are first order accurate even
when the splitting scheme is second order; pass the times as `tstops` instead to
have the integrator land on them exactly.
"""
function SciMLBase.savevalues!(
        integrator::OperatorSplittingIntegrator,
        force_save = false, reduce_size = true
    )::Tuple{Bool, Bool}
    saved = false
    savedexactly = false
    integrator.save_opts.save_on || return saved, savedexactly
    (; save_everystep, save_end) = integrator.save_opts
    tf = integrator.sol.prob.tspan[2]

    # The heaps store raw times and carry the direction in their ordering, so the
    # comparison is scaled by tdir rather than the stored value.
    tdir_t = integrator.tdir * integrator.t
    while !isempty(integrator.saveat) &&
            integrator.tdir * first(integrator.saveat) <= tdir_t
        curt = pop!(integrator.saveat)
        if curt == integrator.t
            # `save_end` owns the final point; leave it to the postamble.
            (!save_end && curt == tf) && continue
            saved = true
            savedexactly = true
            _save_current!(integrator)
        else
            saved = true
            integrator.saveiter += 1
            RecursiveArrayTools.copyat_or_push!(
                integrator.sol.t, integrator.saveiter, curt
            )
            # Freshly allocated by the interpolant, so hand over ownership.
            RecursiveArrayTools.copyat_or_push!(
                integrator.sol.u, integrator.saveiter, integrator(curt), false
            )
        end
    end

    if force_save || (
            save_everystep && !_saved_at_current_t(integrator) &&
                (save_end || integrator.t != tf)
        )
        saved = true
        savedexactly = true
        _save_current!(integrator)
    end

    return saved, savedexactly
end

# Make sure the final time point is in the solution. Mirrors OrdinaryDiffEqCore's
# `solution_endpoint_match_cur_integrator!`.
function solution_endpoint_match_cur_integrator!(integrator::OperatorSplittingIntegrator)
    integrator.save_opts.save_on || return nothing
    integrator.save_opts.save_end || return nothing
    _saved_at_current_t(integrator) && return nothing
    _save_current!(integrator)
    return nothing
end

# ---------------------------------------------------------------------------
# Callbacks
#
# Callbacks run on the outermost integrator only. A condition evaluated between two
# inner splits would see a state that is not an approximation of the split solution
# at any time point -- the splits are stages, not steps -- so there is nothing
# meaningful for a condition or an `affect!` to act on down there.
#
# The heavy lifting (root finding, `save_positions`, `affect!` dispatch) is
# DiffEqBase's; this is the per-step driver, modelled on OrdinaryDiffEqCore's
# `handle_callbacks!`. It also owns the plain `saveat` save when no callback saved.
# ---------------------------------------------------------------------------

# Type-stable dispatch into `DiffEqBase.apply_callback!`: the explicit
# `if (return) else (rest)` chain compiles to a switch over a heterogeneous tuple.
@generated function apply_ith_callback!(
        integrator,
        time, upcrossing, event_idx, cb_idx,
        callbacks::NTuple{
            N,
            Union{DiffEqBase.ContinuousCallback, DiffEqBase.VectorContinuousCallback},
        }
    ) where {N}
    ex = quote
        throw(BoundsError(callbacks, cb_idx))
    end
    for i in 1:N
        ex = quote
            if (cb_idx == $i)
                return DiffEqBase.apply_callback!(
                    integrator, callbacks[$i], time, upcrossing, event_idx
                )
            else
                $ex
            end
        end
    end
    return ex
end

handle_callbacks!(::SplitSubIntegrator) = nothing

function handle_callbacks!(integrator::OperatorSplittingIntegrator)
    discrete_callbacks = integrator.callback.discrete_callbacks
    continuous_callbacks = integrator.callback.continuous_callbacks

    continuous_modified = false
    discrete_modified = false
    saved_in_cb = false

    if !(continuous_callbacks isa Tuple{})
        time, upcrossing, event_occurred, event_idx, idx, _counter =
            DiffEqBase.find_first_continuous_callback(
            integrator, continuous_callbacks...
        )
        if event_occurred
            integrator.event_last_time = idx
            integrator.vector_event_last_time = event_idx
            continuous_modified, saved_in_cb = apply_ith_callback!(
                integrator, time, upcrossing, event_idx, idx, continuous_callbacks
            )
            # The step was cut at the event and the state may have jumped, so the
            # controller's error history no longer describes what happens next.
            reinit_node_controller!(integrator)
        else
            # Clearing these is what stops the root finder from nudging `tprev` on a
            # step that had no event.
            integrator.event_last_time = 0
            integrator.vector_event_last_time = 1
        end
    end

    if !integrator.force_stepfail && !(discrete_callbacks isa Tuple{})
        discrete_modified, saved_in_cb = DiffEqBase.apply_discrete_callback!(
            integrator, discrete_callbacks...
        )
    end

    if !saved_in_cb
        SciMLBase.savevalues!(integrator)
    end

    integrator.derivative_discontinuity = continuous_modified | discrete_modified
    return nothing
end

# ---------------------------------------------------------------------------
# Step size control
#
# A splitting node runs a controller only if it is adaptive and its algorithm
# provides an error estimate (written to the node's `EEst` by `_perform_step!`);
# its `controller_cache` is `nothing` otherwise. The controllers themselves are
# the OrdinaryDiffEqCore ones: `setup_controller_cache` resolves a controller's
# knobs against this algorithm and mints the mutable per-solve state that
# `stepsize_controller!`/`step_accept_controller!`/`step_reject_controller!`
# thread between the steps (e.g. `errold` of a PIController). Following the
# OrdinaryDiffEq conventions the step is accepted if `EEst <= 1`.
# ---------------------------------------------------------------------------

const CONTROLLER_KNOB_KEYS = (:qmin, :qmax, :gamma, :qsteady_min, :qsteady_max, :failfactor)

"""
    default_controller(alg::AbstractOperatorSplittingAlgorithm, values::NamedTuple)

The step size controller an adaptive splitting node runs with when `init` is not
given an explicit `controller`. The controller knobs the caller passed to `init`
(`qmin`, `qmax`, `gamma`, `qsteady_min`, `qsteady_max`, `failfactor`) ride along
as overrides; unset ones resolve to the algorithm defaults in
`setup_controller_cache`.
"""
default_controller(::AbstractOperatorSplittingAlgorithm, values::NamedTuple) =
    OrdinaryDiffEqCore.IController(
    NamedTuple{filter(in(CONTROLLER_KNOB_KEYS), keys(values))}(values)
)

function _node_controller(alg::AbstractOperatorSplittingAlgorithm, values::NamedTuple)
    (values.adaptive && SciMLBase.isadaptive(alg)) || return nothing
    values.controller === nothing || return values.controller
    return default_controller(alg, values)
end

function _node_controller_cache(
        alg::AbstractOperatorSplittingAlgorithm, level_cache,
        values::NamedTuple, ::Type{tType}
    ) where {tType}
    controller = _node_controller(alg, values)
    controller === nothing && return nothing
    if _wants_discontinuity_detection(controller)
        throw(
            ArgumentError(
                "discontinuity_detection is not supported by operator splitting nodes. \
                Construct the controller for this node without it."
            )
        )
    end
    return OrdinaryDiffEqCore.setup_controller_cache(alg, level_cache, controller, tType)
end

# The discontinuity handling of OrdinaryDiffEqCore's controllers needs integrator
# state (callbacks, checkpoints) a splitting node does not have, so refuse it
# up front instead of failing deep inside a rejected step.
_wants_discontinuity_detection(controller) =
    hasfield(typeof(controller), :basic) && _basic_discontinuity_detection(controller.basic)
_basic_discontinuity_detection(basic::NamedTuple) =
    get(basic, :discontinuity_detection, false) === true
_basic_discontinuity_detection(basic) =
    hasfield(typeof(basic), :discontinuity_detection) && basic.discontinuity_detection

reinit_node_controller!(integrator::AnySplitIntegrator) =
    reinit_node_controller!(integrator, integrator.controller_cache)
function reinit_node_controller!(integrator::AnySplitIntegrator, ::Nothing)
    integrator.EEst = oftype(integrator.EEst, NaN)
    return nothing
end
function reinit_node_controller!(
        integrator::AnySplitIntegrator,
        controller_cache::OrdinaryDiffEqCore.AbstractControllerCache
    )
    integrator.EEst = one(integrator.EEst)
    OrdinaryDiffEqCore.reinit_controller!(integrator, controller_cache)
    return nothing
end

"""
    alg_adaptive_order(alg::AbstractOperatorSplittingAlgorithm)

Order of the error estimator of an adaptive operator splitting algorithm; every
algorithm with `SciMLBase.isadaptive(alg) == true` has to implement this.
"""
function alg_adaptive_order end

# The controller caches consume the error estimate and the estimator order through
# these OrdinaryDiffEqCore hooks. Our nodes keep `EEst` on the integrator itself.
@inline OrdinaryDiffEqCore.get_EEst(integrator::AnySplitIntegrator) = integrator.EEst
@inline OrdinaryDiffEqCore.set_EEst!(integrator::AnySplitIntegrator, val) =
    integrator.EEst = oftype(integrator.EEst, val)
OrdinaryDiffEqCore.get_current_adaptive_order(
    alg::AbstractOperatorSplittingAlgorithm, cache
) = alg_adaptive_order(alg)
# The generic fallbacks of these two are meant for inner solvers with their own
# tuning (gamma_default falls back to 0, which would ruin the step size law).
OrdinaryDiffEqCore.gamma_default(::AbstractOperatorSplittingAlgorithm) = 9 // 10
OrdinaryDiffEqCore.failfactor_default(::AbstractOperatorSplittingAlgorithm) = 4

@inline step_accept_controller!(integrator::AnySplitIntegrator) =
    step_accept_controller!(integrator, integrator.controller_cache)
step_accept_controller!(integrator::AnySplitIntegrator, ::Nothing) = nothing
function step_accept_controller!(
        integrator::AnySplitIntegrator,
        controller_cache::OrdinaryDiffEqCore.AbstractControllerCache
    )
    q = stepsize_controller!(integrator, controller_cache, integrator.alg)
    dtnew = step_accept_controller!(integrator, controller_cache, integrator.alg, q)
    # The proposal derives from the step actually taken -- which
    # `modify_dt_for_tstops!` may have clipped to a tstop. After such a step the
    # proposal is rebased and regrows at up to qmax per step; that matches what
    # the error-based law knows (EEst belongs to the clipped step) and mirrors
    # OrdinaryDiffEq. `dtcache` mirrors the standing proposal for `reinit!` and
    # introspection.
    integrator.dt = dtnew
    integrator.dtcache = abs(dtnew)
    return nothing
end

@inline step_reject_controller!(integrator::AnySplitIntegrator) =
    step_reject_controller!(integrator, integrator.controller_cache)
step_reject_controller!(integrator::AnySplitIntegrator, ::Nothing) = nothing
function step_reject_controller!(
        integrator::AnySplitIntegrator,
        controller_cache::OrdinaryDiffEqCore.AbstractControllerCache
    )
    stepsize_controller!(integrator, controller_cache, integrator.alg)
    step_reject_controller!(integrator, controller_cache, integrator.alg) # sets dt
    integrator.dtcache = abs(integrator.dt)
    return nothing
end


# Time helpers
tdir(integrator) =
    integrator.tstops.ordering isa BinaryHeaps.FasterForward ? 1 : -1
is_past_t(integrator, t) =
    tdir(integrator) * (t - integrator.t) ≤ zero(integrator.t)
function reached_tstop(integrator, tstop, stop_at_tstop = integrator.dtchangeable)
    if stop_at_tstop
        tdir(integrator) * (integrator.t - tstop) > zero(integrator.t) &&
            error("Integrator missed stop at $tstop (current time=$(integrator.t)). Aborting.")
        return integrator.t ≈ tstop
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
    # `solve!`, `done` and `check_error!` all call this; only the first one after a
    # step may run the finalizers and close out the solution.
    integrator.postamble_done && return nothing
    integrator.postamble_done = true
    DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    solution_endpoint_match_cur_integrator!(integrator)
    # `saveiter` is authoritative: drop whatever a previous, longer run left behind
    # (a `reinit!` without `erase_sol`).
    resize!(integrator.sol.t, integrator.saveiter)
    resize!(integrator.sol.u, integrator.saveiter)
    return nothing
end

function __step!(integrator::AnySplitIntegrator)
    advance_solution_by!(integrator, integrator.dt)
    return nothing
end

# Entry point: dispatch to the algorithm's advance_solution_by!
function advance_solution_by!(integrator::AnySplitIntegrator, dt)
    return advance_solution_by!(integrator, integrator.cache, dt)
end

# Algorithm-level dispatch (implemented in solver.jl per algorithm)
function advance_solution_by!(
        integrator::AnySplitIntegrator,
        cache::AbstractOperatorSplittingCache, dt
    )
    return advance_solution_by!(
        integrator, integrator.child_subintegrators, cache, dt
    )
end

# ---------------------------------------------------------------------------
# advance_solution_by! for a SplitSubIntegrator node
#
# The SplitSubIntegrator is now the *parent* for its own children.
# It carries child_solution_indices and child_synchronizers directly.
#
# Entry point called from integrator.jl for a SplitSubIntegrator node
# ---------------------------------------------------------------------------
function advance_solution_by!(
        outer::OperatorSplittingIntegrator,
        children::Tuple,
        cache::AbstractOperatorSplittingCache,
        dt
    )
    # Success and failure are both handled in step_footer! via the failure
    # escalation protocol; nothing to decide here.
    _perform_step!(outer, children, cache, dt)
    return
end

# Retcode of the first failed child whose failure is fatal (the child is adaptive,
# so it already exhausted its own adaptation); `ReturnCode.Default` if none is.
@unroll function _fatal_child_retcode(children::Tuple)
    @unroll for child in children
        if _child_failed(child) && _child_is_adaptive(child)
            return _failure_retcode(child)
        end
    end
    return ReturnCode.Default
end

@unroll function _first_failed_child_retcode(children::Tuple)
    @unroll for child in children
        _child_failed(child) && return _failure_retcode(child)
    end
    return ReturnCode.Failure
end
_failure_retcode(child::DEIntegrator) = SciMLBase.check_error(child)
_failure_retcode(child::SplitSubIntegrator) = child.status.retcode
_child_is_adaptive(child::DEIntegrator) = child.opts.adaptive
_child_is_adaptive(child::SplitSubIntegrator) = child.controller_cache !== nothing

function advance_solution_by!(
        outer::SplitSubIntegrator,
        children::Tuple,
        cache::AbstractOperatorSplittingCache,
        dt
    )
    _perform_step!(outer, children, cache, dt)

    # On force_stepfail the status is left clean: step_footer! either retries at
    # this level (adaptive) or escalates by writing the failure into the status.
    if !outer.force_stepfail
        outer.status.retcode = ReturnCode.Success
    end

    return
end

# `dt` stays signed through the splitting tree, following the SciML `step!`
# convention: the sign has to match the child's own integration direction, which
# equals the tree's. Advancing a child *against* its direction (negative substeps
# of higher order compositions) is not supported yet: leaf ODEIntegrators cannot
# step against the tdir their tspan fixed at construction.

# Recursion dispatch
function advance_solution_by!(
        outer::AnySplitIntegrator,
        sub::SplitSubIntegrator,
        dt
    )
    SciMLBase.step!(sub, dt, true)
    return nothing
end

# Leaf dispatch
function advance_solution_by!(outer::AnySplitIntegrator, child::DEIntegrator, dt)
    SciMLBase.step!(child, dt, true)
    return nothing
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
        t0, tf,
        tstops, saveat, d_discontinuities, callback,
        config::ConfigTree
    )
    (; f, p) = prob

    child_subintegrators = ntuple(
        i -> _build_child(
            prob,
            alg.inner_algs[i],
            get_operator(f, i),
            p[i],
            uprevouter, uouter, u_master,
            f.solution_indices[i],
            t0, tf,
            tstops, saveat, d_discontinuities, callback,
            config.children[i]
        ),
        length(f.functions)
    )

    return child_subintegrators
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
        t0, tf,
        tstops, saveat, d_discontinuities, callback,
        config::ConfigTree
    )
    dt = config.values.dt
    tType = typeof(dt)

    u_sub = RecursiveArrayTools.recursivecopy(uouter[solution_indices])
    uprev_sub = RecursiveArrayTools.recursivecopy(uprevouter[solution_indices])

    # Recurse: build each consecutive child. Solution indices are relative to the
    # *parent* at every level, so the children have to address this node's buffers,
    # not the outer ones: a child that wires itself as a view into the handed
    # buffer (instead of copying, as the stock leaves do) would otherwise alias
    # the wrong slots of the root vector.
    child_subintegrators = ntuple(
        i -> _build_child(
            prob,
            alg.inner_algs[i],
            get_operator(f, i),
            p[i],
            uprev_sub, u_sub, u_master,
            f.solution_indices[i],
            t0, tf,
            tstops, saveat, d_discontinuities, callback,
            config.children[i]
        ),
        length(f.functions)
    )

    child_solution_indices = ntuple(i -> f.solution_indices[i], length(f.functions))
    child_synchronizers = ntuple(i -> f.synchronizers[i], length(f.functions))

    tstops_internal, _ = tstops_and_saveat_heaps(
        t0, tf, (tstops..., d_discontinuities...), ()
    )

    level_cache = init_cache(
        f, alg;
        uprev = uprev_sub, u = u_sub,
    )

    controller_cache = _node_controller_cache(alg, level_cache, config.values, tType)
    EEst_val = controller_cache === nothing ? tType(NaN) : one(tType)

    sub = SplitSubIntegrator(
        alg,
        u_sub,
        uprev_sub,
        u_master,
        t0, t0, dt, dt,     # t, tprev, dt, dtcache
        isdtchangeable(alg),
        tstops_internal,
        0, 0,           # iter, success_iter
        EEst_val,
        controller_cache,
        false, false, false,  # force_stepfail, last_step_failed, derivative_discontinuity
        SplitSubIntegratorStatus(),
        IntegratorStats(),
        level_cache,
        child_subintegrators,
        solution_indices,
        child_solution_indices,
        child_synchronizers,
        split_integrator_options(config.values),
        sign(dt),  # dt was signed by signed_dt_tree, so its sign is the direction
    )

    return sub
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
        t0::T, tf::T,
        tstops, saveat, d_discontinuities, callback,
        config::ConfigTree
    ) where {S, T, P, F}
    u = uouter[solution_indices]
    u0 = if f isa SciMLBase.AbstractSciMLFunction
        u
    else
        variable_symbols(f) .=> u
    end
    # MTK v11 compiled systems require a symbolic map for u0; plain SciMLFunctions accept arrays.
    prob2 = if p isa NullParameters
        SciMLBase.ODEProblem(f, u0, (t0, tf))
    else
        SciMLBase.ODEProblem(f, u0, (t0, tf), p)
    end

    integrator = SciMLBase.__init(
        prob2, alg;
        dt = config.values.dt,
        tstops,
        saveat = (),
        d_discontinuities,
        save_everystep = false,
        advance_to_tstop = false,
        adaptive = config.values.adaptive,
        controller = config.values.controller,
        verbose = _inner_verbose(config.values.verbose),
        inner_values(config.values)...
    )
    return integrator
end

# ---------------------------------------------------------------------------
# Tree addressing
#
# Only `SplitNode` addresses are accepted here. SciMLBase already gives
# `integrator[i]` the meaning "the i-th state component" via symbolic indexing
# (`Base.getindex(::DEIntegrator, sym)`), so integer indexing is left alone.
# ---------------------------------------------------------------------------
function _tree_child(integrator::AnySplitIntegrator, i::Int)
    children = integrator.child_subintegrators
    checkbounds(Bool, 1:length(children), i) || throw(
        ArgumentError(
            "operator $i is out of range: this splitting node has $(length(children)) subintegrators."
        )
    )
    return children[i]
end

Base.getindex(integrator::AnySplitIntegrator, node::SplitNode) =
    _resolve(integrator, node.path)

# ---------------------------------------------------------------------------
# SciMLBase API
# ---------------------------------------------------------------------------
SciMLBase.has_stats(::AnySplitIntegrator) = true

SciMLBase.has_tstop(i::AnySplitIntegrator) = !isempty(i.tstops)
SciMLBase.first_tstop(i::AnySplitIntegrator) = first(i.tstops)
SciMLBase.pop_tstop!(i::AnySplitIntegrator) = pop!(i.tstops)

DiffEqBase.get_dt(i::AnySplitIntegrator) = i.dt

# Continuous callbacks relax the step size through this after an event.
# `dtcache` mirrors the standing proposal as a magnitude (see `__init`).
function SciMLBase.set_proposed_dt!(integrator::OperatorSplittingIntegrator, dt)
    if integrator.dtcache != abs(dt)
        integrator.dtcache = abs(dt)
        if !isadaptive(integrator)
            integrator.dt = dt
        end
    end
    return nothing
end

"""
    terminate!(integrator::OperatorSplittingIntegrator, retcode = ReturnCode.Terminated)

Stop the integration from inside a callback's `affect!`.

Emptying the outermost `tstops` heap is what ends the `solve!` loop and makes
`done` report completion; the children's heaps are irrelevant once the outer loop
exits, and `postamble!` still closes out the solution.
"""
function SciMLBase.terminate!(
        integrator::OperatorSplittingIntegrator, retcode = ReturnCode.Terminated
    )
    _set_retcode!(integrator, retcode)
    empty!(integrator.tstops)
    return nothing
end

function set_dt!(i::DEIntegrator, dt)
    iszero(dt) && error("dt must be nonzero")
    return i.dt = dt
end

function DiffEqBase.add_tstop!(i::AnySplitIntegrator, t)
    is_past_t(i, t) &&
        error("Cannot add a tstop at $t because that is behind the current \
               integrator time $(i.t)")
    DiffEqBase.add_tstop!.(i.child_subintegrators, t)
    push!(i.tstops, t)
    return nothing
end

function DiffEqBase.add_saveat!(i::OperatorSplittingIntegrator, t)
    is_past_t(i, t) &&
        error("Cannot add a saveat point at $t because that is behind the \
               current integrator time $(i.t)")
    push!(i.saveat, t)
    return nothing
end

# SciMLBase v3 renamed `u_modified!` → `derivative_discontinuity!`. The flag itself
# is stored in the `derivative_discontinuity` field, which is the name DiffEqBase's
# callback machinery reads directly.
@static if isdefined(DiffEqBase, :u_modified!)
    DiffEqBase.u_modified!(i::AnySplitIntegrator, bool) = i.derivative_discontinuity = bool
end
@static if isdefined(SciMLBase, :derivative_discontinuity!)
    SciMLBase.derivative_discontinuity!(i::AnySplitIntegrator, bool) =
        i.derivative_discontinuity = bool
end
