# ---------------------------------------------------------------------------
# Lie-Trotter-Godunov operator splitting
# ---------------------------------------------------------------------------
"""
    LieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

First-order sequential operator splitting algorithm attributed to
[Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
"""
struct LieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType
end

struct LieTrotterGodunovCache{uType, uprevType, iiType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    inner_caches::iiType
end

function init_cache(
        f::GenericSplitFunction, alg::LieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
        inner_caches,
        alias_uprev = true,
        alias_u = false
    )
    _uprev = alias_uprev ? uprev : RecursiveArrayTools.recursivecopy(uprev)
    _u     = alias_u     ? u     : RecursiveArrayTools.recursivecopy(u)
    return LieTrotterGodunovCache(_u, _uprev, inner_caches)
end

# ---------------------------------------------------------------------------
# advance_solution_to! for a SplitSubIntegrator node
#
# The SplitSubIntegrator is now the *parent* for its own children.
# It carries child_solution_indices and child_synchronizers directly.
#
# Entry point called from integrator.jl for a SplitSubIntegrator node
# ---------------------------------------------------------------------------
function advance_solution_to!(
        outer::OperatorSplittingIntegrator,
        children::Tuple,
        cache::AbstractOperatorSplittingCache,
        tnext
    )
    _perform_step!(outer, children, cache, tnext)

    if outer.force_stepfail
        outer.sol = SciMLBase.solution_new_retcode(
            outer.sol,
            ReturnCode.Failure
        )
        return
    end

    # All children succeeded: advance this node's time and counter
    # outer.sol = SciMLBase.solution_new_retcode(
    #     outer.sol,
    #     ReturnCode.Success
    # )
    return
end

function advance_solution_to!(
    outer::SplitSubIntegrator,
    children::Tuple,
    cache::AbstractOperatorSplittingCache,
    tnext
)
    _perform_step!(outer, children, cache, tnext)

    if outer.force_stepfail
        outer.status = SplitSubIntegratorStatus(ReturnCode.Failure)
        return
    end

    # All children succeeded: advance this node's time and counter
    outer.status  = SplitSubIntegratorStatus(ReturnCode.Success)

    return
end

@unroll function _perform_step!(
    outer,
    children::Tuple,
    cache::LieTrotterGodunovCache,
    tnext
)
    i = 0
    @unroll for child in children
        i += 1
        idxs = outer.child_solution_indices[i]
        sync = outer.child_synchronizers[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(outer, child, idxs, sync)
        @timeit_debug "time solve" _do_step!(outer, child, tnext)
        if _child_failed(child)
            outer.force_stepfail = true
            return
        end
        backward_sync_subintegrator!(outer, child, idxs, sync)
    end
end

# ---------------------------------------------------------------------------
# _do_step!: pure integration, no sync.
# The caller (advance_children_*) owns forward/backward sync around this.
# ---------------------------------------------------------------------------

# Leaf: DEIntegrator
function _do_step!(
        outer::OperatorSplittingIntegrator,
        child::DEIntegrator,
        tnext
    )
    dt = tnext - child.t
    SciMLBase.step!(child, dt, true)

    # Unrecoverable failure: error immediately regardless of adaptive/non-adaptive
    if !SciMLBase.successful_retcode(child.sol.retcode) &&
            child.sol.retcode != ReturnCode.Default
        error("Inner integrator failed unrecoverably with retcode \
               $(child.sol.retcode) at t=$(child.t). Aborting.")
    end
    return nothing
end

# Intermediate: SplitSubIntegrator — recurse
function _do_step!(
        outer::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator,
        tnext
    )
    # Sync sub's children among themselves before recursing.
    # (The parent already synced sub.u from master u via forward_sync before
    # calling _do_step!; here we propagate sub.dt down to sub's own children.)
    _sync_sub_children!(sub)
    advance_solution_to!(outer, sub, tnext)
    return nothing
end

# Propagate time-step information from sub to its own children
function _sync_sub_children!(sub::SplitSubIntegrator)
    _sync_sub_children_tuple!(sub.child_subintegrators, sub)
end

@unroll function _sync_sub_children_tuple!(children::Tuple, parent::SplitSubIntegrator)
    @unroll for child in children
        _sync_child_to_sub_parent!(child, parent)
    end
end

function _sync_child_to_sub_parent!(child::DEIntegrator, parent::SplitSubIntegrator)
    @assert child.t == parent.t "($(child.t) != $(parent.t))"
    if !isadaptive(child) && child.dtchangeable
        SciMLBase.set_proposed_dt!(child, parent.dt)
    end
end

function _sync_child_to_sub_parent!(child::SplitSubIntegrator, parent::SplitSubIntegrator)
    @assert child.t == parent.t "($(child.t) != $(parent.t))"
    if !isadaptive(child)
        SciMLBase.set_proposed_dt!(child, parent.dt)
    end
end

# ---------------------------------------------------------------------------
# _child_failed: check whether a child reported a failure
# ---------------------------------------------------------------------------
_child_failed(child::DEIntegrator) =
    child.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)

_child_failed(child::SplitSubIntegrator) =
    child.status.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
