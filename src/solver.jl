# Lie-Trotter-Godunov Splitting Implementation
"""
    LieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

A first order sequential operator splitting algorithm attributed to [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
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
    _u = alias_u ? u : RecursiveArrayTools.recursivecopy(u)
    return LieTrotterGodunovCache(_u, _uprev, inner_caches)
end

# ---------------------------------------------------------------------------
# advance_solution_to! for the outermost integrator with a Tuple of children
# This is the top-level dispatch when the outer integrator's cache is a
# LieTrotterGodunovCache and children are SplitSubIntegrators or DEIntegrators.
# ---------------------------------------------------------------------------
@inline @unroll function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrators::Tuple, cache::LieTrotterGodunovCache, tnext
    )
    (; inner_caches) = cache
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        inner_cache = inner_caches[i]
        _advance_child!(outer_integrator, subinteg, inner_cache, tnext)
        # Check for failure after each child
        if _child_failed(outer_integrator, subinteg)
            outer_integrator.force_stepfail = true
            return
        end
    end
end

# ---------------------------------------------------------------------------
# advance_solution_to! for a SplitSubIntegrator node with LieTrotterGodunov
# This is the recursive dispatch when a SplitSubIntegrator's own cache is LTG.
# ---------------------------------------------------------------------------
@inline @unroll function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        sub::SplitSubIntegrator,
        subintegrators::Tuple,
        solution_indices::Tuple,
        synchronizers::Tuple,
        cache::LieTrotterGodunovCache,
        tnext
    )
    (; inner_caches) = cache
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        synchronizer = synchronizers[i]
        idxs = solution_indices[i]
        inner_cache = inner_caches[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(
            outer_integrator, subinteg, idxs, synchronizer
        )
        @timeit_debug "time solve" _advance_child!(
            outer_integrator, subinteg, inner_cache, tnext
        )
        if _child_failed(outer_integrator, subinteg)
            sub.status = SplitSubIntegratorStatus(ReturnCode.Failure)
            return
        end
        backward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
    end
    # All children succeeded: mark this node as successful
    sub.status = SplitSubIntegratorStatus(ReturnCode.Success)
    # Accept the sub-step: copy u into uprev for potential future rollback
    accept_step!(sub)
    sub.t = tnext
    sub.iter += 1
end

# ---------------------------------------------------------------------------
# _advance_child!: dispatch on child type
# ---------------------------------------------------------------------------

# Child is a SplitSubIntegrator: call its own advance_solution_to!
function _advance_child!(
        outer_integrator::OperatorSplittingIntegrator,
        child::SplitSubIntegrator,
        _inner_cache,   # ignored — child uses its own cache
        tnext
    )
    # Forward sync from the outer master u into this child's u
    forward_sync_subintegrator!(
        outer_integrator, child, child.solution_indices, NoExternalSynchronization()
    )
    advance_solution_to!(outer_integrator, child, tnext)
    backward_sync_subintegrator!(
        outer_integrator, child, child.solution_indices, NoExternalSynchronization()
    )
end

# Child is a leaf DEIntegrator
function _advance_child!(
        outer_integrator::OperatorSplittingIntegrator,
        child::DEIntegrator,
        _inner_cache,
        tnext
    )
    dt = tnext - child.t
    SciMLBase.step!(child, dt, true)
    # If the leaf adaptive integrator failed unrecoverably, error immediately
    if !SciMLBase.successful_retcode(child.sol.retcode) &&
            child.sol.retcode != ReturnCode.Default
        if isadaptive(child)
            error("Adaptive inner integrator failed unrecoverably with retcode $(child.sol.retcode). Aborting.")
        end
        # non-adaptive failure: signal to parent
    end
end

# ---------------------------------------------------------------------------
# _child_failed: check whether a child reported a failure
# ---------------------------------------------------------------------------
function _child_failed(outer_integrator, child::DEIntegrator)
    return child.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
end

function _child_failed(outer_integrator, child::SplitSubIntegrator)
    return child.status.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
end
