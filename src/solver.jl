# Lie-Trotter-Godunov Splitting Implementation
"""
    LieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

A first order sequential operator splitting algorithm attributed to [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
"""
struct LieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
    # transfer_algs::TransferTupleType # Tuple of transfer algorithms from the master solution into the individual ones
end

struct LieTrotterGodunovCache{uType, uprevType, iiType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    inner_caches::iiType
end

function init_cache(f::GenericSplitFunction, alg::LieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
        inner_caches,
        alias_uprev = true,
        alias_u = false
)
    _uprev = alias_uprev ? uprev : SciMLBase.recursivecopy(uprev)
    _u = alias_u ? u : SciMLBase.recursivecopy(u)
    LieTrotterGodunovCache(_u, _uprev, inner_caches)
end

@inline @unroll function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrators::Tuple, solution_indices::Tuple,
        synchronizers::Tuple, cache::LieTrotterGodunovCache, tnext)
    # We assume that the integrators are already synced
    @unpack inner_caches = cache
    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        synchronizer = synchronizers[i]
        idxs = solution_indices[i]
        cache = inner_caches[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
        @timeit_debug "time solve" advance_solution_to!(
            outer_integrator, subinteg, idxs, synchronizer, cache, tnext)
        if !(subinteg isa Tuple) &&
           subinteg.sol.retcode ∉
           (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
            return
        end
        backward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
    end
end

# Strang-Marchuk Splitting Implementation
"""
    StrangMarchuk <: AbstractOperatorSplittingAlgorithm

A second order symmetric operator splitting algorithm attributed to [Str:1968:ccd,Mar:1971:tsm](@cite).
This implements the classical ABA scheme where for operators A and B:
- First half-step with A (dt/2)
- Full step with B (dt)
- Second half-step with A (dt/2)

For two operators, this gives second-order accuracy in time.

# References
* G. Strang, On the construction and comparison of difference schemes, SIAM Journal on
Numerical Analysis, 5 (1968), pp. 506–517
* G. I. Marchuk, On the theory of the splitting-up method, in Numerical Solution of Partial
Differential Equations-II, Academic Press, 1971, pp. 469 – 500
"""
struct StrangMarchuk{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
end

struct StrangMarchukCache{uType, uprevType, iiType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    inner_caches::iiType
end

function init_cache(f::GenericSplitFunction, alg::StrangMarchuk;
        uprev::AbstractArray, u::AbstractVector,
        inner_caches,
        alias_uprev = true,
        alias_u = false
)
    _uprev = alias_uprev ? uprev : SciMLBase.recursivecopy(uprev)
    _u = alias_u ? u : SciMLBase.recursivecopy(u)
    StrangMarchukCache(_u, _uprev, inner_caches)
end

@inline function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrators::Tuple, solution_indices::Tuple,
        synchronizers::Tuple, cache::StrangMarchukCache, tnext)
    # Strang-Marchuk ABA splitting scheme
    # For two operators A and B: A(dt/2) -> B(dt) -> A(dt/2)
    # This achieves second-order accuracy through symmetry

    @unpack inner_caches = cache
    tcurr = outer_integrator.t
    dt = tnext - tcurr
    thalf = tcurr + dt / 2

    # We require exactly 2 subintegrators for the classical ABA scheme
    if length(subintegrators) != 2
        error("StrangMarchuk splitting requires exactly 2 operators")
    end

    # First operator A at half step
    subinteg_A = subintegrators[1]
    synchronizer_A = synchronizers[1]
    idxs_A = solution_indices[1]
    cache_A = inner_caches[1]

    # Second operator B at full step
    subinteg_B = subintegrators[2]
    synchronizer_B = synchronizers[2]
    idxs_B = solution_indices[2]
    cache_B = inner_caches[2]

    # Step 1: A for dt/2
    @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg_A, idxs_A, synchronizer_A)
    @timeit_debug "time solve A (half)" advance_solution_to!(
        outer_integrator, subinteg_A, idxs_A, synchronizer_A, cache_A, thalf)
    if !(subinteg_A isa Tuple) &&
       subinteg_A.sol.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return
    end
    backward_sync_subintegrator!(outer_integrator, subinteg_A, idxs_A, synchronizer_A)

    # Step 2: B for dt
    @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg_B, idxs_B, synchronizer_B)
    @timeit_debug "time solve B (full)" advance_solution_to!(
        outer_integrator, subinteg_B, idxs_B, synchronizer_B, cache_B, tnext)
    if !(subinteg_B isa Tuple) &&
       subinteg_B.sol.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return
    end
    backward_sync_subintegrator!(outer_integrator, subinteg_B, idxs_B, synchronizer_B)

    # Step 3: A for dt/2
    @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg_A, idxs_A, synchronizer_A)
    @timeit_debug "time solve A (half)" advance_solution_to!(
        outer_integrator, subinteg_A, idxs_A, synchronizer_A, cache_A, tnext)
    if !(subinteg_A isa Tuple) &&
       subinteg_A.sol.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return
    end
    backward_sync_subintegrator!(outer_integrator, subinteg_A, idxs_A, synchronizer_A)
end
