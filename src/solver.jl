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
            outer_integrator.force_stepfail = true
        end
        outer_integrator.force_stepfail && return
        backward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
    end
end

OrdinaryDiffEqCore.alg_order(alg::LieTrotterGodunov) = 1

# Adaptive Lie-Trotter-Godunov Splitting Implementation
"""
    PalindromicPairLieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

A second order sequential operator splitting algorithm using the midpoint rule.
"""
struct PalindromicPairLieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
    # transfer_algs::TransferTupleType # Tuple of transfer algorithms from the master solution into the individual ones
end

struct PalindromicPairLieTrotterGodunovCache{uType, uprevType, iiType} <: AbstractOperatorSplittingCache
    u::uType
    u0::uType
    u2::uType
    udiff::uType
    uprev::uprevType
    inner_caches::iiType
end

function init_cache(f::GenericSplitFunction, alg::PalindromicPairLieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
        inner_caches,
        alias_uprev = true,
        alias_u = false
)
    @assert length(inner_caches) == 2 "PP-LTG works only for two operators, but $(length(inner_caches)) have been provided."

    _uprev = alias_uprev ? uprev : SciMLBase.recursivecopy(uprev)
    _u = alias_u ? u : SciMLBase.recursivecopy(u)
    PalindromicPairLieTrotterGodunovCache(_u, copy(u), copy(u), copy(u), _uprev, inner_caches)
end

@inline function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrators::Tuple, solution_indices::Tuple,
        synchronizers::Tuple, cache::PalindromicPairLieTrotterGodunovCache, tnext)
    advance_solution_to_palindromic!(
        outer_integrator, subintegrators, reverse(subintegrators),
        solution_indices, synchronizers, cache, tnext,
    )
end

@inline @unroll function advance_solution_to_palindromic!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrators::Tuple, rsubintegrators::Tuple, solution_indices::Tuple,
        synchronizers::Tuple, cache::PalindromicPairLieTrotterGodunovCache, tnext)
    # @unpack u0, u2, udiff, uprev, inner_caches = cache
    @unpack udiff, uprev, inner_caches = cache

    # FIXME
    # u0 .= outer_integrator.u
    u0 = copy(outer_integrator.u)

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
            integrator.force_stepfail = true
        end
        outer_integrator.force_stepfail && return
        backward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
    end

    # Store solution
    # FIXME
    # u2 .= outer_integrator.u
    u2 = copy(outer_integrator.u)

    # Roll back
    outer_integrator.u .= u0

    @unroll for subinteg in rsubintegrators
        synchronizer = synchronizers[i]
        idxs = solution_indices[i]
        cache = inner_caches[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
        @timeit_debug "time solve" advance_solution_to!(
            outer_integrator, subinteg, idxs, synchronizer, cache, tnext)
        if !(subinteg isa Tuple) &&
        subinteg.sol.retcode ∉
        (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
            integrator.force_stepfail = true
            return
        end
        backward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
        i -= 1
    end

    if outer_integrator.opts.adaptive
        # FIXME
        # udiff .= outer_integrator.u - u2
        udiff = outer_integrator.u - u2
        outer_integrator.EEst = outer_integrator.opts.internalnorm(udiff, tnext)
    end

    outer_integrator.u .+= u2
    outer_integrator.u ./= 2
end

OrdinaryDiffEqCore.isadaptive(alg::PalindromicPairLieTrotterGodunov) = true
OrdinaryDiffEqCore.alg_order(alg::PalindromicPairLieTrotterGodunov) = 2

# @inline function stepsize_controller!(integrator::OperatorSplittingIntegrator, alg::PalindromicPairLieTrotterGodunov)
#     return nothing
# end

# @inline function step_accept_controller!(integrator::OperatorSplittingIntegrator, alg::PalindromicPairLieTrotterGodunov, q)
#     integrator.dt = integrator.dtcache
#     return nothing
# end
# @inline function step_reject_controller!(integrator::OperatorSplittingIntegrator, alg::PalindromicPairLieTrotterGodunov, q)
#     return nothing # Do nothing
# end
