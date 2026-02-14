# Lie-Trotter-Godunov Splitting Implementation
"""
    LieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

A first order sequential operator splitting algorithm attributed to [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
"""
struct LieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
    # transfer_algs::TransferTupleType # Tuple of transfer algorithms from the master solution into the individual ones
end

"""
    OperatorSplittingMinimalSolution

Minimal solution struct for subintegrators that carries just a retcode field for failure communication.
"""
mutable struct OperatorSplittingMinimalSolution{R}
    retcode::R
end

OperatorSplittingMinimalSolution() = OperatorSplittingMinimalSolution(ReturnCode.Default)

"""
    LieTrotterGodunovCache

Enhanced cache for Lie-Trotter-Godunov splitting that can act as a subintegrator.
Contains fields for adaptive time stepping and nested problem handling.
"""
mutable struct LieTrotterGodunovCache{uType, uprevType, tType, dtType, solType, controllerType, EEstType, iiType, subintTreeType, solidxTreeType, syncTreeType, statsType} <: AbstractOperatorSplittingCache
    # Solution state
    u::uType
    uprev::uprevType
    
    # Time stepping state  
    t::tType
    tprev::tType
    dt::dtType
    dtcache::dtType
    
    # Minimal solution for retcode communication
    sol::solType
    
    # Adaptive stepping fields
    controller::controllerType
    EEst::EEstType
    iter::Int
    stats::statsType
    
    # Inner caches and subintegrator trees
    inner_caches::iiType
    subintegrator_tree::subintTreeType
    solution_index_tree::solidxTreeType
    synchronizer_tree::syncTreeType
end

function init_cache(
        f::GenericSplitFunction, alg::LieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
        inner_caches,
        subintegrator_tree = inner_caches,
        solution_index_tree = ntuple(i -> nothing, length(inner_caches)),
        synchronizer_tree = ntuple(i -> NoExternalSynchronization(), length(inner_caches)),
        t = zero(eltype(u)),
        dt = zero(eltype(u)),
        controller = nothing,
        alias_uprev = true,
        alias_u = false
    )
    _uprev = alias_uprev ? uprev : RecursiveArrayTools.recursivecopy(uprev)
    _u = alias_u ? u : RecursiveArrayTools.recursivecopy(u)
    tType = typeof(t)
    sol = OperatorSplittingMinimalSolution()
    return LieTrotterGodunovCache(
        _u, _uprev,
        t, copy(t), dt, copy(dt),
        sol,
        controller, zero(eltype(u)), 0, IntegratorStats(),
        inner_caches, subintegrator_tree, solution_index_tree, synchronizer_tree
    )
end

@inline @unroll function advance_solution_to!(
        outer_integrator::OperatorSplittingIntegrator,
        subintegrators::Tuple, solution_indices::Tuple,
        synchronizers::Tuple, cache::LieTrotterGodunovCache, tnext
    )
    # We assume that the integrators are already synced
    (; inner_caches, subintegrator_tree, solution_index_tree, synchronizer_tree) = cache
    
    # Update cache's own time state
    cache.tprev = cache.t
    cache.t = tnext
    
    # Reset sol.retcode to default before attempting the step
    cache.sol.retcode = ReturnCode.Default
    
    # For each inner operator
    i = 0
    @unroll for subinteg in subintegrators
        i += 1
        synchronizer = synchronizers[i]
        idxs = solution_indices[i]
        inner_cache = inner_caches[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
        @timeit_debug "time solve" advance_solution_to!(
            outer_integrator, subinteg, idxs, synchronizer, inner_cache, tnext
        )
        
        # Check return code and propagate failure
        if !(subinteg isa Tuple)
            # For single integrator
            if subinteg.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
                cache.sol.retcode = subinteg.sol.retcode
                return
            end
        elseif subinteg isa AbstractOperatorSplittingCache
            # For enhanced cache acting as subintegrator
            if subinteg.sol.retcode ∉ (ReturnCode.Default, ReturnCode.Success)
                cache.sol.retcode = subinteg.sol.retcode
                return
            end
        end
        
        backward_sync_subintegrator!(outer_integrator, subinteg, idxs, synchronizer)
    end
    
    # If we got here, mark success
    cache.sol.retcode = ReturnCode.Success
    cache.iter += 1
end
