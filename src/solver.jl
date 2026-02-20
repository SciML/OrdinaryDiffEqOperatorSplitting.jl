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

function Base.show(io::IO, alg::LieTrotterGodunov)
    print(io, "LTG (")
    for inner_alg in alg.inner_algs[1:end-1]
        Base.show(io, inner_alg)
        print(io, " -> ")
    end
    length(alg.inner_algs) > 0 && Base.show(io, alg.inner_algs[end])
    print(io, ")")
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

@unroll function _perform_step!(
    outer,
    children::Tuple,
    cache::LieTrotterGodunovCache,
    dt
)
    i = 0
    @unroll for child in children
        i += 1
        idxs = outer.child_solution_indices[i]
        sync = outer.child_synchronizers[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(outer, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(outer, child, dt)
        if _child_failed(child)
            outer.force_stepfail = true
            return
        end
        backward_sync_subintegrator!(outer, child, idxs, sync)
    end
end
