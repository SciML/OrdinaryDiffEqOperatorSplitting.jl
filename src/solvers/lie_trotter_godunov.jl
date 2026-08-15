# ---------------------------------------------------------------------------
# Lie-Trotter-Godunov operator splitting
# ---------------------------------------------------------------------------
"""
    LieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

First-order sequential operator splitting algorithm attributed to
[Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
"""
struct LieTrotterGodunov{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
end

Base.show(io::IO, alg::LieTrotterGodunov) = _show_scheme(io, "LTG", alg.inner_algs)

struct LieTrotterGodunovCache{uType, uprevType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
end

function init_cache(
        f::GenericSplitFunction, alg::LieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
    )
    return LieTrotterGodunovCache(u, uprev)
end

@unroll function _perform_step!(
        parent,
        children::Tuple,
        cache::LieTrotterGodunovCache,
        dt
    )
    i = 0
    @unroll for child in children
        i += 1
        _advance_child!(parent, child, i, dt)
        parent.force_stepfail && return
    end
end

# Lie-Trotter keeps its hand-written step; the table exists only so it can serve as an
# `AdjointPair` base, which is what makes `AdjointPair(LieTrotterGodunov(...))` and
# `PalindromicPairLieTrotterGodunov` the same scheme.
order(::LieTrotterGodunov) = 1
coefficients(alg::LieTrotterGodunov) =
    SplittingCoefficients(ntuple(_ -> 1 // 1, length(alg.inner_algs)))
