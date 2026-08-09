# ---------------------------------------------------------------------------
# Strang-Marchuk operator splitting
# ---------------------------------------------------------------------------
"""
    StrangMarchuk <: AbstractOperatorSplittingAlgorithm

Second-order symmetric (palindromic) operator splitting algorithm attributed to
[Str:1968:ccd,Mar:1971:tsm](@cite).

For ``N`` operators the scheme performs

``A_1(\\Delta t/2) \\to \\cdots \\to A_{N-1}(\\Delta t/2) \\to A_N(\\Delta t) \\to A_{N-1}(\\Delta t/2) \\to \\cdots \\to A_1(\\Delta t/2)``

achieving second-order accuracy through symmetry.
"""
struct StrangMarchuk{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
end

Base.show(io::IO, alg::StrangMarchuk) = _show_scheme(io, "SM", alg.inner_algs)

order(::StrangMarchuk) = 2

struct StrangMarchukCache{uType, uprevType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
end

function init_cache(
        f::GenericSplitFunction, alg::StrangMarchuk;
        uprev::AbstractArray, u::AbstractVector,
    )
    return StrangMarchukCache(u, uprev)
end

# Forward pass: A₁(dt/2) → … → Aₙ₋₁(dt/2) → Aₙ(dt)
@unroll function _sm_forward_pass!(parent, children::Tuple, half_dt, dt)
    N = length(children)
    i = 0
    @unroll for child in children
        i += 1
        _advance_child!(parent, child, i, i < N ? half_dt : dt)
        parent.force_stepfail && return
    end
end

# Reverse pass: Aₙ₋₁(dt/2) → … → A₁(dt/2)
@unroll function _sm_reverse_pass!(parent, rev_front::Tuple, half_dt, N)
    j = 0
    @unroll for child in rev_front
        j += 1
        _advance_child!(parent, child, N - j, half_dt)
        parent.force_stepfail && return
    end
end

function _perform_step!(
        parent,
        children::Tuple,
        cache::StrangMarchukCache,
        dt
    )
    half_dt = dt / 2

    # Skip sync of for first solve, because it is already in sync
    mark_next_sync_continuous(parent)

    _sm_forward_pass!(parent, children, half_dt, dt)
    parent.force_stepfail && return

    _sm_reverse_pass!(parent, reverse(children[1:(end - 1)]), half_dt, length(children))
    parent.force_stepfail && return

    return
end
