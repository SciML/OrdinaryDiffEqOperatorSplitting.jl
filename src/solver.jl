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

function Base.show(io::IO, alg::LieTrotterGodunov)
    print(io, "LTG (")
    for inner_alg in alg.inner_algs[1:(end - 1)]
        Base.show(io, inner_alg)
        print(io, " -> ")
    end
    length(alg.inner_algs) > 0 && Base.show(io, alg.inner_algs[end])
    return print(io, ")")
end

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

        idxs = parent.child_solution_indices[i]
        sync = parent.child_synchronizers[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(parent, child, dt)
        if _child_failed(child)
            parent.force_stepfail = true
            return
        end

        @timeit_debug "sync <-" backward_sync_subintegrator!(parent, child, idxs, sync)
    end
end

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

function Base.show(io::IO, alg::StrangMarchuk)
    print(io, "SM (")
    for inner_alg in alg.inner_algs[1:(end - 1)]
        Base.show(io, inner_alg)
        print(io, " -> ")
    end
    length(alg.inner_algs) > 0 && Base.show(io, alg.inner_algs[end])
    return print(io, ")")
end

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
        step_dt = i < N ? half_dt : dt

        idxs = parent.child_solution_indices[i]
        sync = parent.child_synchronizers[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(parent, child, step_dt)
        if _child_failed(child)
            parent.force_stepfail = true
            return
        end

        @timeit_debug "sync <-" backward_sync_subintegrator!(parent, child, idxs, sync)
    end
end

# Reverse pass: Aₙ₋₁(dt/2) → … → A₁(dt/2)
@unroll function _sm_reverse_pass!(parent, rev_front::Tuple, half_dt, N)
    j = 0
    @unroll for child in rev_front
        j += 1
        i = N - j

        idxs = parent.child_solution_indices[i]
        sync = parent.child_synchronizers[i]

        @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(parent, child, half_dt)
        if _child_failed(child)
            parent.force_stepfail = true
            return
        end

        @timeit_debug "sync <-" backward_sync_subintegrator!(parent, child, idxs, sync)
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
