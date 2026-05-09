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

        backward_sync_subintegrator!(parent, child, idxs, sync)
    end
end

# ---------------------------------------------------------------------------
# Strang-Marchuk operator splitting
# ---------------------------------------------------------------------------
"""
    StrangMarchuk <: AbstractOperatorSplittingAlgorithm

Second-order symmetric operator splitting algorithm attributed to
[Str:1968:ccd,Mar:1971:tsm](@cite).

For two operators ``A`` and ``B`` the scheme performs
``A(\\Delta t/2) \\to B(\\Delta t) \\to A(\\Delta t/2)``,
achieving second-order accuracy through symmetry.
"""
struct StrangMarchuk{AlgTupleType} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType
    function StrangMarchuk(inner_algs::T) where {T <: Tuple}
        length(inner_algs) == 2 || throw(
            ArgumentError("StrangMarchuk requires exactly 2 inner algorithms, got $(length(inner_algs))")
        )
        return new{T}(inner_algs)
    end
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

function _perform_step!(
        parent,
        children::Tuple,
        cache::StrangMarchukCache,
        dt
    )
    half_dt = dt / 2

    # A(dt/2)
    let child = children[1]
        idxs = parent.child_solution_indices[1]
        sync = parent.child_synchronizers[1]
        @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(parent, child, half_dt)
        if _child_failed(child)
            parent.force_stepfail = true
            return
        end
        backward_sync_subintegrator!(parent, child, idxs, sync)
    end

    # B(dt)
    let child = children[2]
        idxs = parent.child_solution_indices[2]
        sync = parent.child_synchronizers[2]
        @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(parent, child, dt)
        if _child_failed(child)
            parent.force_stepfail = true
            return
        end
        backward_sync_subintegrator!(parent, child, idxs, sync)
    end

    # If B contaminated the solution (e.g. NaN), skip the second A step
    # and let the outer check_error! detect instability on the next iteration.
    if !all(isfinite, parent.u)
        _force_set_time!(children[1], children[2].t)
        return
    end

    # A(dt/2)
    let child = children[1]
        idxs = parent.child_solution_indices[1]
        sync = parent.child_synchronizers[1]
        @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
        @timeit_debug "time solve" advance_solution_by!(parent, child, half_dt)
        if _child_failed(child)
            parent.force_stepfail = true
            return
        end
        backward_sync_subintegrator!(parent, child, idxs, sync)
    end

    # Snap child 1's accumulated time (two half-steps) to match child 2's
    # (one full step) to prevent floating-point drift.
    try_snap_children_to_tstop!(children[1], children[2].t)
end
