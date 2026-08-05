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
        if child_failed(child)
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
        if child_failed(child)
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
        if child_failed(child)
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

# ---------------------------------------------------------------------------
# Palindromic pair of Lie-Trotter-Godunov splittings
# ---------------------------------------------------------------------------
"""
    PalindromicPairLieTrotterGodunov <: AbstractOperatorSplittingAlgorithm

Second-order sequential operator splitting algorithm.

One step solves the palindromic pair of [`LieTrotterGodunov`](@ref) sequences

``A_1(\\Delta t) \\to \\cdots \\to A_N(\\Delta t)`` and
``A_N(\\Delta t) \\to \\cdots \\to A_1(\\Delta t)``

from the same initial value. The leading splitting error of a Lie-Trotter sequence
is ``\\frac{\\Delta t^2}{2}\\sum_{i<j} [A_j, A_i]``, and reversing the sequence
flips the sign of every pairwise commutator, so the average of the pair -- which is
taken as the solution -- is second order accurate for any number of operators. Half
the pair difference estimates the local splitting error of a single sequence and
drives the step size controller, making this the only splitting algorithm in this
package that supports adaptive time stepping of the splitting itself.

Both the order statement and the error estimate account for the *splitting* error
only: they presume the inner solvers resolve their sub-problems accurately relative
to it (adaptive inner solvers, or fixed steps well below the splitting step). With
coarse fixed-step inner solvers -- say `Euler()` stepping at the splitting step size
-- the overall method degrades to the inner order and the controller is blind to
that part of the error.
"""
struct PalindromicPairLieTrotterGodunov{AlgTupleType <: Tuple} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType # Tuple of timesteppers for inner problems
end

function Base.show(io::IO, alg::PalindromicPairLieTrotterGodunov)
    print(io, "PPLTG (")
    for inner_alg in alg.inner_algs[1:(end - 1)]
        Base.show(io, inner_alg)
        print(io, " <-> ")
    end
    length(alg.inner_algs) > 0 && Base.show(io, alg.inner_algs[end])
    return print(io, ")")
end

@inline SciMLBase.isadaptive(::PalindromicPairLieTrotterGodunov) = true
# The pair difference estimates the O(dt²) leading error term of a first order
# sequence, so the controller sees a first order error estimator.
alg_adaptive_order(::PalindromicPairLieTrotterGodunov) = 1

struct PalindromicPairLieTrotterGodunovCache{uType, uprevType, uforwardType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    uforward::uforwardType # end state of the A₁ → A₂ sequence; reused as the residual buffer
end

function init_cache(
        f::GenericSplitFunction, alg::PalindromicPairLieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
    )
    return PalindromicPairLieTrotterGodunovCache(u, uprev, similar(u))
end

function _ppltg_advance_child!(parent, child, i, dt)
    idxs = parent.child_solution_indices[i]
    sync = parent.child_synchronizers[i]

    @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
    @timeit_debug "time solve" advance_solution_by!(parent, child, dt)
    if child_failed(child)
        parent.force_stepfail = true
        return
    end

    @timeit_debug "sync <-" backward_sync_subintegrator!(parent, child, idxs, sync)
    return
end

# Forward sequence: A₁(dt) → … → A_N(dt)
@unroll function _ppltg_forward_pass!(parent, children::Tuple, dt)
    i = 0
    @unroll for child in children
        i += 1
        _ppltg_advance_child!(parent, child, i, dt)
        parent.force_stepfail && return
    end
end

# Reverse sequence: A_N(dt) → … → A₁(dt)
@unroll function _ppltg_reverse_pass!(parent, rchildren::Tuple, dt, N)
    j = 0
    @unroll for child in rchildren
        j += 1
        _ppltg_advance_child!(parent, child, N + 1 - j, dt)
        parent.force_stepfail && return
    end
end

function _perform_step!(
        parent,
        children::Tuple,
        cache::PalindromicPairLieTrotterGodunovCache,
        dt
    )
    (; uforward) = cache

    _ppltg_forward_pass!(parent, children, dt)
    parent.force_stepfail && return
    uforward .= parent.u

    # Rewind to the initial state of the step; uprev is untouched while stepping.
    parent.u .= parent.uprev
    rollback_children!(parent)

    _ppltg_reverse_pass!(parent, reverse(children), dt, length(children))
    parent.force_stepfail && return

    # The average of the pair is the second order solution ...
    parent.u .= (parent.u .+ uforward) ./ 2
    if parent.controller_cache !== nothing
        # ... and half the pair difference the local error of a single sequence.
        (; abstol, reltol, internalnorm) = parent.opts
        @. uforward = (parent.u - uforward) /
            (abstol + max(abs(parent.u), abs(parent.uprev)) * reltol)
        OrdinaryDiffEqCore.set_EEst!(parent, internalnorm(uforward, parent.t + dt))
    end
    return
end
