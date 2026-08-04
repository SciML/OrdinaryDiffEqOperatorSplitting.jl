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

function _advance_child!(parent, child, i, dt)
    idxs = parent.child_solution_indices[i]
    sync = parent.child_synchronizers[i]

    @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
    @timeit_debug "time solve" advance_solution_by!(parent, child, dt)
    if _child_failed(child)
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
        _advance_child!(parent, child, i, dt)
        parent.force_stepfail && return
    end
end

# Reverse sequence: A_N(dt) → … → A₁(dt)
@unroll function _ppltg_reverse_pass!(parent, rchildren::Tuple, dt, N)
    j = 0
    @unroll for child in rchildren
        j += 1
        _advance_child!(parent, child, N + 1 - j, dt)
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
        parent.EEst = internalnorm(uforward, parent.t + dt)
    end
    return
end

# ---------------------------------------------------------------------------
# Coefficient-table splitting schemes
# ---------------------------------------------------------------------------
function _require_two_operators(scheme, inner_algs)
    n = length(inner_algs)
    n == 2 || throw(
        ArgumentError(
            "$scheme is a two-operator (AB) table but got $n operators. Group the \
             operators into a nested GenericSplitFunction to use it with more."
        )
    )
    return nothing
end

"""
    SplittingCoefficients(stages::NTuple{N, T}...)

Coefficients of an `S`-stage splitting scheme over `N` operators, one tuple per stage:
stage `j` advances operator `i` by `stages[j][i] * dt`.

This is the generalization to `N` operators of the two-operator (`AB`) and
three-operator (`ABC`) coefficient tables of
[AuzHofKetKoc:2017:psm](@cite); their tables are the `N = 2` and `N = 3` cases.

Each operator's coefficients must sum to one, the consistency condition, and that is
checked here. The remaining order conditions are not, so a table that constructs
successfully can still fail to attain the order it claims.
"""
struct SplittingCoefficients{S, N, T}
    a::NTuple{S, NTuple{N, T}}

    # Do not simplify this to `NTuple{S, NTuple{N, T}}`: the empty tuple matches it for
    # any element type, leaving parameters unbound for `S == 0` (and `T` unbound for
    # `N == 0`). A leading element plus a counted `Vararg` rules both out.
    function SplittingCoefficients(
            stage1::Tuple{T, Vararg{T, K}},
            rest::Tuple{T, Vararg{T, K}}...
        ) where {T, K}
        a = (stage1, rest...)
        N = K + 1
        S = length(a)
        for i in 1:N
            total = sum(a[j][i] for j in 1:S)
            total ≈ one(T) || throw(
                ArgumentError(
                    "operator $i's coefficients sum to $total rather than 1, so the \
                     scheme is not consistent."
                )
            )
        end
        return new{S, N, T}(a)
    end
end

"""
    coefficients(alg)

The [`SplittingCoefficients`](@ref) table of a coefficient-driven splitting algorithm.
"""
function coefficients end

"""
    order(alg)

Order of consistency of a splitting algorithm, counting the splitting error only.
"""
function order end

"""
    Ruth3 <: AbstractOperatorSplittingAlgorithm

Third-order splitting scheme of [Rut:1983:cim](@cite), in three stages.

Its coefficients are exactly rational, and -- as is unavoidable for any real
splitting scheme of order three or above -- some of them are negative, so parts of
the step run backward in time.

As for every splitting scheme here the order statement covers the *splitting* error
only, and presumes the inner solvers resolve their subproblems accurately relative
to it.
"""
struct Ruth3{AlgTupleType <: Tuple} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType

    function Ruth3(inner_algs::Tuple)
        _require_two_operators("Ruth3", inner_algs)
        return new{typeof(inner_algs)}(inner_algs)
    end
end

function Base.show(io::IO, alg::Ruth3)
    print(io, "Ruth3 (")
    for inner_alg in alg.inner_algs[1:(end - 1)]
        Base.show(io, inner_alg)
        print(io, " -> ")
    end
    length(alg.inner_algs) > 0 && Base.show(io, alg.inner_algs[end])
    return print(io, ")")
end

const RUTH3_COEFFICIENTS = SplittingCoefficients(
    (7 // 24, 2 // 3), (3 // 4, -2 // 3), (-1 // 24, 1 // 1)
)

coefficients(::Ruth3) = RUTH3_COEFFICIENTS
order(::Ruth3) = 3

"""
    Yoshida4 <: AbstractOperatorSplittingAlgorithm

Fourth-order splitting scheme of [Yos:1990:cho](@cite), the "triple jump".

Built by composing three Strang steps of lengths ``w_1 h``, ``w_0 h`` and ``w_1 h``
with ``w_1 = 1/(2 - 2^{1/3})`` and ``w_0 = -2^{1/3} w_1``, then merging the adjacent
flows the composition leaves next to each other. That merging is what makes it eight
flow evaluations rather than nine, and it leaves the last stage's second coefficient
zero.

``w_0`` is negative, so a substantial part of each step runs backward in time -- the
second operator's cumulative time reaches ``1.35\\,h`` before returning through
``-0.35\\,h``.

As for every splitting scheme here the order statement covers the *splitting* error
only, and presumes the inner solvers resolve their subproblems accurately relative
to it.
"""
struct Yoshida4{AlgTupleType <: Tuple} <: AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType

    function Yoshida4(inner_algs::Tuple)
        _require_two_operators("Yoshida4", inner_algs)
        return new{typeof(inner_algs)}(inner_algs)
    end
end

function Base.show(io::IO, alg::Yoshida4)
    print(io, "Yoshida4 (")
    for inner_alg in alg.inner_algs[1:(end - 1)]
        Base.show(io, inner_alg)
        print(io, " -> ")
    end
    length(alg.inner_algs) > 0 && Base.show(io, alg.inner_algs[end])
    return print(io, ")")
end

const YOSHIDA4_W1 = 1 / (2 - cbrt(2))
const YOSHIDA4_W0 = -cbrt(2) * YOSHIDA4_W1

const YOSHIDA4_COEFFICIENTS = SplittingCoefficients(
    (YOSHIDA4_W1 / 2, YOSHIDA4_W1),
    ((YOSHIDA4_W1 + YOSHIDA4_W0) / 2, YOSHIDA4_W0),
    ((YOSHIDA4_W1 + YOSHIDA4_W0) / 2, YOSHIDA4_W1),
    (YOSHIDA4_W1 / 2, 0.0),
)

coefficients(::Yoshida4) = YOSHIDA4_COEFFICIENTS
order(::Yoshida4) = 4

function init_cache(
        f::GenericSplitFunction, alg::Yoshida4;
        uprev::AbstractArray, u::AbstractVector,
    )
    return SplittingCoefficientsCache(u, uprev, coefficients(alg))
end

order(::StrangMarchuk) = 2
order(::PalindromicPairLieTrotterGodunov) = 2

# Lie-Trotter keeps its hand-written step; the table exists only so it can serve as an
# `AdjointPair` base, which is what makes `AdjointPair(LieTrotterGodunov(...))` and
# `PalindromicPairLieTrotterGodunov` the same scheme.
order(::LieTrotterGodunov) = 1
coefficients(alg::LieTrotterGodunov) =
    SplittingCoefficients(ntuple(_ -> 1 // 1, length(alg.inner_algs)))

struct SplittingCoefficientsCache{uType, uprevType, coeffType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    coeffs::coeffType
end

function init_cache(
        f::GenericSplitFunction, alg::Ruth3;
        uprev::AbstractArray, u::AbstractVector,
    )
    return SplittingCoefficientsCache(u, uprev, coefficients(alg))
end

function _perform_step!(
        parent,
        children::Tuple,
        cache::SplittingCoefficientsCache,
        dt
    )
    # Deliberately no `mark_next_sync_continuous`: that shortcut needs the previous step
    # to have left `parent.u` equal to the buffer of the child solved first, which holds
    # for StrangMarchuk only because its reverse pass ends on operator 1. A general table
    # ends on operator N, so operator 1's buffer is stale and skipping its forward sync
    # resumes from stale state -- the first step stays exact and every later one is wrong.
    _table_stages!(parent, children, cache.coeffs.a, dt)
    return
end

@unroll function _table_stages!(parent, children, stages::Tuple, dt)
    @unroll for stage in stages
        _table_stage!(parent, children, stage, dt)
        parent.force_stepfail && return
    end
end

@unroll function _table_stage!(parent, children::Tuple, stage, dt)
    i = 0
    @unroll for child in children
        i += 1
        coefficient = stage[i]
        # A zero coefficient is the identity flow, so it is skipped sync and all.
        if !iszero(coefficient)
            _advance_child!(parent, child, i, coefficient * dt)
            parent.force_stepfail && return
        end
    end
end

# ---------------------------------------------------------------------------
# Adjoint pairs
# ---------------------------------------------------------------------------
"""
    AdjointPair(base) <: AbstractOperatorSplittingAlgorithm

Adaptive splitting scheme of order `p+1` built from a `base` scheme of odd order `p`,
following [AuzHofKetKoc:2017:psm](@cite), eq. (3.2).

One step runs `base` and its adjoint ``\\mathcal{S}^*`` from the same initial value.
Their leading error terms are ``C h^{p+1}`` and ``(-1)^p C h^{p+1}``, so for odd `p`
the signs oppose: the average is a solution of order `p+1`, and half the difference is
an asymptotically correct estimate of the base scheme's local error, which drives the
step size controller. A step therefore costs twice the base scheme.

The base scheme's order must be **odd**. For even `p` the two leading terms are
*equal* rather than opposite, so averaging cancels nothing and the difference stops
being an error estimate.

``\\mathcal{S}^*(h, u) = \\mathcal{S}^{-1}(-h, u)`` is the base scheme's entire flat
sequence of flows reversed, every coefficient keeping its sign and its operator, so it
reuses the same table and needs no extra coefficients.

[`PalindromicPairLieTrotterGodunov`](@ref) is this construction at `p = 1`.

As everywhere here, the order and the estimate cover the *splitting* error only and
presume the inner solvers resolve their subproblems accurately relative to it.
"""
struct AdjointPair{BaseType, AlgTupleType <: Tuple} <: AbstractOperatorSplittingAlgorithm
    base::BaseType
    inner_algs::AlgTupleType # aliases `base.inner_algs`, so the tree machinery works unchanged

    function AdjointPair(base::AbstractOperatorSplittingAlgorithm)
        p = order(base)
        isodd(p) || throw(
            ArgumentError(
                "AdjointPair needs a base scheme of odd order, got order $p. For even \
                 orders a scheme and its adjoint share the same leading error term, so \
                 averaging them raises no order and their difference is not an error \
                 estimate."
            )
        )
        return new{typeof(base), typeof(base.inner_algs)}(base, base.inner_algs)
    end
end

function Base.show(io::IO, alg::AdjointPair)
    print(io, "AdjointPair (")
    Base.show(io, alg.base)
    return print(io, ")")
end

coefficients(alg::AdjointPair) = coefficients(alg.base)
order(alg::AdjointPair) = order(alg.base) + 1

@inline SciMLBase.isadaptive(::AdjointPair) = true
# The estimate measures the *base* scheme's leading error, not the pair's.
alg_adaptive_order(alg::AdjointPair) = order(alg.base)

struct AdjointPairCache{uType, uprevType, uforwardType, coeffType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    uforward::uforwardType # end state of the base sequence; reused as the residual buffer
    coeffs::coeffType
end

function init_cache(
        f::GenericSplitFunction, alg::AdjointPair;
        uprev::AbstractArray, u::AbstractVector,
    )
    return AdjointPairCache(u, uprev, similar(u), coefficients(alg))
end

function _perform_step!(
        parent,
        children::Tuple,
        cache::AdjointPairCache,
        dt
    )
    (; uforward, coeffs) = cache

    _table_stages!(parent, children, coeffs.a, dt)
    parent.force_stepfail && return
    uforward .= parent.u

    # Rewind to the initial state of the step; uprev is untouched while stepping.
    parent.u .= parent.uprev
    rollback_children!(parent)

    _table_stages_adjoint!(
        parent, reverse(children), reverse(coeffs.a), dt, length(children)
    )
    parent.force_stepfail && return

    # The average of the pair is the order p+1 solution ...
    parent.u .= (parent.u .+ uforward) ./ 2
    if parent.controller_cache !== nothing
        # ... and half the pair difference the local error of the base scheme.
        (; abstol, reltol, internalnorm) = parent.opts
        @. uforward = (parent.u - uforward) /
            (abstol + max(abs(parent.u), abs(parent.uprev)) * reltol)
        parent.EEst = internalnorm(uforward, parent.t + dt)
    end
    return
end

# The adjoint reverses the whole flat sequence of flows, so both the stage order and
# the operator order within each stage are reversed.
@unroll function _table_stages_adjoint!(parent, rchildren, rstages::Tuple, dt, N)
    @unroll for stage in rstages
        _table_stage_adjoint!(parent, rchildren, stage, dt, N)
        parent.force_stepfail && return
    end
end

@unroll function _table_stage_adjoint!(parent, rchildren::Tuple, stage, dt, N)
    j = 0
    @unroll for child in rchildren
        j += 1
        i = N + 1 - j
        coefficient = stage[i]
        if !iszero(coefficient)
            _advance_child!(parent, child, i, coefficient * dt)
            parent.force_stepfail && return
        end
    end
end
