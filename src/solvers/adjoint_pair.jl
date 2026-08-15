# ---------------------------------------------------------------------------
# Pairs of a scheme and its adjoint
#
# Running a scheme together with its adjoint from the same initial value gives both a
# solution one order higher and an error estimate, which is what makes these the
# adaptive splitting schemes of the package. See [AuzHofKetKoc:2017:psm](@cite), §3.
# ---------------------------------------------------------------------------

"""
    _pair_average_and_estimate!(parent, uforward, dt)

Combine the two members of a scheme/adjoint pair, `parent.u` and `uforward`, into the
higher-order solution and, when the node is adaptive, the local error estimate.

This is the Milne device of [AuzHofKetKoc:2017:psm](@cite), §3: for a base scheme of
odd order `p` the two members have leading error terms `±C h^{p+1}`, so their average
is of order `p+1` and half their difference estimates the base scheme's local error.
`uforward` is overwritten with the scaled residual.
"""
function _pair_average_and_estimate!(parent, uforward, dt)
    parent.u .= (parent.u .+ uforward) ./ 2
    if parent.controller_cache !== nothing
        (; abstol, reltol, internalnorm) = parent.opts
        @. uforward = (parent.u - uforward) /
            (abstol + max(abs(parent.u), abs(parent.uprev)) * reltol)
        OrdinaryDiffEqCore.set_EEst!(parent, internalnorm(uforward, parent.t + dt))
    end
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
drives the step size controller.

This is [`AdjointPair`](@ref) at `p = 1`, written out directly because it needs no
coefficient table and, unlike the table-driven schemes, works for any number of
operators.

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

Base.show(io::IO, alg::PalindromicPairLieTrotterGodunov) =
    _show_scheme(io, "PPLTG", alg.inner_algs, " <-> ")

order(::PalindromicPairLieTrotterGodunov) = 2

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

    _pair_average_and_estimate!(parent, uforward, dt)
    return
end

# ---------------------------------------------------------------------------
# Adjoint pairs of a table-driven scheme
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

    _pair_average_and_estimate!(parent, uforward, dt)
    return
end
