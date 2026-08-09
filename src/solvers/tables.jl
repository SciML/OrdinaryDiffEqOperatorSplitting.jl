# ---------------------------------------------------------------------------
# Schemes defined by a coefficient table
#
# Each of these is a struct, a table, and the two interface methods; the stepping
# itself is the generic table traversal in coefficients.jl.
# ---------------------------------------------------------------------------
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

Base.show(io::IO, alg::Ruth3) = _show_scheme(io, "Ruth3", alg.inner_algs)

const RUTH3_COEFFICIENTS = SplittingCoefficients(
    (7 // 24, 2 // 3), (3 // 4, -2 // 3), (-1 // 24, 1 // 1)
)

coefficients(::Ruth3) = RUTH3_COEFFICIENTS
order(::Ruth3) = 3

function init_cache(
        f::GenericSplitFunction, alg::Ruth3;
        uprev::AbstractArray, u::AbstractVector,
    )
    return SplittingCoefficientsCache(u, uprev, coefficients(alg))
end

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

Base.show(io::IO, alg::Yoshida4) = _show_scheme(io, "Yoshida4", alg.inner_algs)

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
