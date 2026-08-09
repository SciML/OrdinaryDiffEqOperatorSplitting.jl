# ---------------------------------------------------------------------------
# Coefficient tables
#
# A splitting scheme of order three or above is specified by a table of coefficients
# rather than by hand-written passes. This file holds the table type, the interface
# every table-driven scheme implements (`coefficients` and `order`), and the two
# traversals of a table: the scheme itself and its adjoint.
# ---------------------------------------------------------------------------

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

struct SplittingCoefficientsCache{uType, uprevType, coeffType} <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    coeffs::coeffType
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
