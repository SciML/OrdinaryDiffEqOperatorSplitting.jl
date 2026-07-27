module OrdinaryDiffEqOperatorSplitting

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import BinaryHeaps

import SciMLBase, DiffEqBase, SciMLLogging
import SciMLBase: ReturnCode
import SciMLBase: DEIntegrator, NullParameters, isadaptive
import SymbolicIndexingInterface: variable_symbols

import RecursiveArrayTools

import OrdinaryDiffEqCore: OrdinaryDiffEqCore, isdtchangeable,
    stepsize_controller!, step_accept_controller!, step_reject_controller!,
    accept_step_controller

# In OrdinaryDiffEq v7 / DiffEqBase v7, passing verbose::Bool to inner ODE
# integrators is no longer supported. Convert Bool → DEVerbosity when available.
@static if isdefined(DiffEqBase, :DEVerbosity)
    # `DEVerbosity()` is what DiffEqBase itself defaults `verbose` to; spelling it
    # out keeps this off the non-public `DEFAULT_VERBOSE` binding.
    _inner_verbose(verbose::Bool) = verbose ?
        DiffEqBase.DEVerbosity() :
        DiffEqBase.DEVerbosity(SciMLLogging.None())
    const DEFAULT_VERBOSITY = DiffEqBase.DEVerbosity()
else
    const DEFAULT_VERBOSITY = false
end
_inner_verbose(verbose) = verbose

# `verbose` reaches us either as a Bool or, through DiffEqBase v7's `init`, as a
# DEVerbosity whose first type parameter is the on/off flag. Neither can be used in
# a boolean context directly.
_is_verbose(verbose::Bool) = verbose
_is_verbose(verbose) = true
@static if isdefined(DiffEqBase, :DEVerbosity)
    _is_verbose(::DiffEqBase.DEVerbosity{B}) where {B} = B
end

abstract type AbstractOperatorSplitFunction <: SciMLBase.AbstractODEFunction{true} end

"""
    AbstractOperatorSplittingAlgorithm

Abstract supertype for operator-splitting algorithms.

## Extension interface v1

Solver extensions must provide an `inner_algs` tuple matching the associated
[`GenericSplitFunction`](@ref), an [`init_cache`](@ref) method, and an
[`_perform_step!`](@ref) method. This is a developer extension interface, not
an end-user API. It may change in a breaking release of this package.
"""
abstract type AbstractOperatorSplittingAlgorithm end

"""
    AbstractOperatorSplittingCache

Abstract supertype for caches used by operator-splitting algorithms.

## Extension interface v1

Define a concrete cache subtype and return it from [`init_cache`](@ref) when
implementing an [`AbstractOperatorSplittingAlgorithm`](@ref). This is a
developer extension interface, not an end-user API. It may change in a
breaking release of this package.
"""
abstract type AbstractOperatorSplittingCache end

"""
    init_cache(f::GenericSplitFunction, alg::AbstractOperatorSplittingAlgorithm; uprev, u)

Create the cache for an operator-splitting algorithm.

## Extension interface v1

Extensions must dispatch on their concrete algorithm type and return a concrete
[`AbstractOperatorSplittingCache`](@ref). `u` and `uprev` are the mutable local
solution buffers for the current node. This is a developer extension interface,
not an end-user API. It may change in a breaking release of this package.
"""
function init_cache end

"""
    _perform_step!(parent, children::Tuple, cache::AbstractOperatorSplittingCache, dt)

Advance one operator-splitting step for a developer-defined algorithm.

## Extension interface v1

Extensions must synchronize every child before and after advancing it and set
`parent.force_stepfail` when a child fails. This is a developer extension
interface, not an end-user API. It may change in a breaking release of this
package.
"""
function _perform_step! end

@inline SciMLBase.isadaptive(::AbstractOperatorSplittingAlgorithm) = false
@inline isdtchangeable(alg::AbstractOperatorSplittingAlgorithm) = all(isdtchangeable.(alg.inner_algs))

include("function.jl")
include("config_tree.jl")
include("problem.jl")
include("integrator.jl")
include("solver.jl")
include("utils.jl")

export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov, StrangMarchuk,
    PalindromicPairLieTrotterGodunov
export SplitNode, TreeOption

include("precompilation.jl")

end
