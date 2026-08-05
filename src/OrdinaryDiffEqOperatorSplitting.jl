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
    _inner_verbose(verbose::Bool) = verbose ?
        DiffEqBase.DEVerbosity(SciMLLogging.Minimal()) :
        DiffEqBase.DEVerbosity(SciMLLogging.None())
    const DEFAULT_VERBOSITY = DiffEqBase.DEVerbosity(SciMLLogging.Minimal())
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

"""
    AbstractOperatorSplitFunction

Abstract supertype for functions that define an operator-splitting problem.
Concrete subtypes must provide an operator tree and the local state indices used by
[`OperatorSplittingProblem`](@ref). End users normally construct a
[`GenericSplitFunction`](@ref) rather than implementing this interface directly.
"""
abstract type AbstractOperatorSplitFunction <: SciMLBase.AbstractODEFunction{true} end

"""
    AbstractOperatorSplittingAlgorithm

Developer-only abstract supertype for algorithms that advance an
[`OperatorSplittingProblem`](@ref).

# Interface requirements
- Store an `inner_algs` tuple whose shape mirrors the associated
  [`GenericSplitFunction`](@ref).
- Implement [`init_cache`](@ref) to construct a cache for every node.
- Implement [`_perform_step!`](@ref) to advance that node.

This interface is versioned for OrdinaryDiffEq solver developers. It is not a
supported end-user API.
"""
abstract type AbstractOperatorSplittingAlgorithm end

"""
    AbstractOperatorSplittingCache

Developer-only abstract supertype for an algorithm's per-node cache. A concrete
subtype holds references to the node's `u` and `uprev` buffers and any additional
temporary storage required by the splitting scheme. Construct it from
[`init_cache`](@ref).

This interface is versioned for OrdinaryDiffEq solver developers. It is not a
supported end-user API.
"""
abstract type AbstractOperatorSplittingCache end

"""
    init_cache(f::GenericSplitFunction, alg::AbstractOperatorSplittingAlgorithm; uprev, u)

Construct the per-node cache for a developer-defined operator-splitting algorithm.

# Arguments
- `f`: Operator tree at the node being initialized.
- `alg`: Algorithm used at that node.

# Keyword Arguments
- `uprev`: Mutable state buffer for the preceding accepted state.
- `u`: Mutable state buffer for the state currently being advanced.

# Returns
A concrete [`AbstractOperatorSplittingCache`](@ref). The cache must retain the
provided buffers by reference; the integrator owns their allocation and restores them
after rejected steps.

This is a developer extension API, not a supported end-user API.
"""
function init_cache end

"""
    _perform_step!(parent, children::Tuple, cache::AbstractOperatorSplittingCache, dt)

Advance one node of an operator-splitting tree by `dt`.

# Arguments
- `parent`: Integrator node that owns the current solution and rollback buffers.
- `children`: Direct child integrators, either `DEIntegrator`s or nested splitting
  nodes.
- `cache`: Cache returned by [`init_cache`](@ref) for `parent`'s algorithm.
- `dt`: Signed duration of the splitting step.

# Interface requirements
- Synchronize a child before and after advancing it.
- Leave `parent.uprev` unchanged; rejection restores the node from that buffer.
- Set `parent.force_stepfail = true` and return immediately when a child fails.
- For adaptive algorithms, pass the tolerance-scaled local error estimate to
  `OrdinaryDiffEqCore.set_EEst!` when `parent.controller_cache !== nothing`.

This is a developer extension API, not a supported end-user API.
"""
function _perform_step! end

"""
    child_failed(child)

Return whether a direct child integrator has failed while a developer-defined
splitting step is executing.

# Arguments
- `child`: A leaf `DEIntegrator` or nested operator-splitting integrator.

# Returns
`true` when the child cannot be used for the remainder of the current splitting
step. In that case, `_perform_step!` must set `parent.force_stepfail = true` and
return without synchronizing the failed state back to its parent.

This is a developer extension API, not a supported end-user API.
"""
function child_failed end

@inline SciMLBase.isadaptive(::AbstractOperatorSplittingAlgorithm) = false
@inline SciMLBase.isdiscrete(::AbstractOperatorSplittingAlgorithm) = false
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
