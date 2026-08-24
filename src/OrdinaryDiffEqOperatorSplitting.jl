module OrdinaryDiffEqOperatorSplitting

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import BinaryHeaps

import SciMLBase, DiffEqBase, SciMLLogging
import SciMLBase: ReturnCode
import SciMLBase: DEIntegrator, NullParameters, isadaptive
import SymbolicIndexingInterface: variable_symbols

import DiffEqBase: set_proposed_dt!

import RecursiveArrayTools

import OrdinaryDiffEqCore: OrdinaryDiffEqCore, isdtchangeable,
    stepsize_controller!, step_accept_controller!, step_reject_controller!,
    accept_step_controller

# `@verbosity_specifier` expands to code referring to SciMLLogging's names (presets,
# `MessageLevel`, `AbstractVerbosityPreset`) unqualified, so they have to be in scope.
import SciMLLogging: @SciMLMessage, @verbosity_specifier, All, Detailed, InfoLevel,
    Minimal, None, Silent, Standard

# In OrdinaryDiffEq v7 / DiffEqBase v7, passing verbose::Bool to inner ODE
# integrators is no longer supported. Convert Bool → DEVerbosity when available.
@static if isdefined(DiffEqBase, :DEVerbosity)
    """
        OperatorSplittingVerbosity

    Verbosity configuration for the splitting nodes of an operator-splitting solve.

    A splitting node has diagnostics of its own -- the step size it chose and the
    splitting error estimate behind that choice -- which are separate from anything the
    inner integrators report. Those are toggled here, while `inner_verbosity` holds the
    `DiffEqBase.DEVerbosity` handed to every inner integrator, mirroring the way
    `DEVerbosity` itself nests `linear_verbosity` and `nonlinear_verbosity`.

    # Toggles

      - `splitting_step_accepted`: `t`, `dt` and `EEst` of each accepted splitting step.
      - `splitting_step_rejected`: the same for a step the controller rejected.
      - `inner_solver_stats`: a per-operator summary of the inner integrator work --
        steps, rejections, `f` evaluations, Jacobians, `W` factorizations, linear solves
        -- emitted once when the solve finishes.

    # Examples

    ```julia
    # A preset applies to this node and travels to the inner integrators.
    verbose = OperatorSplittingVerbosity(SciMLLogging.All())

    # Splitting diagnostics on, inner integrators quiet.
    verbose = OperatorSplittingVerbosity(
        splitting_step_accepted = SciMLLogging.InfoLevel(),
        inner_solver_stats = SciMLLogging.InfoLevel(),
        inner_verbosity = DiffEqBase.DEVerbosity(SciMLLogging.None()),
    )
    ```

    A `DEVerbosity`, a `SciMLLogging` preset or a `Bool` passed as `verbose` is accepted
    and converted. A `DEVerbosity` becomes the `inner_verbosity` with the splitting
    diagnostics left silent, so existing code keeps exactly its previous output.
    """
    @verbosity_specifier OperatorSplittingVerbosity begin
        toggles = (
            :splitting_step_accepted, :splitting_step_rejected, :inner_solver_stats,
        )

        sub_specifiers = (:inner_verbosity,)

        groups = (
            step_control = (:splitting_step_accepted, :splitting_step_rejected),
            performance = (:inner_solver_stats,),
        )

        presets = (
            None = (
                inner_verbosity = None(),
                splitting_step_accepted = Silent(),
                splitting_step_rejected = Silent(),
                inner_solver_stats = Silent(),
            ),
            Minimal = (
                inner_verbosity = Minimal(),
                splitting_step_accepted = Silent(),
                splitting_step_rejected = Silent(),
                inner_solver_stats = Silent(),
            ),
            Standard = (
                inner_verbosity = Standard(),
                splitting_step_accepted = Silent(),
                splitting_step_rejected = Silent(),
                inner_solver_stats = Silent(),
            ),
            Detailed = (
                inner_verbosity = Detailed(),
                splitting_step_accepted = Silent(),
                splitting_step_rejected = InfoLevel(),
                inner_solver_stats = InfoLevel(),
            ),
            All = (
                inner_verbosity = All(),
                splitting_step_accepted = InfoLevel(),
                splitting_step_rejected = InfoLevel(),
                inner_solver_stats = InfoLevel(),
            ),
        )
    end

    const DEFAULT_VERBOSITY = OperatorSplittingVerbosity(SciMLLogging.Minimal())

    _inner_verbose(verbose::Bool) = verbose ?
        DiffEqBase.DEVerbosity(SciMLLogging.Minimal()) :
        DiffEqBase.DEVerbosity(SciMLLogging.None())
    _inner_verbose(verbose::OperatorSplittingVerbosity) = verbose.inner_verbosity

    # A splitting node owns an `OperatorSplittingVerbosity`; anything else a caller
    # passes is the inner integrators' setting and is wrapped, leaving the splitting
    # diagnostics off so that existing code keeps its previous output.
    _process_verbose(verbose::OperatorSplittingVerbosity) = verbose
    _process_verbose(verbose::DiffEqBase.DEVerbosity) = OperatorSplittingVerbosity(;
        preset = SciMLLogging.Minimal(), inner_verbosity = verbose
    )
    _process_verbose(preset::SciMLLogging.AbstractVerbosityPreset) =
        OperatorSplittingVerbosity(preset)
else
    const DEFAULT_VERBOSITY = false
end
_inner_verbose(verbose) = verbose
_process_verbose(verbose) = verbose

# `verbose` reaches us either as a Bool or, through DiffEqBase v7's `init`, as a
# verbosity specifier whose first type parameter is the on/off flag. Neither can be
# used in a boolean context directly.
_is_verbose(verbose::Bool) = verbose
_is_verbose(verbose) = true
_is_verbose(::SciMLLogging.AbstractVerbositySpecifier{B}) where {B} = B

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
