module OrdinaryDiffEqOperatorSplitting

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import SciMLBase, DiffEqBase, DataStructures
import SciMLBase: ReturnCode
import SciMLBase: DEIntegrator, NullParameters, isadaptive

import RecursiveArrayTools

import OrdinaryDiffEqCore: OrdinaryDiffEqCore, isdtchangeable,
    stepsize_controller!, step_accept_controller!, step_reject_controller!

# In OrdinaryDiffEq v7 / DiffEqBase v7, passing verbose::Bool to inner ODE
# integrators is no longer supported. Convert Bool → DEVerbosity when available.
@static if isdefined(DiffEqBase, :DEVerbosity)
    _inner_verbose(verbose::Bool) = verbose ? DiffEqBase.DEFAULT_VERBOSE : DiffEqBase.DEVerbosity(DiffEqBase.None())
else
    _inner_verbose(verbose::Bool) = verbose
end

abstract type AbstractOperatorSplitFunction <: SciMLBase.AbstractODEFunction{true} end
abstract type AbstractOperatorSplittingAlgorithm end
abstract type AbstractOperatorSplittingCache end

@inline SciMLBase.isadaptive(::AbstractOperatorSplittingAlgorithm) = false
@inline isdtchangeable(alg::AbstractOperatorSplittingAlgorithm) = all(isdtchangeable.(alg.inner_algs))

include("function.jl")
include("problem.jl")
include("integrator.jl")
include("solver.jl")
include("utils.jl")

export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov

include("precompilation.jl")

end
