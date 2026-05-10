module OrdinaryDiffEqOperatorSplitting

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import SciMLBase, DiffEqBase, DataStructures
import SciMLBase: ReturnCode, DEStats
import SciMLBase: DEIntegrator, NullParameters, isadaptive

import RecursiveArrayTools
import LinearAlgebra

import OrdinaryDiffEqCore: OrdinaryDiffEqCore, isdtchangeable,
    stepsize_controller!, step_accept_controller!, step_reject_controller!,
    DEOptions

# DiffEqBase v7 no longer accepts verbose::Bool for inner ODE integrators; convert to DEVerbosity.
_inner_verbose(verbose::Bool) = verbose ? DiffEqBase.DEFAULT_VERBOSE : DiffEqBase.DEVerbosity(DiffEqBase.None())
_inner_verbose(verbose::DiffEqBase.DEVerbosity) = verbose

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
