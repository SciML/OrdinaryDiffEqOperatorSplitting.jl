module OrdinaryDiffEqOperatorSplitting

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import SciMLBase, DiffEqBase, DataStructures
import SciMLBase: ReturnCode
import SciMLBase: DEIntegrator, NullParameters, isadaptive

import RecursiveArrayTools

import OrdinaryDiffEqCore

import UnPack: @unpack

abstract type AbstractOperatorSplitFunction <: SciMLBase.AbstractODEFunction{true} end
abstract type AbstractOperatorSplittingAlgorithm end
abstract type AbstractOperatorSplittingCache end

@inline SciMLBase.isadaptive(::AbstractOperatorSplittingAlgorithm) = false

include("function.jl")
include("problem.jl")
include("integrator.jl")
include("solver.jl")
include("utils.jl")

export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov

end
