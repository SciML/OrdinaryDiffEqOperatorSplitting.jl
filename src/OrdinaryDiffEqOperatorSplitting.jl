module OrdinaryDiffEqOperatorSplitting

import TimerOutputs: @timeit_debug
timeit_debug_enabled() = false

import Unrolled: @unroll

import SciMLBase, DiffEqBase, DataStructures

import OrdinaryDiffEqCore: OrdinaryDiffEqCore, isadaptive, alg_order

import UnPack: @unpack
import DiffEqBase: init, TimeChoiceIterator

abstract type AbstractOperatorSplitFunction <: DiffEqBase.AbstractODEFunction{true} end
abstract type AbstractOperatorSplittingAlgorithm end
abstract type AbstractOperatorSplittingCache end

include("function.jl")
include("problem.jl")
include("integrator.jl")
include("solver.jl")
include("utils.jl")
include("controller.jl")

export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov, PalindromicPairLieTrotterGodunov

end
