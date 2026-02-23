"""
    GenericSplitFunction(functions::Tuple, solution_indices::Tuple)
    GenericSplitFunction(functions::Tuple, solution_indices::Tuple, synchronizers::Tuple)

This type of function describes a set of connected inner functions in mass-matrix form, as usually found in operator splitting procedures.
"""
struct GenericSplitFunction{fSetType <: Tuple, idxSetType <: Tuple, sSetType <: Tuple} <:
    AbstractOperatorSplitFunction
    # Tuple containing the atomic ode functions or further nested split functions.
    functions::fSetType
    # The ranges for the values in the solution vector.
    solution_indices::idxSetType
    # Operators to update the ode function parameters.
    synchronizers::sSetType
    function GenericSplitFunction(fs::Tuple, drs::Tuple, syncers::Tuple)
        @assert length(fs) == length(drs) == length(syncers) "Number of input tuples does not match."
        gsf_recursive_function_type_safety_check.(fs)
        return new{typeof(fs), typeof(drs), typeof(syncers)}(fs, drs, syncers)
    end
end

function gsf_recursive_function_type_safety_check(f::GenericSplitFunction)
    return gsf_recursive_function_type_safety_check.(f.functions)
end

function gsf_recursive_function_type_safety_check(dunno)
    return @warn "One of the inner functions in GenericSplitFunction is of type $(typeof(dunno)) which is not a subtype of SciMLBase.AbstractDiffEqFunction."
end

function gsf_recursive_function_type_safety_check(::SciMLBase.AbstractDiffEqFunction)
    # OK
end

num_operators(f::GenericSplitFunction) = length(f.functions)

"""
    NoExternalSynchronization()

Indicator that no synchronization between parameters and solution vectors is necessary.
"""
struct NoExternalSynchronization end

function GenericSplitFunction(fs::Tuple, drs::Tuple)
    return GenericSplitFunction(fs, drs, ntuple(_ -> NoExternalSynchronization(), length(fs)))
end

@inline get_operator(f::GenericSplitFunction, i::Integer) = f.functions[i]
@inline get_solution_indices(f::GenericSplitFunction, i::Integer) = f.solution_indices[i]
