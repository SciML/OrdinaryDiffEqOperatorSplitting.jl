"""
    GenericSplitFunction(functions::Tuple, solution_indices::Tuple)
    GenericSplitFunction(functions::Tuple, solution_indices::Tuple, syncronizers::Tuple)

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
        @assert length(fs) == length(drs) == length(syncers)
        new{typeof(fs), typeof(drs), typeof(syncers)}(fs, drs, syncers)
    end
end

num_operators(f::GenericSplitFunction) = length(f.functions)

"""
    NoExternalSynchronization()

Indicator that no synchronization between parameters and solution vectors is necessary.
"""
struct NoExternalSynchronization end

function GenericSplitFunction(fs::Tuple, drs::Tuple)
    GenericSplitFunction(fs, drs, ntuple(_->NoExternalSynchronization(), length(fs)))
end

@inline get_operator(f::GenericSplitFunction, i::Integer) = f.functions[i]
@inline get_solution_indices(f::GenericSplitFunction, i::Integer) = f.solution_indices[i]

recursive_null_parameters(f::AbstractOperatorSplitFunction) = @error "Not implemented"
function recursive_null_parameters(f::GenericSplitFunction)
    ntuple(i->recursive_null_parameters(get_operator(f, i)), length(f.functions))
end;
function recursive_null_parameters(f::DiffEqBase.AbstractDiffEqFunction)
    DiffEqBase.NullParameters()
end
