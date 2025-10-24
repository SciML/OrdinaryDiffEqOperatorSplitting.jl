"""
    OperatorSplittingProblem(f::AbstractOperatorSplitFunction, u0, tspan, p::Tuple)
"""
mutable struct OperatorSplittingProblem{
    fType <: AbstractOperatorSplitFunction, uType, tType, pType <: Tuple, K} <:
               DiffEqBase.AbstractODEProblem{uType, tType, true}
    f::fType
    u0::uType
    tspan::tType
    p::pType
    kwargs::K # TODO what to do with these?
    function OperatorSplittingProblem(f::AbstractOperatorSplitFunction,
            u0, tspan, p = recursive_null_parameters(f);
            kwargs...)
        new{typeof(f), typeof(u0),
            typeof(tspan), typeof(p),
            typeof(kwargs)}(f,
            u0,
            tspan,
            p,
            kwargs)
    end
end

num_operators(prob::OperatorSplittingProblem) = num_operators(prob.f)

recursive_null_parameters(f::AbstractOperatorSplitFunction) = @error "Not implemented"
function recursive_null_parameters(f::GenericSplitFunction)
    ntuple(i->recursive_null_parameters(get_operator(f, i)), length(f.functions))
end
function recursive_null_parameters(f) # Wildcard for leafs
    DiffEqBase.NullParameters()
end
