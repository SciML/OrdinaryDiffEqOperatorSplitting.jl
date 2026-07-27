"""
    OperatorSplittingProblem(f::AbstractOperatorSplitFunction, u0, tspan, p::Tuple)

Define an ODE problem whose right-hand side is an operator tree. Use it with
[`LieTrotterGodunov`](@ref) or [`StrangMarchuk`](@ref) through the SciML
`init`/`solve` interface.

# Arguments
- `f`: [`GenericSplitFunction`](@ref) describing the operator tree.
- `u0`: Initial state of the full system.
- `tspan`: Integration interval.
- `p`: Optional tuple of operator parameters. Null parameters are created when
  it is omitted.

# Keywords
Additional keywords are stored with the problem and forwarded by the solver.
"""
mutable struct OperatorSplittingProblem{
        fType <: AbstractOperatorSplitFunction, uType, tType, pType <: Tuple, K,
    } <:
    SciMLBase.AbstractODEProblem{uType, tType, true}
    f::fType
    u0::uType
    tspan::tType
    p::pType
    kwargs::K # TODO what to do with these?
    function OperatorSplittingProblem(
            f::AbstractOperatorSplitFunction,
            u0, tspan, p = recursive_null_parameters(f);
            kwargs...
        )
        return new{
            typeof(f), typeof(u0),
            typeof(tspan), typeof(p),
            typeof(kwargs),
        }(
            f,
            u0,
            tspan,
            p,
            kwargs
        )
    end
end

num_operators(prob::OperatorSplittingProblem) = num_operators(prob.f)

recursive_null_parameters(f::AbstractOperatorSplitFunction) = @error "Not implemented"
function recursive_null_parameters(f::GenericSplitFunction)
    return ntuple(i -> recursive_null_parameters(get_operator(f, i)), length(f.functions))
end
function recursive_null_parameters(f::SciMLBase.AbstractDiffEqFunction)
    return NullParameters()
end
function recursive_null_parameters(f)
    return NullParameters()
end
