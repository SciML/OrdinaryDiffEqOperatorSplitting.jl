using SciMLTesting
using OrdinaryDiffEqOperatorSplitting
using JET
using Test

run_qa(
    OrdinaryDiffEqOperatorSplitting;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # Broadcast overloading has no public spelling for these two:
                # `Base.Broadcast` marks `dotview`, `broadcastable` and
                # `BroadcastStyle` public but not `materialize!` or the
                # `Broadcasted` type they dispatch on, and neither has an alias.
                # src/config_tree.jl needs both to give `opt[...] .= x` its
                # subtree-fill meaning.
                :Broadcasted, :materialize!,
                # https://github.com/SciML/OrdinaryDiffEq.jl/pull/4111 makes this
                # public alongside the rest of the per-algorithm controller
                # defaults. Drop once a release carrying it is registered.
                :failfactor_default,
            ),
        ),
    ),
)
