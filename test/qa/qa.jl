using SciMLTesting
using OrdinaryDiffEqOperatorSplitting

run_qa(
    OrdinaryDiffEqOperatorSplitting;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # Broadcast overloading has no public spelling: `Base.Broadcast` marks
                # `dotview`, `broadcastable` and `BroadcastStyle` public but not
                # `materialize!` or the `Broadcasted` type they dispatch on, and there
                # is no alias for either. src/config_tree.jl needs both to give
                # `opt[...] .= x` its subtree-fill meaning.
                :Broadcasted, :materialize!,
                # Public from OrdinaryDiffEqCore 4.13; the [compat] floor is 4.4 and
                # 4.12 is the newest registered 4.x, so the check still resolves a
                # version without them. Drop once the floor moves past the release.
                :fix_dt_at_bounds!, :handle_tstop!,
                # https://github.com/SciML/OrdinaryDiffEq.jl/pull/4111 makes this
                # public alongside the rest of the per-algorithm controller defaults.
                :failfactor_default,
            ),
        ),
    ),
)
