using SciMLTesting
using OrdinaryDiffEqOperatorSplitting
using JET
using Test

run_qa(
    OrdinaryDiffEqOperatorSplitting;
    # target_defined_modules scopes the report to this package's own modules (the
    # default target_modules=(pkg,) filter hides via-dependency-driven frames).
    jet_kwargs = (; target_defined_modules = true, mode = :basic),
    ei_kwargs = (;
        # Names re-exported through the SciML umbrella chain; accessed via a
        # re-exporting dep rather than the owning package.
        all_qualified_accesses_via_owners = (;
            ignore = (
                :None,               # owner SciMLLogging, via DiffEqBase
                :timedepentdtmin,    # owner DiffEqBase, via OrdinaryDiffEqCore
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :__init, :__solve, :done, :postamble!, :solution_new_retcode,           # SciMLBase
                :DEFAULT_VERBOSE, :NAN_CHECK, :None, :ODE_DEFAULT_NORM,                  # DiffEqBase
                :fix_dt_at_bounds!, :handle_tstop!, :increment_accept!,                  # OrdinaryDiffEqCore
                :increment_reject!, :initialize_d_discontinuities, :initialize_saveat,   # OrdinaryDiffEqCore
                :initialize_tstops, :timedepentdtmin, :IController,                      # OrdinaryDiffEqCore
                :failfactor_default,                                                     # OrdinaryDiffEqCore
                :promote_tspan,                                                          # SciMLBase
                # Broadcast extension points a TreeOption implements (src/config_tree.jl).
                :Broadcasted, :broadcastable, :dotview, :materialize!,                   # Base
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :isdtchangeable,                                                        # OrdinaryDiffEqCore
            ),
        ),
    ),
)
