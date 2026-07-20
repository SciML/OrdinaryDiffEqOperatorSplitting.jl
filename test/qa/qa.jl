using SciMLTesting
using OrdinaryDiffEqOperatorSplitting
using JET
using Test

run_qa(
    OrdinaryDiffEqOperatorSplitting;
    # JET reports 2 genuine errors in src/integrator.jl on the SplitSubIntegrator
    # rollback path: `rollback_children!(::SplitSubIntegrator)` has no matching
    # method and `_rollback_children!` is called (line 514) but never defined.
    # Tracked in https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/issues/87
    # target_defined_modules scopes the report to this package's own modules (the
    # default target_modules=(pkg,) filter hides these via-dependency-driven frames).
    jet_broken = true,
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
                :DEFAULT_VERBOSE, :NAN_CHECK, :None,                                      # DiffEqBase
                :fix_dt_at_bounds!, :handle_tstop!, :increment_accept!,                  # OrdinaryDiffEqCore
                :increment_reject!, :initialize_d_discontinuities, :initialize_saveat,   # OrdinaryDiffEqCore
                :initialize_tstops, :post_newton_controller!, :timedepentdtmin,          # OrdinaryDiffEqCore
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :isdtchangeable,                                                        # OrdinaryDiffEqCore
            ),
        ),
    ),
)
