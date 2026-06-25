using SciMLTesting
using OrdinaryDiffEqOperatorSplitting
using JET
using Test

run_qa(
    OrdinaryDiffEqOperatorSplitting;
    explicit_imports = true,
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
                :DEIntegrator,       # owner SciMLBase, via DiffEqBase
                :None,               # owner SciMLLogging, via DiffEqBase
                :timedepentdtmin,    # owner DiffEqBase, via OrdinaryDiffEqCore
                :variable_symbols,   # owner SymbolicIndexingInterface, via SciMLBase
            ),
        ),
        # Non-public names of upstream deps (SciMLBase / DiffEqBase /
        # OrdinaryDiffEqCore / DataStructures) used to extend/drive the integrator
        # interface. These become public as the base libraries declare them.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiffEqFunction, :AbstractODEAlgorithm, :AbstractODEFunction,   # SciMLBase
                :AbstractODEIntegrator, :AbstractODEProblem, :AbstractSciMLFunction,    # SciMLBase
                :__init, :__solve, :build_solution, :check_error!, :done, :has_reinit,  # SciMLBase
                :has_stats, :isadaptive, :postamble!, :solution_new_retcode,            # SciMLBase
                :variable_symbols,                                                      # SciMLBase
                :DEFAULT_VERBOSE, :DEIntegrator, :NAN_CHECK, :None,                     # DiffEqBase
                :ODE_DEFAULT_ISOUTOFDOMAIN,                                             # DiffEqBase
                :fix_dt_at_bounds!, :handle_tstop!, :increment_accept!,                 # OrdinaryDiffEqCore
                :increment_reject!, :initialize_d_discontinuities, :initialize_saveat,  # OrdinaryDiffEqCore
                :initialize_tstops, :post_newton_controller!, :timedepentdtmin,         # OrdinaryDiffEqCore
                :FasterForward, :FasterReverse,                                         # DataStructures
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :DEIntegrator, :NullParameters, :isadaptive,                            # SciMLBase
                :isdtchangeable, :step_accept_controller!, :step_reject_controller!,    # OrdinaryDiffEqCore
                :stepsize_controller!,                                                  # OrdinaryDiffEqCore
            ),
        ),
    ),
)
