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
        # Names still non-public in the registered releases (SciMLBase 3.27.0,
        # DiffEqBase 7.6.0, OrdinaryDiffEqCore 4.4.0), used to extend/drive the
        # integrator interface. Verified against those releases via Base.ispublic;
        # the SciMLBase make-public round promoted AbstractDiffEqFunction /
        # AbstractODEFunction / AbstractSciMLFunction / AbstractODEAlgorithm /
        # AbstractODEProblem / build_solution / check_error! / isadaptive (now
        # dropped), DiffEqBase 7.6 made ODE_DEFAULT_ISOUTOFDOMAIN public (dropped),
        # and the heap ordering names moved to BinaryHeaps (public, dropped).
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractODEIntegrator, :__init, :__solve, :done, :has_reinit,           # SciMLBase
                :has_stats, :postamble!, :solution_new_retcode, :variable_symbols,       # SciMLBase
                :DEFAULT_VERBOSE, :NAN_CHECK, :None,                                      # DiffEqBase
                :fix_dt_at_bounds!, :handle_tstop!, :increment_accept!,                  # OrdinaryDiffEqCore
                :increment_reject!, :initialize_d_discontinuities, :initialize_saveat,   # OrdinaryDiffEqCore
                :initialize_tstops, :post_newton_controller!, :timedepentdtmin,          # OrdinaryDiffEqCore
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :DEIntegrator,                                                          # SciMLBase
                :isdtchangeable, :step_accept_controller!, :step_reject_controller!,    # OrdinaryDiffEqCore
                :stepsize_controller!,                                                  # OrdinaryDiffEqCore
            ),
        ),
    ),
)
