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
                # Controller-cache protocol names. Several are `public` only in newer
                # OrdinaryDiffEqCore (e.g. post_newton_controller! from 4.7); keep them
                # all ignored so QA stays valid across the whole [compat] range even
                # though the pinned QA manifest resolves a newer version.
                :setup_controller_cache, :reinit_controller!, :post_newton_controller!,  # OrdinaryDiffEqCore
                :get_EEst, :set_EEst!, :get_current_adaptive_order,                      # OrdinaryDiffEqCore
                :gamma_default, :failfactor_default, :AbstractControllerCache,           # OrdinaryDiffEqCore
                :promote_tspan,                                                          # SciMLBase
                # Shared linear interpolation kernels, reused so that the
                # integrator's own dense output and `sol(t)` (which goes through
                # SciMLBase's `LinearInterpolation`) cannot drift apart.
                :linear_interpolant, :linear_interpolant!,                                # SciMLBase
                # Broadcast extension points a TreeOption implements (src/config_tree.jl).
                :Broadcasted, :broadcastable, :dotview, :materialize!,                   # Base
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :isdtchangeable,                                                        # OrdinaryDiffEqCore
                # Public only in newer OrdinaryDiffEqCore; see the note above.
                :stepsize_controller!, :step_accept_controller!,                        # OrdinaryDiffEqCore
                :step_reject_controller!, :accept_step_controller,                      # OrdinaryDiffEqCore
            ),
        ),
    ),
)
