using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
import SciMLBase: ReturnCode
import OrdinaryDiffEqCore
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

# Non-commuting pair of linear operators, [A, B] ≠ 0, so the splitting error is
# nonzero and the palindromic pair's error estimator has something real to measure.
A = [-1.0 0.0; 0.0 -2.0]
B = [0.0 0.5; 0.5 0.0]
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)
fA = ODEFunction(odeA)
fB = ODEFunction(odeB)

dofs = [1, 2]
u0 = [1.0, 1.0]
tspan = (0.0, 1.0)
fsplit = GenericSplitFunction((fA, fB), (dofs, dofs))
prob = OperatorSplittingProblem(fsplit, u0, tspan)
trueu = exp(tspan[2] * (A + B)) * u0

PPLTG = PalindromicPairLieTrotterGodunov

@testset "PPLTG adaptivity" begin
    @testset "Error estimate of a single step" begin
        # With Euler sub-solvers taking a single step per pass the whole pair is
        # reproducible by hand: forward sequence uA → uAB, backward uB → uBA, the
        # solution is the pair average, and EEst the tolerance-scaled half difference.
        dt = 0.05
        abstol, reltol = 1.0e-6, 1.0e-3
        integ = DiffEqBase.init(
            prob, PPLTG((Euler(), Euler())); dt, abstol, reltol
        )

        uA = u0 .+ dt .* (A * u0)
        uAB = uA .+ dt .* (B * uA)
        uB = u0 .+ dt .* (B * u0)
        uBA = uB .+ dt .* (A * uB)
        u_expected = (uAB .+ uBA) ./ 2
        resid = (uBA .- uAB) ./ 2 ./
            (abstol .+ max.(abs.(u_expected), abs.(u0)) .* reltol)
        EEst_expected = sqrt(sum(abs2, resid) / length(resid)) # ODE_DEFAULT_NORM

        DiffEqBase.step!(integ)
        @test integ.u ≈ u_expected
        @test integ.EEst ≈ EEst_expected
        @test integ.t ≈ dt

        # The accepted step drives the I controller with exponent
        # 1/(alg_adaptive_order + 1) = 1/2 and gamma = 9/10; the first accepted step
        # may grow by up to qmax_first_step = 10^4. The rtol absorbs the fastpower
        # approximation upstream while still discriminating a wrong exponent (~10%).
        q_expected = clamp(sqrt(EEst_expected) / (9 / 10), 1 / 10^4, 5)
        @test integ.dt ≈ dt / q_expected rtol = 1.0e-3
        @test integ.dtcache ≈ dt / q_expected rtol = 1.0e-3
    end

    @testset "Error estimate scales with dt²" begin
        # For single Euler passes over linear operators the pair difference is
        # exactly dt²[A,B]u₀ and u₀ dominates the residual scaling, so halving dt
        # divides EEst by exactly four.
        EEsts = map((0.02, 0.01)) do dt
            integ = DiffEqBase.init(prob, PPLTG((Euler(), Euler())); dt)
            DiffEqBase.step!(integ)
            integ.EEst
        end
        @test EEsts[1] / EEsts[2] ≈ 4 rtol = 1.0e-6
    end

    @testset "Controller expands the step size under loose tolerances" begin
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 1.0e-3, reltol = 1.0e-2, abstol = 1.0e-4
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        # A fixed-step run would need 1000 steps; the controller settles near the
        # equilibrium step size within a handful of them.
        @test integ.stats.naccept < 100
        @test integ.dtcache > 1.0e-2
        @test maximum(abs, integ.u .- trueu) < 0.05
    end

    @testset "Commuting operators: steps grow to the tolerance-free maximum" begin
        # With commuting operators the splitting is exact, EEst ≈ 0, and the
        # controller must grow dt by qmax every step until the tstop truncates it.
        odeC1(du, u, p, t) = (du .= -u; nothing)
        odeC2(du, u, p, t) = (du .= -0.5 .* u; nothing)
        fsplit_c = GenericSplitFunction(
            (ODEFunction(odeC1), ODEFunction(odeC2)), (dofs, dofs)
        )
        prob_c = OperatorSplittingProblem(fsplit_c, u0, tspan)
        integ = DiffEqBase.init(prob_c, PPLTG((Tsit5(), Tsit5())); dt = 1.0e-3)
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.naccept ≤ 6
        @test maximum(abs, integ.u .- exp(-1.5) .* u0) < 1.0e-3
    end

    @testset "Rejected steps roll back cleanly" begin
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 0.5, reltol = 1.0e-6, abstol = 1.0e-8
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.nreject ≥ 1
        @test maximum(abs, integ.u .- trueu) < 1.0e-3
    end

    @testset "Rejections with time-dependent operators" begin
        # If a rejected step failed to rewind the child integrator clocks, the retry
        # would integrate the wrong time interval, which a time-dependent operator
        # turns into a visible error against the reference solve.
        odeAt(du, u, p, t) = (c = 1 + sin(2π * t); du[1] = -c * u[1]; du[2] = -2c * u[2]; nothing)
        fsplit_t = GenericSplitFunction((ODEFunction(odeAt), fB), (dofs, dofs))
        prob_t = OperatorSplittingProblem(fsplit_t, u0, tspan)

        ref = DiffEqBase.init(
            prob_t, PPLTG((Tsit5(), Tsit5())); dt = 1.0e-3, adaptive = false
        )
        DiffEqBase.solve!(ref)

        integ = DiffEqBase.init(
            prob_t, PPLTG((Tsit5(), Tsit5()));
            dt = 0.4, reltol = 1.0e-5, abstol = 1.0e-8
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.nreject ≥ 1
        @test maximum(abs, integ.u .- ref.u) < 1.0e-3
    end

    @testset "Tighter tolerances give more steps and smaller errors" begin
        results = map((1.0e-2, 1.0e-4, 1.0e-6)) do reltol
            integ = DiffEqBase.init(
                prob, PPLTG((Tsit5(), Tsit5()));
                dt = 0.1, reltol, abstol = reltol * 1.0e-2
            )
            DiffEqBase.solve!(integ)
            @test integ.sol.retcode == ReturnCode.Success
            (err = maximum(abs, integ.u .- trueu), naccept = integ.stats.naccept)
        end
        @test issorted(collect(r.naccept for r in results))
        @test issorted(collect(r.err for r in results); rev = true)
    end

    @testset "Loose inner tolerances choke the estimator (documented)" begin
        # docs/src/topics/adaptivity.md: the pair difference contains the inner
        # solver error, which for sub-problems with fast internal dynamics tracks
        # the *inner tolerance* rather than the splitting dt. Inner tolerances
        # looser than the splitting tolerances then put a dt-independent floor
        # into EEst and the controller shrinks dt until DtLessThanMin. The fast
        # dynamics matter: on a slow smooth problem a high-order leaf undershoots
        # a loose tolerance by orders of magnitude and no floor appears.
        odeAslow(du, u, p, t) = (du[1] = -u[1]; du[2] = -1.01 * u[2]; nothing)
        odeBfast(du, u, p, t) = (du[1] = 1000.0 * u[2]; du[2] = -1000.0 * u[1]; nothing)
        f_fast = GenericSplitFunction(
            (ODEFunction(odeAslow), ODEFunction(odeBfast)), (dofs, dofs)
        )
        prob_fast = OperatorSplittingProblem(f_fast, copy(u0), tspan)
        trueu_fast = exp([-1.0 1000.0; -1000.0 -1.01]) * u0

        function solve_with_leaf_tols(atol_leaf, rtol_leaf; kwargs...)
            abstol = TreeOption(f_fast, 1.0e-6)  # splitting node target
            reltol = TreeOption(f_fast, 1.0e-4)
            abstol[1] = atol_leaf
            abstol[2] = atol_leaf
            reltol[1] = rtol_leaf
            reltol[2] = rtol_leaf
            dtmin = TreeOption(f_fast, 0.0)
            dtmin[] = 1.0e-3                     # splitting node only!
            integ = DiffEqBase.init(
                prob_fast, PPLTG((Tsit5(), Tsit5()));
                dt = 0.1, abstol, reltol, dtmin, verbose = false, kwargs...
            )
            DiffEqBase.solve!(integ)
            return integ
        end

        # Leaves 100x looser than the splitting tolerances: EEst floor, abort.
        choked = solve_with_leaf_tols(1.0e-4, 1.0e-2)
        @test choked.sol.retcode == ReturnCode.DtLessThanMin

        # Same splitting tolerances and dtmin, leaves 10^4x tighter: fine.
        # (Global error ~ splitting reltol accumulated over the ~60 steps.)
        healthy = solve_with_leaf_tols(1.0e-10, 1.0e-8)
        @test healthy.sol.retcode == ReturnCode.Success
        @test maximum(abs, healthy.u .- trueu_fast) < 5.0e-3
    end

    @testset "Unreachable tolerances abort with DtLessThanMin" begin
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 0.1, reltol = 1.0e-12, abstol = 1.0e-14, dtmin = 0.01,
            verbose = false
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.DtLessThanMin
    end

    @testset "Non-adaptive PPLTG runs fixed steps" begin
        integ = DiffEqBase.init(
            prob, PPLTG((Euler(), Euler())); dt = 0.05, adaptive = false
        )
        @test integ.controller_cache === nothing
        @test isnan(integ.EEst)
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.iter == 20
        @test integ.dtcache ≈ 0.05
        @test isnan(integ.EEst)
    end

    @testset "reject_step! restores the whole subtree, including leaf u" begin
        # StrangMarchuk skips the forward sync of its first child on a retry
        # (`next_sync_is_continuous`), so the rollback itself has to restore leaf
        # states -- rewinding only the clocks would silently resume the retry from
        # the failed attempt's state.
        integ = DiffEqBase.init(prob, StrangMarchuk((Euler(), Euler())); dt = 0.05)
        DiffEqBase.step!(integ)
        for child in integ.child_subintegrators
            child.u .+= 1.0 # pollute, as if a failed attempt had advanced the child
            child.t += 0.5
        end
        OS.reject_step!(integ)
        @test integ.u == integ.uprev
        for child in integ.child_subintegrators
            @test child.u == integ.u[dofs]
            @test child.t == integ.t
            @test child.tprev == integ.t
        end
    end

    @testset "Per-node adaptive defaults" begin
        # Without an `adaptive` keyword every node adapts exactly if its own
        # algorithm can: the splitting node stays fixed-step while a Tsit5 leaf
        # adapts and an Euler leaf does not.
        integ = DiffEqBase.init(prob, LieTrotterGodunov((Tsit5(), Euler())); dt = 0.1)
        @test integ.opts.adaptive == false
        @test integ.controller_cache === nothing
        @test integ.child_subintegrators[1].opts.adaptive == true
        @test integ.child_subintegrators[2].opts.adaptive == false
    end

    @testset "discontinuity_detection is refused up front" begin
        @test_throws ArgumentError DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 0.1,
            controller = OrdinaryDiffEqCore.IController(discontinuity_detection = true)
        )
    end

    @testset "Adaptive root with a nested non-adaptive splitting node" begin
        # B split once more into two identical halves handled by an inner LTG node.
        # The inner node stays non-adaptive by default, and a rejection at the root
        # has to roll the whole subtree back.
        odeBhalf(du, u, p, t) = (du[1] = 0.25 * u[2]; du[2] = 0.25 * u[1]; nothing)
        fBh = ODEFunction(odeBhalf)
        f_nested = GenericSplitFunction(
            (fA, GenericSplitFunction((fBh, fBh), (dofs, dofs))), (dofs, dofs)
        )
        prob_nested = OperatorSplittingProblem(f_nested, u0, tspan)
        alg_nested = PPLTG((Tsit5(), LieTrotterGodunov((Tsit5(), Tsit5()))))

        integ = DiffEqBase.init(
            prob_nested, alg_nested; dt = 0.4, reltol = 1.0e-5, abstol = 1.0e-8
        )
        @test integ.controller_cache !== nothing
        @test integ.child_subintegrators[2].controller_cache === nothing
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.nreject ≥ 1
        @test maximum(abs, integ.u .- trueu) < 1.0e-3
    end

    @testset "Three-operator PPLTG adapts" begin
        # A plus the strictly upper and lower triangles of B: pairwise non-commuting,
        # so the N-ary pair difference measures a genuine splitting error.
        odeB1(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.0; nothing)
        odeB2(du, u, p, t) = (du[1] = 0.0; du[2] = 0.5 * u[1]; nothing)
        fsplit3 = GenericSplitFunction(
            (fA, ODEFunction(odeB1), ODEFunction(odeB2)), (dofs, dofs, dofs)
        )
        prob3 = OperatorSplittingProblem(fsplit3, u0, tspan)
        integ = DiffEqBase.init(
            prob3, PPLTG((Tsit5(), Tsit5(), Tsit5()));
            dt = 0.5, reltol = 1.0e-6, abstol = 1.0e-8
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.nreject ≥ 1
        @test maximum(abs, integ.u .- trueu) < 1.0e-3
    end

    @testset "Nested PPLTG: the inner node adapts and rejects on its own" begin
        # B split into its strictly lower and upper triangle, which do not commute,
        # handled by an inner adaptive PPLTG. The inner node gets a much tighter
        # splitting tolerance than the root, so it has to reject and subcycle
        # within the intervals the root hands it.
        odeBu(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.0; nothing)
        odeBl(du, u, p, t) = (du[1] = 0.0; du[2] = 0.5 * u[1]; nothing)
        f_nested = GenericSplitFunction(
            (fA, GenericSplitFunction((ODEFunction(odeBu), ODEFunction(odeBl)), (dofs, dofs))),
            (dofs, dofs)
        )
        prob_nested = OperatorSplittingProblem(f_nested, u0, tspan)
        alg_nested = PPLTG((Tsit5(), PPLTG((Tsit5(), Tsit5()))))

        reltol = TreeOption(f_nested, 1.0e-3)
        reltol[2] = 1.0e-8
        integ = DiffEqBase.init(
            prob_nested, alg_nested; dt = 0.5, reltol, abstol = 1.0e-10
        )
        sub = integ.child_subintegrators[2]
        @test sub.controller_cache !== nothing
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        # The inner node ran its own controller: it rejected at least once and
        # accepted far more (sub-cycled) steps than the root.
        @test sub.stats.nreject ≥ 1
        @test sub.stats.naccept > integ.stats.naccept
        @test sub.EEst ≤ 1
        @test maximum(abs, integ.u .- trueu) < 1.0e-2
    end

    @testset "PIController threads its state between steps" begin
        controller = OrdinaryDiffEqCore.PIController(0.35, 0.2)
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 0.1, controller, reltol = 1.0e-6, abstol = 1.0e-8
        )
        @test integ.controller_cache isa OrdinaryDiffEqCore.PIControllerCache
        errold0 = integ.controller_cache.errold
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.controller_cache.errold != errold0 # the error history was carried
        @test maximum(abs, integ.u .- trueu) < 1.0e-3
        ufinal = copy(integ.u)
        niters = integ.iter
        # reinit! resets the controller memory, so a rerun is bit-identical.
        DiffEqBase.reinit!(integ)
        @test integ.controller_cache.errold == errold0
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.u == ufinal
        @test integ.iter == niters
    end

    @testset "Explicitly passed controller is used" begin
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 0.1, controller = OrdinaryDiffEqCore.IController()
        )
        @test integ.controller_cache isa OrdinaryDiffEqCore.IControllerCache
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
    end
end
