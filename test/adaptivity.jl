using OrdinaryDiffEqOperatorSplitting
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
