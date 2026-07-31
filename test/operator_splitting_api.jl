using OrdinaryDiffEqOperatorSplitting
using Test

import SciMLBase: ReturnCode
import DiffEqBase: DiffEqBase, ODEFunction, ODEProblem
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using ModelingToolkit
using SciMLIterators: TimeChoiceIterator, intervals

# ---------------------------------------------------------------------------
# Reference problem
# ---------------------------------------------------------------------------
tspan = (0.0, 100.0)
u0 = [
    0.7611944793397108
    0.9059606424982555
    0.5755174199139956
]
trueA = [
    -0.1 0.0 -0.0;
    0.0 -0.1 0.0;
    -0.0 0.0 -0.1
]
trueB = [
    -0.0 0.0 -0.01;
    0.0 -0.0 0.0;
    -0.01 0.0 -0.0
]
function ode_true(du, u, p, t)
    du .= -0.1u
    du[1] -= 0.01u[3]
    return du[3] -= 0.01u[1]
end
trueu = exp((tspan[2] - tspan[1]) * (trueA + trueB)) * u0

# Setup individual functions
function ode1(du, u, p, t)
    return @. du = -0.1u
end
f1 = ODEFunction(ode1)

function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    return du[2] = -0.01u[1]
end
f2 = ODEFunction(ode2)

function ode3(du, u, p, t)
    du[1] = -0.005u[2]
    return du[2] = -0.005u[1]
end
f3 = ODEFunction(ode3)

@independent_variables time
Dt = Differential(time)
@variables u1(time) u2(time)
eqs = [
    Dt(u1) ~ -0.01u2,
    Dt(u2) ~ -0.01u1,
]
@named testmodel2 = System(eqs, time)
testsys2 = mtkcompile(testmodel2; sort_eqs = false)

# Steps a child takes per outer step: StrangMarchuk steps child 1 twice (two
# half-steps), the palindromic pair steps every child twice (once per sequence).
_sub1_iter_factor(::LieTrotterGodunov) = 1
_sub1_iter_factor(::StrangMarchuk) = 2
_sub1_iter_factor(::PalindromicPairLieTrotterGodunov) = 2
_sub2_iter_factor(::LieTrotterGodunov) = 1
_sub2_iter_factor(::StrangMarchuk) = 1
_sub2_iter_factor(::PalindromicPairLieTrotterGodunov) = 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@testset "reinit and convergence" begin
    dt = 0.01π

    # Here we describe index sets f1dofs and f2dofs that map the
    # local indices in f1 and f2 into the global problem. Just put
    # ode_true and ode1/ode2 side by side to see how they connect.
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    fsplit1a = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))
    fsplit1b = GenericSplitFunction((f1, testsys2), (f1dofs, f2dofs))

    prob1a = OperatorSplittingProblem(fsplit1a, u0, tspan)
    prob1b = OperatorSplittingProblem(fsplit1b, u0, tspan)

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f3dofs = [1, 2]
    fsplit2_inner = GenericSplitFunction((f3, f3), (f3dofs, f3dofs))
    fsplit2_outer = GenericSplitFunction((f1, fsplit2_inner), (f1dofs, f2dofs))

    prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)

    nsteps = ceil(Int, (tspan[2] - tspan[1]) / dt)

    for TimeStepperType in (LieTrotterGodunov, StrangMarchuk, PalindromicPairLieTrotterGodunov)
        @testset "$tstepper" for (prob, tstepper) in (
                (prob1a, TimeStepperType((Euler(), Euler()))),
                (prob1a, TimeStepperType((Tsit5(), Euler()))),
                (prob1a, TimeStepperType((Euler(), Tsit5()))),
                (prob1a, TimeStepperType((Tsit5(), Tsit5()))),
                (prob1b, TimeStepperType((Euler(), Euler()))),
                (prob1b, TimeStepperType((Tsit5(), Euler()))),
                (prob1b, TimeStepperType((Euler(), Tsit5()))),
                (prob1b, TimeStepperType((Tsit5(), Tsit5()))),
                (prob2, TimeStepperType((Euler(), TimeStepperType((Euler(), Euler()))))),
                (prob2, TimeStepperType((Euler(), TimeStepperType((Tsit5(), Euler()))))),
                (prob2, TimeStepperType((Euler(), TimeStepperType((Euler(), Tsit5()))))),
                (prob2, TimeStepperType((Tsit5(), TimeStepperType((Tsit5(), Euler()))))),
                (prob2, TimeStepperType((Tsit5(), TimeStepperType((Euler(), Tsit5()))))),
                (prob2, TimeStepperType((Tsit5(), TimeStepperType((Tsit5(), Tsit5()))))),
            )
            integrator = DiffEqBase.init(
                prob, tstepper, dt = dt, verbose = true, alias_u0 = false, adaptive = false
            )
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default

            sub1 = integrator.child_subintegrators[1]
            sub2 = integrator.child_subintegrators[2]
            expected_sub1_iters = _sub1_iter_factor(tstepper) * nsteps
            expected_sub2_iters = _sub2_iter_factor(tstepper) * nsteps

            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            ufinal = copy(integrator.u)
            @test isapprox(ufinal, trueu, atol = 1.0e-6)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            @test sub1.t ≈ tspan[2]
            @test sub1.iter == expected_sub1_iters

            @test sub2.t ≈ tspan[2]
            @test sub2.iter == expected_sub2_iters

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (u, t) in TimeChoiceIterator(integrator, tspan[1]:5.0:tspan[2])
            end
            @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (uprev, tprev, u, t) in intervals(integrator)
            end
            @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            @test sub1.t ≈ tspan[2]
            @test sub1.iter == expected_sub1_iters

            @test sub2.t ≈ tspan[2]
            @test sub2.iter == expected_sub2_iters
        end
    end

    @testset "Adaptive splitting | $tstepper" for (prob, tstepper) in (
            (prob1a, PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5()))),
            (
                prob2, PalindromicPairLieTrotterGodunov(
                    (Tsit5(), PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5())))
                ),
            ),
        )
        # PPLTG is adaptive by default; the integrator interface has to keep working
        # while the controller reshapes the step sequence.
        integrator = DiffEqBase.init(
            prob, tstepper, dt = dt, verbose = true, alias_u0 = false
        )
        @test integrator.opts.adaptive
        @test integrator.controller_cache !== nothing
        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
        DiffEqBase.solve!(integrator)
        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
        ufinal = copy(integrator.u)
        @test isapprox(ufinal, trueu, atol = 1.0e-6)
        @test integrator.t ≈ tspan[2]
        niters = integrator.iter

        # A reinitialized adaptive solve is deterministic.
        DiffEqBase.reinit!(integrator; dt = dt)
        @test integrator.dt == dt
        @test integrator.dt == integrator.dtcache
        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
        DiffEqBase.solve!(integrator)
        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
        @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
        @test integrator.t ≈ tspan[2]
        @test integrator.iter == niters

        # The iteration protocols land on the same time points despite the
        # controller-driven step sequence in between.
        DiffEqBase.reinit!(integrator; dt = dt)
        for (u, t) in TimeChoiceIterator(integrator, tspan[1]:5.0:tspan[2])
        end
        @test isapprox(integrator.u, trueu, atol = 1.0e-6)
        @test integrator.t ≈ tspan[2]

        DiffEqBase.reinit!(integrator; dt = dt)
        for (uprev, tprev, u, t) in intervals(integrator)
        end
        @test isapprox(integrator.u, trueu, atol = 1.0e-6)
        @test integrator.t ≈ tspan[2]
    end

    @testset "PalindromicPairLieTrotterGodunov requires two operators" begin
        @test_throws ArgumentError PalindromicPairLieTrotterGodunov(
            (Euler(), Euler(), Euler())
        )
        @test_throws ArgumentError PalindromicPairLieTrotterGodunov((Euler(),))
    end

    @testset "StrangMarchuk with 3 operators" begin
        dt = 0.01π
        # f1 + f3 + f3 = f1 + f2, so the reference solution is the same trueu.
        f1dofs = [1, 2, 3]
        f3dofs = [1, 3]
        fsplit3 = GenericSplitFunction((f1, f3, f3), (f1dofs, f3dofs, f3dofs))
        prob3 = OperatorSplittingProblem(fsplit3, u0, tspan)
        nsteps = ceil(Int, (tspan[2] - tspan[1]) / dt)

        @testset "$tstepper" for tstepper in (
                StrangMarchuk((Euler(), Euler(), Euler())),
                StrangMarchuk((Tsit5(), Euler(), Tsit5())),
                StrangMarchuk((Tsit5(), Tsit5(), Tsit5())),
            )
            integrator = DiffEqBase.init(
                prob3, tstepper, dt = dt, verbose = true, alias_u0 = false, adaptive = false
            )
            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            @test isapprox(integrator.u, trueu, atol = 1.0e-6)
            @test integrator.t ≈ tspan[2]
            @test integrator.iter == nsteps

            sub1 = integrator.child_subintegrators[1]
            sub2 = integrator.child_subintegrators[2]
            sub3 = integrator.child_subintegrators[3]
            # Palindromic: children 1 & 2 get two half-steps, child 3 gets one full step
            @test sub1.iter == 2 * nsteps
            @test sub2.iter == 2 * nsteps
            @test sub3.iter == nsteps
        end
    end

    @testset "Convergence order" begin
        # Use non-commuting operators so splitting error is non-zero.
        # A = diag(-1,-2), B = [0 0.5; 0.5 0] have [A,B] ≠ 0.
        function ode_conv_A(du, u, p, t)
            du[1] = -u[1]
            return du[2] = -2 * u[2]
        end
        function ode_conv_B(du, u, p, t)
            du[1] = 0.5 * u[2]
            return du[2] = 0.5 * u[1]
        end
        fA = ODEFunction(ode_conv_A)
        fB = ODEFunction(ode_conv_B)

        conv_tspan = (0.0, 1.0)
        conv_u0 = [1.0, 1.0]
        conv_trueu = exp(conv_tspan[2] * [-1.0 0.5; 0.5 -2.0]) * conv_u0

        conv_dofs = [1, 2]
        fsplit_conv = GenericSplitFunction((fA, fB), (conv_dofs, conv_dofs))
        prob_conv = OperatorSplittingProblem(fsplit_conv, conv_u0, conv_tspan)

        dts = [0.1, 0.05, 0.025]
        for (TimeStepperType, expected_order) in (
                (LieTrotterGodunov, 1),
                (StrangMarchuk, 2),
                (PalindromicPairLieTrotterGodunov, 2),
            )
            @testset "$TimeStepperType (order $expected_order)" begin
                errors = map(dts) do dt_i
                    tstepper = TimeStepperType((Tsit5(), Tsit5()))
                    integrator = DiffEqBase.init(
                        prob_conv, tstepper, dt = dt_i, verbose = false,
                        alias_u0 = false, adaptive = false
                    )
                    DiffEqBase.solve!(integrator)
                    maximum(abs, integrator.u .- conv_trueu)
                end
                for i in 1:(length(errors) - 1)
                    rate = log2(errors[i] / errors[i + 1])
                    @test rate ≈ expected_order atol = 0.3
                end
            end
        end
    end

    @testset "Instability detection" begin
        dt = 0.01π

        function ode_NaN(du, u, p, t)
            du[1] = NaN
            du[2] = 0.01u[1]
        end

        f1dofs = [1, 2, 3]
        f3dofs = [1, 3]

        f_NaN = ODEFunction(ode_NaN)
        fsplit_NaN = GenericSplitFunction((f1, f_NaN), (f1dofs, f3dofs))
        prob_NaN = OperatorSplittingProblem(fsplit_NaN, u0, tspan)

        for TimeStepperType in (LieTrotterGodunov, StrangMarchuk, PalindromicPairLieTrotterGodunov)
            @testset "Solver type $TimeStepperType | $tstepper" for tstepper in (
                    TimeStepperType((Euler(), Euler())),
                    TimeStepperType((Tsit5(), Euler())),
                    TimeStepperType((Euler(), Tsit5())),
                    TimeStepperType((Tsit5(), Tsit5())),
                )
                integrator_NaN = DiffEqBase.init(
                    prob_NaN, tstepper, dt = dt, verbose = true, alias_u0 = false
                )
                @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator_NaN)
                @test integrator_NaN.sol.retcode ∈
                    (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
            end
        end
    end
end
