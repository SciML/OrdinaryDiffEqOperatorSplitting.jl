using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
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

# Mock solver extension exercising the documented developer extension interface v1:
# an algorithm carrying `inner_algs`, a concrete cache, `init_cache`, and
# `_perform_step!`. Sweeping the operators in reverse order is still a complete
# first-order splitting. It does not give a different answer than
# LieTrotterGodunov on the reference problem -- `trueA` is a multiple of the
# identity, so the two operators commute and the sweep order cancels out.
struct ReverseLieTrotterGodunov{AlgTupleType} <: OS.AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType
end

struct ReverseLieTrotterGodunovCache{uType, uprevType} <: OS.AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
end

function OS.init_cache(
        f::GenericSplitFunction, alg::ReverseLieTrotterGodunov;
        uprev::AbstractArray, u::AbstractVector,
    )
    return ReverseLieTrotterGodunovCache(u, uprev)
end

function OS._perform_step!(
        parent, children::Tuple, cache::ReverseLieTrotterGodunovCache, dt
    )
    for i in reverse(eachindex(children))
        child = children[i]
        idxs = parent.child_solution_indices[i]
        sync = parent.child_synchronizers[i]

        OS.forward_sync_subintegrator!(parent, child, idxs, sync)
        OS.advance_solution_by!(parent, child, dt)
        if OS._child_failed(child)
            parent.force_stepfail = true
            return nothing
        end
        OS.backward_sync_subintegrator!(parent, child, idxs, sync)
    end
    return nothing
end

@testset "solver extension interface v1" begin
    split_f = GenericSplitFunction((f1, f2), ([1, 2, 3], [1, 3]))
    prob = OperatorSplittingProblem(split_f, u0, tspan)
    alg = ReverseLieTrotterGodunov((Euler(), Euler()))

    integrator = DiffEqBase.init(prob, alg; dt = 0.01, verbose = false)
    @test integrator.cache isa ReverseLieTrotterGodunovCache
    DiffEqBase.solve!(integrator)
    @test integrator.sol.retcode == ReturnCode.Success

    function err(dt)
        integ = DiffEqBase.init(prob, alg; dt = dt, verbose = false)
        DiffEqBase.solve!(integ)
        return maximum(abs, integ.u - trueu)
    end
    @test err(0.1) / err(0.01) ≈ 10 rtol = 0.1
end

@testset "nested rollback restores child buffers" begin
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    f3dofs = [1, 2]
    nested_f = GenericSplitFunction((f3, f3), (f3dofs, f3dofs))
    split_f = GenericSplitFunction((f1, nested_f), (f1dofs, f2dofs))
    prob = OperatorSplittingProblem(split_f, u0, (0.0, 1.0))
    alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
    integrator = DiffEqBase.init(prob, alg; dt = 0.1, adaptive = false, verbose = false)

    nested_child = integrator.child_subintegrators[2]
    nested_child.u .= 0
    nested_child.uprev .= 0
    OS.rollback_children!(integrator)

    @test nested_child.u == integrator.u[nested_child.solution_indices]
    @test nested_child.uprev == nested_child.u
end


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

    @testset "Three operators" begin
        dt = 0.01π
        # f1 + f3 + f3 = f1 + f2, so the reference solution is the same trueu.
        f1dofs = [1, 2, 3]
        f3dofs = [1, 3]
        fsplit3 = GenericSplitFunction((f1, f3, f3), (f1dofs, f3dofs, f3dofs))
        prob3 = OperatorSplittingProblem(fsplit3, u0, tspan)
        nsteps = ceil(Int, (tspan[2] - tspan[1]) / dt)

        # StrangMarchuk solves all but the last operator twice (two half-steps);
        # the palindromic pair solves every operator once per sequence.
        sub_iter_factors(::StrangMarchuk) = (2, 2, 1)
        sub_iter_factors(::PalindromicPairLieTrotterGodunov) = (2, 2, 2)

        @testset "$tstepper" for tstepper in (
                StrangMarchuk((Euler(), Euler(), Euler())),
                StrangMarchuk((Tsit5(), Euler(), Tsit5())),
                StrangMarchuk((Tsit5(), Tsit5(), Tsit5())),
                PalindromicPairLieTrotterGodunov((Euler(), Euler(), Euler())),
                PalindromicPairLieTrotterGodunov((Tsit5(), Euler(), Tsit5())),
                PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5(), Tsit5())),
            )
            integrator = DiffEqBase.init(
                prob3, tstepper, dt = dt, verbose = true, alias_u0 = false, adaptive = false
            )
            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            @test isapprox(integrator.u, trueu, atol = 1.0e-6)
            @test integrator.t ≈ tspan[2]
            @test integrator.iter == nsteps

            for (i, factor) in pairs(sub_iter_factors(tstepper))
                @test integrator.child_subintegrators[i].t ≈ tspan[2]
                @test integrator.child_subintegrators[i].iter == factor * nsteps
            end
        end
    end

    # Convergence orders are covered systematically in test/convergence.jl.

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
            # An adaptive root (PPLTG by default) retries escalated non-adaptive
            # failures until its dt reaches dtmin, so it may also end in
            # DtLessThanMin instead of surfacing the child diagnosis directly.
            expected_retcodes = if TimeStepperType === PalindromicPairLieTrotterGodunov
                (
                    DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN,
                    DiffEqBase.ReturnCode.DtLessThanMin,
                )
            else
                (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
            end
            @testset "Solver type $TimeStepperType | $tstepper" for tstepper in (
                    TimeStepperType((Euler(), Euler())),
                    TimeStepperType((Tsit5(), Euler())),
                    TimeStepperType((Euler(), Tsit5())),
                    TimeStepperType((Tsit5(), Tsit5())),
                )
                integrator_NaN = DiffEqBase.init(
                    prob_NaN, tstepper, dt = dt, verbose = false, alias_u0 = false
                )
                @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator_NaN)
                @test integrator_NaN.sol.retcode ∈ expected_retcodes
            end
        end
    end
end
