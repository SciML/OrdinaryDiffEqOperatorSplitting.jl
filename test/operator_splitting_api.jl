using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting: OrdinaryDiffEqOperatorSplitting as OS
using Test

import SciMLBase: ReturnCode
import DiffEqBase: DiffEqBase, ODEFunction, ODEProblem
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using ModelingToolkit

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

# Test whether adaptive code path works in principle
struct FakeAdaptiveAlgorithm{T, T2} <: OS.AbstractOperatorSplittingAlgorithm
    alg::T
    inner_algs::T2   # delegate inner_algs to the wrapped algorithm
end
FakeAdaptiveAlgorithm(alg) = FakeAdaptiveAlgorithm(alg, alg.inner_algs)

struct FakeAdaptiveAlgorithmCache{T} <: OS.AbstractOperatorSplittingCache
    cache::T
end

@inline DiffEqBase.isadaptive(::FakeAdaptiveAlgorithm) = true

@inline function OS.stepsize_controller!(
        integrator::OS.OperatorSplittingIntegrator, alg::FakeAdaptiveAlgorithm
    )
    return nothing
end

@inline function OS.step_accept_controller!(
        integrator::OS.OperatorSplittingIntegrator, alg::FakeAdaptiveAlgorithm, q
    )
    integrator.dt = integrator.dtcache
    return nothing
end
@inline function OS.step_reject_controller!(
        integrator::OS.OperatorSplittingIntegrator, alg::FakeAdaptiveAlgorithm, q
    )
    error("The tests should never run into this scenario!")
    return nothing
end

# Override init_cache to wrap the inner cache in FakeAdaptiveAlgorithmCache
function OS.init_cache(
        f::GenericSplitFunction, alg::FakeAdaptiveAlgorithm;
        kwargs...
    )
    inner_cache = OS.init_cache(f, alg.alg; kwargs...)
    return FakeAdaptiveAlgorithmCache(inner_cache)
end

@inline DiffEqBase.get_tmp_cache(
    integrator::OS.OperatorSplittingIntegrator,
    alg::OS.AbstractOperatorSplittingAlgorithm,
    cache::FakeAdaptiveAlgorithmCache
) = DiffEqBase.get_tmp_cache(integrator, alg, cache.cache)

@inline function OS._perform_step!(
        outer_integrator,
        subintegrators::Tuple,
        cache::FakeAdaptiveAlgorithmCache,
        dt
    )
    return OS._perform_step!(
        outer_integrator, subintegrators, cache.cache, dt
    )
end

FakeAdaptiveLTG(inner) = FakeAdaptiveAlgorithm(LieTrotterGodunov(inner))

function Base.show(io::IO, alg::FakeAdaptiveAlgorithm)
    print(io, "FAKE (")
    Base.show(io, alg.alg)
    return print(io, ")")
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

    for TimeStepperType in (LieTrotterGodunov, FakeAdaptiveLTG)
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

            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            ufinal = copy(integrator.u)
            @test isapprox(ufinal, trueu, atol = 1.0e-6)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            @test sub1.t ≈ tspan[2]
            @test sub1.iter == nsteps

            @test sub2.t ≈ tspan[2]
            @test sub2.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            while !SciMLBase.done(integrator)
                DiffEqBase.step!(integrator)
            end
            @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            while !SciMLBase.done(integrator)
                DiffEqBase.step!(integrator)
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
            @test sub1.iter == nsteps

            @test sub2.t ≈ tspan[2]
            @test sub2.iter == nsteps
        end
    end

    for TimeStepperType in (FakeAdaptiveLTG,)
        @testset "Adaptive solver type $TimeStepperType | $tstepper" for (prob, tstepper) in (
                (prob1a, TimeStepperType((Tsit5(), Tsit5()))),
                (prob2, TimeStepperType((Tsit5(), TimeStepperType((Tsit5(), Tsit5()))))),
            )
            integrator = DiffEqBase.init(
                prob, tstepper, dt = dt, verbose = true, alias_u0 = false, adaptive = true
            )
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            ufinal = copy(integrator.u)
            @test isapprox(ufinal, trueu, atol = 1.0e-6)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.dt == dt
            @test integrator.dt == integrator.dtcache
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            while !SciMLBase.done(integrator)
                DiffEqBase.step!(integrator)
            end
            @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            while !SciMLBase.done(integrator)
                DiffEqBase.step!(integrator)
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

        for TimeStepperType in (LieTrotterGodunov,)
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

    @testset "reinit! resets dtcache when dt changes" begin
        dt_coarse = 0.1
        dt_fine = dt_coarse / 2

        fsplit_rc = GenericSplitFunction((f1, f2), ([1, 2, 3], [1, 3]))
        prob_rc = OperatorSplittingProblem(fsplit_rc, u0, tspan)
        tstepper_rc = LieTrotterGodunov((Euler(), Euler()))

        integrator_rc = DiffEqBase.init(prob_rc, tstepper_rc; dt = dt_coarse, alias_u0 = false)
        DiffEqBase.solve!(integrator_rc)
        @test integrator_rc.sol.retcode == DiffEqBase.ReturnCode.Success
        @test integrator_rc.dtcache ≈ dt_coarse

        DiffEqBase.reinit!(integrator_rc; dt = dt_fine)
        @test integrator_rc.dt ≈ dt_fine
        @test integrator_rc.dtcache ≈ dt_fine
        DiffEqBase.solve!(integrator_rc)
        @test integrator_rc.sol.retcode == DiffEqBase.ReturnCode.Success
        @test integrator_rc.dtcache ≈ dt_fine
    end

    @testset "Nested instability propagation" begin
        dt = 0.01π

        function ode_nan_nested(du, u, p, t)
            du[1] = NaN
            du[2] = 0.01u[1]
        end
        f_nan_nested = ODEFunction(ode_nan_nested)

        # Inner split: one leg produces NaN, nested inside an outer split.
        # f3dofs_n = [1,2] indexes into the 2-element view selected by f2dofs = [1,3].
        fsplit_inner_n = GenericSplitFunction((f3, f_nan_nested), ([1, 2], [1, 2]))
        fsplit_outer_n = GenericSplitFunction((f1, fsplit_inner_n), ([1, 2, 3], [1, 3]))
        prob_nested = OperatorSplittingProblem(fsplit_outer_n, u0, tspan)

        tstepper_n = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        integrator_n = DiffEqBase.init(prob_nested, tstepper_n; dt = dt, alias_u0 = false)
        DiffEqBase.solve!(integrator_n)
        @test integrator_n.sol.retcode ∈ (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
    end

    @testset "verbose=false suppresses warnings without affecting retcode" begin
        dt = 0.01π

        function ode_nan_quiet(du, u, p, t)
            du[1] = NaN
            du[2] = 0.01u[1]
        end
        f_nan_quiet = ODEFunction(ode_nan_quiet)

        fsplit_q = GenericSplitFunction((f1, f_nan_quiet), ([1, 2, 3], [1, 3]))
        prob_q = OperatorSplittingProblem(fsplit_q, u0, tspan)

        integrator_q = DiffEqBase.init(
            prob_q, LieTrotterGodunov((Euler(), Euler()));
            dt = dt, verbose = false, alias_u0 = false
        )
        DiffEqBase.solve!(integrator_q)
        @test integrator_q.sol.retcode ∈ (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
    end
end
