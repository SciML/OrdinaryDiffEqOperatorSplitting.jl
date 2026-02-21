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
@mtkmodel TestModelODE2 begin
    @variables begin
        u1(time)
        u2(time)
    end
    @equations begin
        Dt(u1) ~ -0.01u2
        Dt(u2) ~ -0.01u1
    end
end
@named testmodel2 = TestModelODE2()
testsys2 = mtkcompile(testmodel2; sort_eqs = false)

# ---------------------------------------------------------------------------
# FakeAdaptiveAlgorithm — tests adaptive code path
#
# With the new interface FakeAdaptiveAlgorithm no longer needs to override
# build_subintegrator_tree_with_cache.  It just wraps the standard cache in
# its own FakeAdaptiveAlgorithmCache.
# ---------------------------------------------------------------------------
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
    print(io, ")")
end


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@testset "reinit and convergence" begin
    dt = 0.01π

    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    fsplit1a = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))
    fsplit1b = GenericSplitFunction((f1, testsys2), (f1dofs, f2dofs))

    prob1a = OperatorSplittingProblem(fsplit1a, u0, tspan)
    prob1b = OperatorSplittingProblem(fsplit1b, u0, tspan)

    f3dofs = [1, 3]
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

            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            ufinal = copy(integrator.u)
            @test isapprox(ufinal, trueu, atol = 1.0e-6)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            # SplitSubIntegrators now carry t and iter at each level
            sub1 = integrator.child_subintegrators[1]
            @test sub1.t ≈ tspan[2]
            @test sub1.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, tspan[1]:5.0:tspan[2])
            end
            @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
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
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, tspan[1]:5.0:tspan[2])
            end
            @test isapprox(ufinal, integrator.u, atol = 1.0e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == nsteps

            DiffEqBase.reinit!(integrator; dt = dt)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
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
end
