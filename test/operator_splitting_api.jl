using OrdinaryDiffEqOperatorSplitting
using Test

import SciMLBase: ReturnCode
import DiffEqBase: DiffEqBase, ODEFunction, ODEProblem
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

# Reference
tspan = (0.0, 100.0)
u0 = [0.7611944793397108
      0.9059606424982555
      0.5755174199139956]
trueA = [-0.1 0.0 -0.0;
         0.0 -0.1 0.0;
         -0.0 0.0 -0.1]
trueB = [-0.0 0.0 -0.01;
         0.0 -0.0 0.0;
         -0.01 0.0 -0.0]
function ode_true(du, u, p, t)
    du .= -0.1u
    du[1] -= 0.01u[3]
    du[3] -= 0.01u[1]
end
trueu = exp((tspan[2] - tspan[1]) * (trueA + trueB)) * u0

# Setup individual functions
# Diagonal components
function ode1(du, u, p, t)
    @. du = -0.1u
end
f1 = ODEFunction(ode1)

# Offdiagonal components
function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
end
f2 = ODEFunction(ode2)

@testset "reinit and convergence" begin
    dt = 0.01π

    # Here we describe index sets f1dofs and f2dofs that map the
    # local indices in f1 and f2 into the global problem. Just put
    # ode_true and ode1/ode2 side by side to see how they connect.
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    fsplit1 = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))

    # Now the usual setup just with our new problem type.
    prob1 = OperatorSplittingProblem(fsplit1, u0, tspan)

    # Now some recursive splitting
    function ode3(du, u, p, t)
        du[1] = -0.005u[2]
        du[2] = -0.005u[1]
    end
    f3 = ODEFunction(ode3)
    # The time stepper carries the individual solver information.

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    f3dofs = [1, 3]
    fsplit2_inner = GenericSplitFunction((f3, f3), (f3dofs, f3dofs))
    fsplit2_outer = GenericSplitFunction((f1, fsplit2_inner), (f1dofs, f2dofs))

    prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)
    for TimeStepperType in (LieTrotterGodunov,)
        @testset "Solver type $TimeStepperType | $tstepper" for (prob, tstepper) in (
            (prob1, TimeStepperType((Euler(), Euler()))),
            (prob1, TimeStepperType((Tsit5(), Euler()))),
            (prob1, TimeStepperType((Euler(), Tsit5()))),
            (prob1, TimeStepperType((Tsit5(), Tsit5()))),
            (prob2, TimeStepperType((Euler(), TimeStepperType((Euler(), Euler()))))),
            (prob2, TimeStepperType((Euler(), TimeStepperType((Tsit5(), Euler()))))),
            (prob2, TimeStepperType((Euler(), TimeStepperType((Euler(), Tsit5()))))),
            (prob2, TimeStepperType((Tsit5(), TimeStepperType((Tsit5(), Euler()))))),
            (prob2, TimeStepperType((Tsit5(), TimeStepperType((Euler(), Tsit5()))))),
            (prob2, TimeStepperType((Tsit5(), TimeStepperType((Tsit5(), Tsit5())))))
        )
            # The remaining code works as usual.
            integrator = DiffEqBase.init(
                prob, tstepper, dt = dt, verbose = true, alias_u0 = false)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            ufinal = copy(integrator.u)
            @test isapprox(ufinal, trueu, atol = 1e-2)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            # @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)

            DiffEqBase.reinit!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, tspan[1]:5.0:tspan[2])
            end
            @test isapprox(ufinal, integrator.u, atol = 1e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            # @test integrator.iter == ...

            DiffEqBase.reinit!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
            end
            @test isapprox(ufinal, integrator.u, atol = 1e-12)
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)

            DiffEqBase.reinit!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
            DiffEqBase.solve!(integrator)
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
            @test integrator.t ≈ tspan[2]
            @test integrator.dtcache ≈ dt
            @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
        end
    end

    @testset "Instbility detectioon" begin
        dt = 0.01π

        function ode_NaN(du, u, p, t)
            du[1] = NaN
            du[2] = 0.01u[1]
        end

        f1dofs = [1, 2, 3]
        f2dofs = [1, 3]

        f_NaN = ODEFunction(ode_NaN)
        f_NaN_dofs = f3dofs
        fsplit_NaN = GenericSplitFunction((f1, f_NaN), (f1dofs, f_NaN_dofs))
        prob_NaN = OperatorSplittingProblem(fsplit_NaN, u0, tspan)

        for TimeStepperType in (LieTrotterGodunov,)
            @testset "Solver type $TimeStepperType | $tstepper" for (prob, tstepper) in (
                (prob1, TimeStepperType((Euler(), Euler()))),
                (prob1, TimeStepperType((Tsit5(), Euler()))),
                (prob1, TimeStepperType((Euler(), Tsit5()))),
                (prob1, TimeStepperType((Tsit5(), Tsit5())))
            )
                integrator_NaN = DiffEqBase.init(
                    prob_NaN, tstepper, dt = dt, verbose = true, alias_u0 = false)
                @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator_NaN)
                @test integrator_NaN.sol.retcode ∈
                      (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
            end
        end
    end
end
