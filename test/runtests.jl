using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import UnPack: @unpack

import SciMLBase: SciMLBase, ReturnCode,
import DiffEqBase: DiffEqBase, ODEFunction, ODEProblem
using OrdinaryDiffEqTsit5

@testset "Operator Splitting API" begin
    # Reference
    function ode_true(du, u, p, t)
        du .= -0.1u
        du[1] += 0.01u[3]
        du[3] += 0.01u[1]
    end

    # Setup individual functions
    # Diagonal components
    function ode1(du, u, p, t)
        @. du = -0.1u
    end
    # Offdiagonal components
    function ode2(du, u, p, t)
        du[1] = 0.01u[2]
        du[2] = 0.01u[1]
    end

    f1 = ODEFunction(ode1)
    f2 = ODEFunction(ode2)

    # Here we describe index sets f1dofs and f2dofs that map the
    # local indices in f1 and f2 into the global problem. Just put
    # ode_true and ode1/ode2 side by side to see how they connect.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    fsplit1 = GenericSplitFunction((f1,f2), (f1dofs, f2dofs))

    # Now the usual setup just with our new problem type.
    # u0 = rand(3)
    u0 = [0.7611944793397108
        0.9059606424982555
        0.5755174199139956]
    tspan = (0.0,100.0)
    prob = OperatorSplittingProblem(fsplit1, u0, tspan)

    # Now some recursive splitting
    function ode3(du, u, p, t)
        du[1] = 0.005u[2]
        du[2] = 0.005u[1]
    end
    f3 = ODEFunction(ode3)
    # The time stepper carries the individual solver information.

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    f3dofs = [1,3]
    fsplit2_inner = GenericSplitFunction((f2,f3), (f3dofs, f3dofs))
    fsplit2_outer = GenericSplitFunction((f1,fsplit2_inner), (f1dofs, f2dofs))

    prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)

    function ode_NaN(du, u, p, t)
        du[1] = NaN
        du[2] = 0.01u[1]
    end

    f_NaN = ODEFunction(ode_NaN)
    f_NaN_dofs = f3dofs
    fsplit_NaN = GenericSplitFunction((f1,f_NaN), (f1dofs, f_NaN_dofs))
    prob_NaN = OperatorSplittingProblem(fsplit_NaN, u0, tspan)

    function ode2_force_half(du, u, p, t)
        du[1] = 0.5
        du[2] = 0.5
    end

    f2half = ODEFunction(ode2_force_half)
    fsplit_force_half = GenericSplitFunction((f1,f2half), (f1dofs, f2dofs))
    prob_force_half = OperatorSplittingProblem(fsplit_force_half, u0, tspan)

    dt = 0.01π
    @testset "OperatorSplitting" begin
        for TimeStepperType in (LieTrotterGodunov,)
            timestepper = TimeStepperType(
                (Tsit5(), Tsit5())
            )
            timestepper_inner = TimeStepperType(
                (Tsit5(), Tsit5())
            )
            timestepper2 = TimeStepperType(
                (Tsit5(), timestepper_inner)
            )

            for (tstepper1, tstepper_inner, tstepper2) in (
                    (timestepper, timestepper_inner, timestepper2),
                    )
                # The remaining code works as usual.
                integrator = DiffEqBase.init(prob, tstepper1, dt=dt, verbose=true, alias_u0=false)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal = copy(integrator.u)
                @test ufinal ≉ u0 # Make sure the solve did something

                # DiffEqBase.reinit!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0)
                # end
                # @test  isapprox(ufinal, integrator.u, atol=1e-8)

                # DiffEqBase.reinit!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
                # end
                # @test  isapprox(ufinal, integrator.u, atol=1e-8)

                # DiffEqBase.reinit!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # DiffEqBase.solve!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success

                integrator2 = DiffEqBase.init(prob2, tstepper2, dt=dt, verbose=true, alias_u0=false)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal2 = copy(integrator2.u)
                @test ufinal2 ≉ u0 # Make sure the solve did something

                # DiffEqBase.reinit!(integrator2)
                # @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                # for (u, t) in DiffEqBase.TimeChoiceIterator(integrator2, 0.0:5.0:100.0)
                # end
                # @test isapprox(ufinal2, integrator2.u, atol=1e-8)

                # DiffEqBase.reinit!(integrator2)
                # @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                # DiffEqBase.solve!(integrator2)
                # @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Success
                @testset "NaNs" begin
                    integrator_NaN = DiffEqBase.init(prob_NaN, tstepper1, dt=dt, verbose=true, alias_u0=false)
                    @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                    DiffEqBase.solve!(integrator_NaN)
                    @test integrator_NaN.sol.retcode ∈ (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
                end
            end
        end
    end
end
