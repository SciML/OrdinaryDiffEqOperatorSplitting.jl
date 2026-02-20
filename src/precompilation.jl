using PrecompileTools: @compile_workload
using OrdinaryDiffEqLowOrderRK: Euler
import DiffEqBase: DiffEqBase, step!, solve!

function _precompile_ode1(du, u, p, t)
    @. du = -0.1u
    return
end

function _precompile_ode2(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
    return
end

function _precompile_ode3(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
end

@compile_workload begin
    # Setup minimal test problem for precompilation
    tspan = (0.0, 0.1)
    u0 = [1.0, 1.0, 1.0]

    f1 = DiffEqBase.ODEFunction(_precompile_ode1)
    f2 = DiffEqBase.ODEFunction(_precompile_ode2)
    f3 = DiffEqBase.ODEFunction(_precompile_ode3)

    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    f3dofs = [2, 3]
    fsplitinner = GenericSplitFunction((f2, f3), (f2dofs, f3dofs))
    fsplit = GenericSplitFunction((f1, fsplitinner), (f1dofs, [1,2,3]))

    prob = OperatorSplittingProblem(fsplit, u0, tspan)
    tstepper = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))

    # Precompile init and a few steps
    integrator = DiffEqBase.init(prob, tstepper, dt = 0.01, verbose = false)
    step!(integrator)
    solve!(integrator)
end
