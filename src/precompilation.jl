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
    return du[2] = -0.01u[1]
end

@compile_workload begin
    # Setup minimal test problem for precompilation
    tspan = (0.0, 0.1)
    u0 = [1.0, 1.0, 1.0]

    f1 = SciMLBase.ODEFunction(_precompile_ode1)
    f2 = SciMLBase.ODEFunction(_precompile_ode2)
    f3 = SciMLBase.ODEFunction(_precompile_ode3)

    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    f3dofs = [2, 3]
    fsplitinner = GenericSplitFunction((f2, f3), (f2dofs, f3dofs))
    fsplit = GenericSplitFunction((f1, fsplitinner), (f1dofs, [1, 2, 3]))

    prob = OperatorSplittingProblem(fsplit, u0, tspan)

    # Precompile LieTrotterGodunov
    tstepper_ltg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
    integrator = DiffEqBase.init(prob, tstepper_ltg, dt = 0.01, verbose = false)
    step!(integrator)
    solve!(integrator)

    # Precompile StrangMarchuk
    fsplit_sm = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))
    prob_sm = OperatorSplittingProblem(fsplit_sm, u0, tspan)
    tstepper_sm = StrangMarchuk((Euler(), Euler()))
    integrator_sm = DiffEqBase.init(prob_sm, tstepper_sm, dt = 0.01, verbose = false)
    step!(integrator_sm)
    solve!(integrator_sm)
end
