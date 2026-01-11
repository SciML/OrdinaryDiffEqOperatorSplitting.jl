using PrecompileTools: @compile_workload
using OrdinaryDiffEqLowOrderRK: Euler

function _precompile_ode1(du, u, p, t)
    return @. du = -0.1u
end

function _precompile_ode2(du, u, p, t)
    du[1] = -0.01u[2]
    return du[2] = -0.01u[1]
end

@compile_workload begin
    # Setup minimal test problem for precompilation
    tspan = (0.0, 0.1)
    u0 = [1.0, 1.0, 1.0]

    f1 = DiffEqBase.ODEFunction(_precompile_ode1)
    f2 = DiffEqBase.ODEFunction(_precompile_ode2)

    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    fsplit = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))

    prob = OperatorSplittingProblem(fsplit, u0, tspan)
    tstepper = LieTrotterGodunov((Euler(), Euler()))

    # Precompile init and a few steps
    integrator = DiffEqBase.init(prob, tstepper, dt = 0.01, verbose = false)
    DiffEqBase.step!(integrator)
    DiffEqBase.solve!(integrator)
end
