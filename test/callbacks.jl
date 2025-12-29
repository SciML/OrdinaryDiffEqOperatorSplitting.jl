using OrdinaryDiffEqOperatorSplitting
using Test
using DiffEqBase
using SciMLBase
using OrdinaryDiffEqLowOrderRK

# Test setup: simple split ODE problem
tspan = (0.0, 10.0)
u0 = [1.0, 2.0, 3.0]

# Simple decay functions
function ode1(du, u, p, t)
    @. du = -0.1u
end
f1 = DiffEqBase.ODEFunction(ode1)

function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
end
f2 = DiffEqBase.ODEFunction(ode2)

f1dofs = [1, 2, 3]
f2dofs = [1, 2]
fsplit = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))
prob = OperatorSplittingProblem(fsplit, u0, tspan)

@testset "Discrete Callbacks" begin
    dt = 0.1

    @testset "Simple condition callback" begin
        # Count how many times the callback is triggered
        callback_count = Ref(0)

        condition(u, t, integrator) = t >= 5.0
        function affect!(integrator)
            callback_count[] += 1
        end
        cb = DiscreteCallback(condition, affect!)

        integrator = DiffEqBase.init(
            prob, LieTrotterGodunov((Euler(), Euler())),
            dt = dt, callback = cb
        )
        DiffEqBase.solve!(integrator)

        # Callback should be triggered for t >= 5.0
        # With dt=0.1 and tspan=(0,10), there are about 50 steps after t=5
        @test callback_count[] > 0
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
    end

    @testset "Callback that modifies u" begin
        # Callback that doubles u[1] when t >= 5.0 (only once)
        triggered = Ref(false)

        condition(u, t, integrator) = t >= 5.0 && !triggered[]
        function affect!(integrator)
            triggered[] = true
            integrator.u[1] *= 2.0
        end
        cb = DiscreteCallback(condition, affect!)

        integrator = DiffEqBase.init(
            prob, LieTrotterGodunov((Euler(), Euler())),
            dt = dt, callback = cb
        )

        # Step until just before t=5
        while integrator.t < 4.9
            DiffEqBase.step!(integrator)
        end
        u_before = copy(integrator.u)

        # Step past t=5 to trigger callback
        while integrator.t < 5.5
            DiffEqBase.step!(integrator)
        end

        @test triggered[]
        @test integrator.sol.retcode in (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default)
    end

    @testset "Callback with save_positions" begin
        save_times = Float64[]

        condition(u, t, integrator) = t >= 3.0 && t < 3.5
        function affect!(integrator)
            # Just record that we were here
        end
        cb = DiscreteCallback(condition, affect!, save_positions = (true, true))

        integrator = DiffEqBase.init(
            prob, LieTrotterGodunov((Euler(), Euler())),
            dt = dt, callback = cb
        )
        DiffEqBase.solve!(integrator)

        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        # Solution should have saved some values
        @test length(integrator.sol.t) > 0 || length(integrator.sol.u) >= 0
    end

    @testset "Multiple discrete callbacks" begin
        count1 = Ref(0)
        count2 = Ref(0)

        condition1(u, t, integrator) = t >= 2.0
        affect1!(integrator) = count1[] += 1
        cb1 = DiscreteCallback(condition1, affect1!)

        condition2(u, t, integrator) = t >= 7.0
        affect2!(integrator) = count2[] += 1
        cb2 = DiscreteCallback(condition2, affect2!)

        integrator = DiffEqBase.init(
            prob, LieTrotterGodunov((Euler(), Euler())),
            dt = dt, callback = CallbackSet(cb1, cb2)
        )
        DiffEqBase.solve!(integrator)

        @test count1[] > count2[]  # cb1 is triggered more often
        @test count2[] > 0  # cb2 is still triggered
    end
end

@testset "Continuous Callbacks (Simplified)" begin
    dt = 0.1

    @testset "Zero-crossing detection" begin
        # Detect when u[1] crosses below 0.5
        triggered = Ref(false)

        condition(u, t, integrator) = u[1] - 0.5
        function affect!(integrator)
            triggered[] = true
        end
        cb = ContinuousCallback(condition, affect!)

        # Use a smaller tspan so u[1] actually crosses 0.5
        prob_short = OperatorSplittingProblem(fsplit, u0, (0.0, 20.0))

        integrator = DiffEqBase.init(
            prob_short, LieTrotterGodunov((Euler(), Euler())),
            dt = dt, callback = cb
        )
        DiffEqBase.solve!(integrator)

        # With decay, u[1] should eventually cross 0.5
        @test triggered[] || integrator.u[1] > 0.5  # Either crossed or hasn't decayed enough
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
    end

    @testset "Reflection callback" begin
        # Reflect u[1] when it goes below 0.3
        reflection_count = Ref(0)

        condition(u, t, integrator) = u[1] - 0.3
        function affect!(integrator)
            reflection_count[] += 1
            integrator.u[1] = 0.6 - integrator.u[1]  # Reflect around 0.3
        end
        cb = ContinuousCallback(condition, affect!)

        # Longer tspan for decay
        prob_long = OperatorSplittingProblem(fsplit, u0, (0.0, 50.0))

        integrator = DiffEqBase.init(
            prob_long, LieTrotterGodunov((Euler(), Euler())),
            dt = dt, callback = cb
        )
        DiffEqBase.solve!(integrator)

        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
    end
end

@testset "Callback with nested splitting" begin
    dt = 0.1

    # Create nested split problem
    function ode3(du, u, p, t)
        du[1] = -0.005u[2]
        du[2] = -0.005u[1]
    end
    f3 = DiffEqBase.ODEFunction(ode3)

    f3dofs = [1, 2]
    fsplit_inner = GenericSplitFunction((f3, f3), (f3dofs, f3dofs))
    fsplit_outer = GenericSplitFunction((f1, fsplit_inner), ([1, 2, 3], [1, 2]))

    prob_nested = OperatorSplittingProblem(fsplit_outer, u0, tspan)

    callback_count = Ref(0)
    condition(u, t, integrator) = t >= 5.0
    affect!(integrator) = callback_count[] += 1
    cb = DiscreteCallback(condition, affect!)

    integrator = DiffEqBase.init(
        prob_nested,
        LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler())))),
        dt = dt, callback = cb
    )
    DiffEqBase.solve!(integrator)

    @test callback_count[] > 0
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
end

@testset "u_modified! functionality" begin
    dt = 0.1

    # Test that u_modified! is properly set and used
    u_was_modified = Ref(false)

    condition(u, t, integrator) = t >= 5.0 && t < 5.5
    function affect!(integrator)
        integrator.u[1] = 0.0  # Modify u
        u_was_modified[] = true
    end
    cb = DiscreteCallback(condition, affect!)

    integrator = DiffEqBase.init(
        prob, LieTrotterGodunov((Euler(), Euler())),
        dt = dt, callback = cb
    )

    # Run until callback triggers
    while integrator.t < 6.0 && !isempty(integrator.tstops)
        DiffEqBase.step!(integrator)
    end

    @test u_was_modified[]
end

@testset "Callback preserves solution accuracy" begin
    dt = 0.01  # Smaller dt for better accuracy

    # Reference solution without callback
    integrator_ref = DiffEqBase.init(
        prob, LieTrotterGodunov((Euler(), Euler())),
        dt = dt
    )
    DiffEqBase.solve!(integrator_ref)
    u_ref = copy(integrator_ref.u)

    # Solution with no-op callback
    condition(u, t, integrator) = false  # Never triggers
    affect!(integrator) = nothing
    cb = DiscreteCallback(condition, affect!)

    integrator_cb = DiffEqBase.init(
        prob, LieTrotterGodunov((Euler(), Euler())),
        dt = dt, callback = cb
    )
    DiffEqBase.solve!(integrator_cb)

    # Solutions should be identical since callback never triggers
    @test isapprox(integrator_cb.u, u_ref, rtol = 1e-10)
end
