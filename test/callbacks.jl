using OrdinaryDiffEqOperatorSplitting
using Test

import SciMLBase
import SciMLBase: ReturnCode
import DiffEqBase: DiffEqBase, ODEFunction, DiscreteCallback, ContinuousCallback,
    VectorContinuousCallback, CallbackSet
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

const OS = OrdinaryDiffEqOperatorSplitting

# ---------------------------------------------------------------------------
# Reference problem. The third component is decoupled and solved by the first
# operator alone, so `u[3](t) = 3exp(-t/10)` exactly -- which gives continuous
# callback tests an analytic event time to compare against.
# ---------------------------------------------------------------------------
const U0 = [1.0, 2.0, 3.0]
const TSPAN = (0.0, 1.0)

f1 = ODEFunction((du, u, p, t) -> (@. du = -0.1u))
f2 = ODEFunction(
    function (du, u, p, t)
        du[1] = -0.01u[2]
        du[2] = -0.01u[1]
        return nothing
    end
)
fsplit = GenericSplitFunction((f1, f2), ([1, 2, 3], [1, 2]))

make_prob(tspan = TSPAN) = OperatorSplittingProblem(fsplit, copy(U0), tspan)
ltg() = LieTrotterGodunov((Euler(), Euler()))
# Accurate inner solvers, so that an event time's error is the interpolant's alone.
ltg_exact() = LieTrotterGodunov((Tsit5(), Tsit5()))

# Exact crossing time of u[3] = level.
exact_crossing(level) = -10 * log(level / 3)

@testset "discrete callbacks" begin
    prob = make_prob()

    @testset "condition is evaluated once per outer step" begin
        calls = Ref(0)
        cb = DiscreteCallback((u, t, integrator) -> (calls[] += 1; false), integrator -> nothing)
        DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb))
        @test calls[] == 10
    end

    @testset "nested splitting does not run callbacks per inner stage" begin
        inner = GenericSplitFunction((f1, f1), ([1, 2], [1, 2]))
        outer = GenericSplitFunction((f1, inner), ([1, 2, 3], [1, 2]))
        nprob = OperatorSplittingProblem(outer, copy(U0), TSPAN)
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        calls = Ref(0)
        cb = DiscreteCallback((u, t, integrator) -> (calls[] += 1; false), integrator -> nothing)
        DiffEqBase.solve!(DiffEqBase.init(nprob, alg; dt = 0.1, callback = cb))
        @test calls[] == 10   # outer steps only, whatever the inner node does
    end

    @testset "affect! fires and sees the integrator" begin
        seen = Float64[]
        cb = DiscreteCallback(
            (u, t, integrator) -> t >= 0.45,
            integrator -> push!(seen, integrator.t)
        )
        sol = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb))
        @test seen ≈ [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        @test sol.retcode == ReturnCode.Success
    end

    @testset "save_positions brackets the affect!" begin
        cb = DiscreteCallback(
            (u, t, integrator) -> isapprox(t, 0.5),
            integrator -> (integrator.u[1] += 10.0)
        )
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb, tstops = [0.5])
        )
        # Two points at the event time: the state before and after the affect!.
        idxs = findall(t -> isapprox(t, 0.5), sol.t)
        @test length(idxs) == 2
        @test sol.u[idxs[2]][1] - sol.u[idxs[1]][1] ≈ 10.0

        # save_positions = (false, false) records nothing extra.
        cb_quiet = DiscreteCallback(
            (u, t, integrator) -> isapprox(t, 0.5),
            integrator -> (integrator.u[1] += 10.0);
            save_positions = (false, false)
        )
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb_quiet, tstops = [0.5])
        )
        @test count(t -> isapprox(t, 0.5), sol.t) == 0
    end

    @testset "a modified u reaches the whole subintegrator tree" begin
        # Every child holds its own copy of its slice, and the palindromic schemes
        # skip the forward sync of their first child.
        for alg in (
                ltg(),
                StrangMarchuk((Euler(), Euler())),
                PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5())),
            )
            cb = DiscreteCallback(
                (u, t, integrator) -> isapprox(t, 0.5),
                integrator -> (integrator.u .= [100.0, 200.0, 300.0])
            )
            integrator = DiffEqBase.init(
                prob, alg; dt = 0.1, callback = cb, tstops = [0.5]
            )
            while integrator.t < 0.5
                DiffEqBase.step!(integrator)
            end
            @test integrator.u == [100.0, 200.0, 300.0]
            @test integrator.child_subintegrators[1].u == [100.0, 200.0, 300.0]
            @test integrator.child_subintegrators[2].u == [100.0, 200.0]
            OS.validate_time_point(integrator)
        end
    end

    @testset "a modified u actually changes the trajectory" begin
        plain = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, tstops = [0.5])
        )
        cb = DiscreteCallback(
            (u, t, integrator) -> isapprox(t, 0.5),
            integrator -> (integrator.u .*= 2)
        )
        bumped = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb, tstops = [0.5])
        )
        # Doubling halfway through must roughly double the endpoint of a linear ODE.
        @test bumped.u[end] ≈ 2 .* plain.u[end] rtol = 1.0e-8
    end

    @testset "multiple callbacks in a CallbackSet" begin
        # Thresholds sit between grid points, and times are compared with isapprox,
        # because of ulp drift (0.1 * 8 is 0.7999999999999999, not 0.8).
        early = Float64[]
        late = Float64[]
        cb1 = DiscreteCallback((u, t, integrator) -> t >= 0.35, integrator -> push!(early, integrator.t))
        cb2 = DiscreteCallback((u, t, integrator) -> t >= 0.85, integrator -> push!(late, integrator.t))
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, callback = CallbackSet(cb1, cb2))
        )
        @test early ≈ [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        @test late ≈ [0.9, 1.0]
        @test sol.retcode == ReturnCode.Success
    end

    @testset "sol.stats.ncondition counts condition evaluations" begin
        cb = DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
        sol = DiffEqBase.solve!(integrator)
        @test sol.stats.ncondition == 10
    end
end

@testset "terminate!" begin
    prob = make_prob()

    @testset "stops the solve and sets the retcode" begin
        cb = DiscreteCallback(
            (u, t, integrator) -> t >= 0.5,
            integrator -> SciMLBase.terminate!(integrator)
        )
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
        sol = DiffEqBase.solve!(integrator)
        @test sol.retcode == ReturnCode.Terminated
        @test sol.t[end] ≈ 0.5
        @test integrator.t ≈ 0.5
        @test SciMLBase.done(integrator)
    end

    @testset "accepts an explicit retcode" begin
        cb = DiscreteCallback(
            (u, t, integrator) -> t >= 0.5,
            integrator -> SciMLBase.terminate!(integrator, ReturnCode.Success)
        )
        sol = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb))
        @test sol.retcode == ReturnCode.Success
    end
end

@testset "continuous callbacks" begin
    prob = make_prob()
    level = 2.9
    exact = exact_crossing(level)

    @testset "locates the crossing and fires exactly once" begin
        hits = Float64[]
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - level,
            integrator -> push!(hits, integrator.t)
        )
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg_exact(); dt = 0.05, callback = cb)
        )
        @test length(hits) == 1
        @test hits[1] ≈ exact atol = 1.0e-3
        @test hits[1] in sol.t
        @test issorted(sol.t)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "event time converges under refinement" begin
        # The located root is at least second order accurate. The step grid moves
        # with dt, so the constant is noisy: assert only that a factor-of-four
        # refinement buys a factor of four.
        errs = map((0.1, 0.025)) do dt
            hits = Float64[]
            cb = ContinuousCallback(
                (u, t, integrator) -> u[3] - level,
                integrator -> push!(hits, integrator.t)
            )
            DiffEqBase.solve!(
                DiffEqBase.init(prob, ltg_exact(); dt = dt, callback = cb)
            )
            abs(hits[1] - exact)
        end
        @test errs[2] < errs[1] / 4
    end

    @testset "affect_neg! handles the downcrossing" begin
        ups = Ref(0)
        downs = Ref(0)
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - level,
            integrator -> (ups[] += 1);
            affect_neg! = integrator -> (downs[] += 1)
        )
        DiffEqBase.solve!(DiffEqBase.init(prob, ltg_exact(); dt = 0.05, callback = cb))
        # u[3] decays through the level, so this is a downcrossing only.
        @test ups[] == 0
        @test downs[] == 1
    end

    @testset "the tree is re-anchored to the event time" begin
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - level,
            integrator -> nothing
        )
        integrator = DiffEqBase.init(prob, ltg_exact(); dt = 0.05, callback = cb)
        while integrator.t < exact
            DiffEqBase.step!(integrator)
        end
        @test integrator.t ≈ exact atol = 1.0e-3
        OS.validate_time_point(integrator)
        @test integrator.child_subintegrators[1].u ≈ integrator.u
        @test integrator.child_subintegrators[2].u ≈ integrator.u[1:2]
    end

    @testset "an affect! that modifies u reaches the children" begin
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - level,
            integrator -> (integrator.u .= [7.0, 8.0, 9.0])
        )
        integrator = DiffEqBase.init(prob, ltg_exact(); dt = 0.05, callback = cb)
        while integrator.t < exact
            DiffEqBase.step!(integrator)
        end
        @test integrator.u == [7.0, 8.0, 9.0]
        @test integrator.child_subintegrators[1].u == [7.0, 8.0, 9.0]
        @test integrator.child_subintegrators[2].u == [7.0, 8.0]
    end

    @testset "NoRootFind fires at the step endpoint" begin
        hits = Float64[]
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - level,
            integrator -> push!(hits, integrator.t);
            rootfind = SciMLBase.NoRootFind
        )
        DiffEqBase.solve!(DiffEqBase.init(prob, ltg_exact(); dt = 0.1, callback = cb))
        @test length(hits) == 1
        # No root finding: the event is reported at the end of the bracketing step.
        @test hits[1] ≈ 0.4
    end

    @testset "works with an adaptive splitting" begin
        hits = Float64[]
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - level,
            integrator -> push!(hits, integrator.t)
        )
        # `dtmax` matters: this problem is so nearly linear that the controller
        # otherwise spans the rest of the tspan in one step, and the event is only
        # as accurate as the (linear) interpolant over the bracketing step.
        sol = DiffEqBase.solve!(
            DiffEqBase.init(
                prob, PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5()));
                dt = 0.05, dtmax = 0.05, callback = cb
            )
        )
        @test length(hits) == 1
        @test hits[1] ≈ exact atol = 1.0e-3
        @test sol.retcode == ReturnCode.Success
    end

    @testset "a long adaptive step degrades the event time" begin
        # An event is only as accurate as the interpolant over the step that brackets
        # it, so capping the step with dtmax improves it.
        function locate(; kwargs...)
            hits = Float64[]
            cb = ContinuousCallback(
                (u, t, integrator) -> u[3] - level,
                integrator -> push!(hits, integrator.t)
            )
            DiffEqBase.solve!(
                DiffEqBase.init(
                    prob, PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5()));
                    dt = 0.05, callback = cb, kwargs...
                )
            )
            return only(hits)
        end

        @test abs(locate(dtmax = 0.05) - exact) < abs(locate() - exact)
    end

    @testset "interp_points = 0 skips the safety sweep" begin
        calls = Ref(0)
        cb = ContinuousCallback(
            (u, t, integrator) -> (calls[] += 1; u[3] - level),
            integrator -> nothing;
            interp_points = 0
        )
        DiffEqBase.solve!(DiffEqBase.init(prob, ltg_exact(); dt = 0.25, callback = cb))
        sweeping = Ref(0)
        cb2 = ContinuousCallback(
            (u, t, integrator) -> (sweeping[] += 1; u[3] - level),
            integrator -> nothing
        )
        DiffEqBase.solve!(DiffEqBase.init(prob, ltg_exact(); dt = 0.25, callback = cb2))
        # The default `interp_points = 11` sweeps the interpolant for sign changes
        # the endpoints missed; with `0` the condition is only seen at the endpoints.
        @test calls[] < sweeping[]
    end
end

@testset "VectorContinuousCallback" begin
    prob = make_prob()
    levels = (2.9, 2.85)

    # Current DiffEqBase hands `affect!` a mask of simultaneous events rather than a
    # single index; accept either shape.
    fired_index(idx::Integer) = Int(idx)
    fired_index(mask) = only(findall(!iszero, mask))

    fired = Tuple{Int, Float64}[]
    cb = VectorContinuousCallback(
        function (out, u, t, integrator)
            out[1] = u[3] - levels[1]
            out[2] = u[3] - levels[2]
            return nothing
        end,
        (integrator, idx) -> push!(fired, (fired_index(idx), integrator.t)),
        2
    )
    integrator = DiffEqBase.init(prob, ltg_exact(); dt = 0.05, callback = cb)
    @test integrator.callback_cache !== nothing
    sol = DiffEqBase.solve!(integrator)

    @test length(fired) == 2
    @test first.(fired) == [1, 2]                       # the higher level is crossed first
    @test last(fired[1]) ≈ exact_crossing(levels[1]) atol = 1.0e-3
    @test last(fired[2]) ≈ exact_crossing(levels[2]) atol = 1.0e-3
    @test sol.retcode == ReturnCode.Success

    @testset "no cache is allocated without a vector callback" begin
        scalar_cb = ContinuousCallback((u, t, integrator) -> u[3] - 2.9, integrator -> nothing)
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = scalar_cb)
        @test integrator.callback_cache === nothing
    end
end

@testset "callback initialization" begin
    prob = make_prob()

    @testset "an initializer can add tstops" begin
        # The pattern PresetTimeCallback uses.
        hits = Float64[]
        cb = DiscreteCallback(
            (u, t, integrator) -> t in (0.35, 0.65),
            integrator -> push!(hits, integrator.t);
            initialize = function (c, u, t, integrator)
                DiffEqBase.add_tstop!(integrator, 0.35)
                DiffEqBase.add_tstop!(integrator, 0.65)
                return SciMLBase.derivative_discontinuity!(integrator, false)
            end
        )
        sol = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb))
        @test hits == [0.35, 0.65]
        @test sol.retcode == ReturnCode.Success
    end

    @testset "an initializer that modifies u is reflected everywhere" begin
        cb = DiscreteCallback(
            (u, t, integrator) -> false,
            integrator -> nothing;
            initialize = function (c, u, t, integrator)
                integrator.u .= [5.0, 6.0, 7.0]
                return SciMLBase.derivative_discontinuity!(integrator, true)
            end
        )
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
        @test integrator.u == [5.0, 6.0, 7.0]
        @test integrator.uprev == [5.0, 6.0, 7.0]
        @test integrator.child_subintegrators[1].u == [5.0, 6.0, 7.0]
        @test integrator.child_subintegrators[2].u == [5.0, 6.0]
        # The first saved point is the modified state, not the problem's u0.
        @test integrator.sol.u[1] == [5.0, 6.0, 7.0]
    end

    @testset "finalize runs exactly once" begin
        finals = Ref(0)
        cb = DiscreteCallback(
            (u, t, integrator) -> false,
            integrator -> nothing;
            finalize = (c, u, t, integrator) -> (finals[] += 1)
        )
        DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb))
        @test finals[] == 1
    end

    @testset "reinit! keeps an initializer's modification" begin
        # The child reinit must not undo what the initializer did, so the callbacks
        # have to be initialized after the subintegrator tree is restored.
        cb = DiscreteCallback(
            (u, t, integrator) -> false,
            integrator -> nothing;
            initialize = function (c, u, t, integrator)
                integrator.u .= [5.0, 6.0, 7.0]
                return SciMLBase.derivative_discontinuity!(integrator, true)
            end
        )
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
        DiffEqBase.solve!(integrator)
        DiffEqBase.reinit!(integrator)
        @test integrator.u == [5.0, 6.0, 7.0]
        @test integrator.uprev == [5.0, 6.0, 7.0]
        @test integrator.child_subintegrators[1].u == [5.0, 6.0, 7.0]
        @test integrator.child_subintegrators[2].u == [5.0, 6.0]
        @test integrator.sol.u[1] == [5.0, 6.0, 7.0]
    end

    @testset "reinit! re-runs the initializers" begin
        inits = Ref(0)
        cb = DiscreteCallback(
            (u, t, integrator) -> false,
            integrator -> nothing;
            initialize = function (c, u, t, integrator)
                inits[] += 1
                return SciMLBase.derivative_discontinuity!(integrator, false)
            end
        )
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
        @test inits[] == 1
        DiffEqBase.solve!(integrator)
        DiffEqBase.reinit!(integrator)
        @test inits[] == 2
        DiffEqBase.reinit!(integrator; reinit_callbacks = false)
        @test inits[] == 2
        @test DiffEqBase.solve!(integrator).retcode == ReturnCode.Success
    end
end

@testset "callbacks combined with saving" begin
    prob = make_prob()

    @testset "saveat and an event coexist" begin
        cb = ContinuousCallback(
            (u, t, integrator) -> u[3] - 2.9,
            integrator -> nothing
        )
        sol = DiffEqBase.solve!(
            DiffEqBase.init(
                prob, ltg_exact(); dt = 0.1, saveat = [0.2, 0.8], callback = cb
            )
        )
        @test issorted(sol.t)
        @test 0.2 in sol.t
        @test 0.8 in sol.t
        @test sol.t[1] == 0.0
        @test sol.t[end] == 1.0
    end

    @testset "a discrete callback still leaves a usable interpolation" begin
        cb = DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
        sol = DiffEqBase.solve!(
            DiffEqBase.init(
                prob, ltg(); dt = 0.1, callback = cb, save_everystep = true
            )
        )
        @test sol(0.35) ≈ (sol.u[4] .+ sol.u[5]) ./ 2
    end
end
