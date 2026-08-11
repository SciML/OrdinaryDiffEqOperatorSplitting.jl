using OrdinaryDiffEqOperatorSplitting
using Test

import SciMLBase
import SciMLBase: ReturnCode
import DiffEqBase: DiffEqBase, ODEFunction, DiscreteCallback, CallbackSet
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

const OS = OrdinaryDiffEqOperatorSplitting

# ---------------------------------------------------------------------------
# Reference problem: two linear operators sharing the first two components.
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

# Swept by every test that is not about one particular scheme.
algs() = (
    LieTrotterGodunov((Tsit5(), Tsit5())),
    StrangMarchuk((Tsit5(), Tsit5())),
    PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5())),
    Ruth3((Tsit5(), Tsit5())),
    AdjointPair(Ruth3((Tsit5(), Tsit5()))),
)

function exact(t)
    p = 1.5 * exp(-0.11t)     # (1, 1) mode
    m = -0.5 * exp(-0.09t)    # (1, -1) mode
    return [p + m, p - m, 3 * exp(-0.1t)]
end
rhs(u) = [-0.1u[1] - 0.01u[2], -0.1u[2] - 0.01u[1], -0.1u[3]]

@testset "saved time points" begin
    prob = make_prob()

    @testset "default saves only the interval endpoints" begin
        sol = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1))
        @test sol.t == [0.0, 1.0]
        @test length(sol.u) == 2
        @test sol.u[1] == U0
        @test sol.retcode == ReturnCode.Success
    end

    @testset "save_everystep saves every accepted step" begin
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, save_everystep = true)
        )
        @test length(sol.t) == 11          # t0 plus ten steps of dt = 0.1
        @test sol.t[1] == 0.0
        @test sol.t[end] == 1.0
        @test issorted(sol.t)
        # No duplicate of the final point from `save_end`.
        @test allunique(sol.t)
    end

    @testset "saveat vector" begin
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = [0.25, 0.5])
        )
        @test sol.t == [0.0, 0.25, 0.5, 1.0]
    end

    @testset "saveat number excludes t0 and does not duplicate tf" begin
        sol = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = 0.25))
        @test sol.t == [0.0, 0.25, 0.5, 0.75, 1.0]
        # A step that does not divide the span evenly leaves tf to `save_end`.
        sol = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = 0.3))
        @test sol.t ≈ [0.0, 0.3, 0.6, 0.9, 1.0]
    end

    @testset "saveat points outside the tspan are dropped" begin
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = [-1.0, 0.5, 7.0])
        )
        @test sol.t == [0.0, 0.5, 1.0]
    end

    @testset "save_start / save_end / save_on" begin
        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = [0.5], save_start = false)
        )
        @test sol.t == [0.5, 1.0]

        sol = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = [0.5], save_end = false)
        )
        @test sol.t == [0.0, 0.5]

        sol = DiffEqBase.solve!(
            DiffEqBase.init(
                prob, ltg(); dt = 0.1, saveat = [0.5],
                save_on = false, save_everystep = true
            )
        )
        @test isempty(sol.t)
        @test isempty(sol.u)
    end

    @testset "backward integration keeps the direction ordering" begin
        bprob = make_prob((1.0, 0.0))
        sol = DiffEqBase.solve!(
            DiffEqBase.init(bprob, ltg(); dt = 0.1, saveat = [0.25, 0.75])
        )
        @test sol.t == [1.0, 0.75, 0.25, 0.0]

        sol = DiffEqBase.solve!(DiffEqBase.init(bprob, ltg(); dt = 0.1, saveat = 0.25))
        @test sol.t == [1.0, 0.75, 0.5, 0.25, 0.0]
    end

    @testset "asking for output does not change the trajectory" begin
        # saveat points are interpolated, never stepped onto.
        plain = DiffEqBase.solve!(DiffEqBase.init(prob, ltg(); dt = 0.1))
        dense = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = 0.017, save_everystep = true)
        )
        @test dense.u[end] == plain.u[end]
    end

    @testset "saveat times passed as tstops are landed on, not interpolated" begin
        # The documented way to get output accurate to the order of the scheme.
        ts = [0.25, 0.55]
        landed = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, tstops = ts, saveat = ts)
        )
        interpolated = DiffEqBase.solve!(
            DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = ts)
        )
        @test issorted(landed.t)
        @test allunique(landed.t)
        for t in ts
            @test t in landed.t
            @test t in interpolated.t
        end
        # Same times, different values: stepped state vs. interpolated one.
        for t in ts
            a = landed.u[findfirst(==(t), landed.t)]
            b = interpolated.u[findfirst(==(t), interpolated.t)]
            @test a != b
            # ... but both approximate the same solution.
            @test isapprox(a, b; rtol = 1.0e-3)
        end
    end

    @testset "add_saveat!" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.step!(integrator)
        DiffEqBase.add_saveat!(integrator, 0.65)
        sol = DiffEqBase.solve!(integrator)
        @test sol.t == [0.0, 0.65, 1.0]
        # Cannot save behind the current time.
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.step!(integrator)
        DiffEqBase.step!(integrator)
        @test_throws ErrorException DiffEqBase.add_saveat!(integrator, 0.05)
    end

    @testset "other algorithms" begin
        for alg in algs()
            sol = DiffEqBase.solve!(
                DiffEqBase.init(prob, alg; dt = 0.1, saveat = [0.5])
            )
            @test sol.t[1] == 0.0
            @test sol.t[end] == 1.0
            @test 0.5 in sol.t
            @test sol.retcode == ReturnCode.Success
        end
    end

    @testset "nested splitting saves only at the outer level" begin
        inner = GenericSplitFunction((f1, f1), ([1, 2], [1, 2]))
        outer = GenericSplitFunction((f1, inner), ([1, 2, 3], [1, 2]))
        nprob = OperatorSplittingProblem(outer, copy(U0), TSPAN)
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        integrator = DiffEqBase.init(nprob, alg; dt = 0.1, saveat = [0.5])
        sol = DiffEqBase.solve!(integrator)
        @test sol.t == [0.0, 0.5, 1.0]
        # The inner node owns no solution storage of its own.
        @test !hasproperty(integrator.child_subintegrators[2], :saveiter)
    end
end

@testset "interpolation" begin
    prob = make_prob()

    @testset "$(nameof(typeof(alg)))" for alg in algs()
        integrator = DiffEqBase.init(prob, alg; dt = 0.1)
        DiffEqBase.step!(integrator)
        DiffEqBase.step!(integrator)
        (; tprev, t) = integrator
        tmid = (tprev + t) / 2

        @test integrator(tprev) ≈ integrator.uprev
        @test integrator(t) ≈ integrator.u
        @test integrator(tmid) ≈ exact(tmid) rtol = 1.0e-2
        @test integrator(tmid, Val{1}) ≈ rhs(exact(tmid)) rtol = 1.0e-2
        # For an adaptive scheme `dt` is already the next proposal, so the endpoints
        # above are only reproduced if the interpolant does not key off it.
        SciMLBase.isadaptive(alg) && @test integrator.dt != t - tprev

        out = similar(integrator.u)
        integrator(out, tmid)
        @test out ≈ integrator(tmid)
        @test integrator(tmid, Val{0}; idxs = 2) ≈ integrator(tmid)[2]
        sub = similar(integrator.u, 2)
        integrator(sub, tmid, Val{0}; idxs = [1, 3])
        @test sub ≈ integrator(tmid)[[1, 3]]

        sol = DiffEqBase.solve!(
            DiffEqBase.init(
                prob, alg; dt = 0.1, saveat = [0.05], save_everystep = true
            )
        )
        @test sol.u[findfirst(==(0.05), sol.t)] ≈ exact(0.05) rtol = 1.0e-2
        for τ in (0.35, 0.62, 0.97)
            @test sol(τ) ≈ exact(τ) rtol = 1.0e-2
        end
    end
end

@testset "change_t_via_interpolation!" begin
    prob = make_prob()

    @testset "moves the whole tree back to the interpolated state" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.step!(integrator)
        (; tprev, t) = integrator
        tmid = (tprev + t) / 2
        expected = integrator(tmid)

        SciMLBase.change_t_via_interpolation!(integrator, tmid)
        @test integrator.t == tmid
        @test integrator.u ≈ expected
        OS.validate_time_point(integrator)
        for child in integrator.child_subintegrators
            @test child.t == tmid
        end
        @test integrator.child_subintegrators[1].u ≈ expected[1:3]
        @test integrator.child_subintegrators[2].u ≈ expected[1:2]
    end

    @testset "refuses to move before tprev" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.step!(integrator)
        DiffEqBase.step!(integrator)
        @test_throws ErrorException SciMLBase.change_t_via_interpolation!(
            integrator, integrator.tprev - 0.01
        )
    end

    @testset "moving to the current t is a no-op" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.step!(integrator)
        u_before = copy(integrator.u)
        SciMLBase.change_t_via_interpolation!(integrator, integrator.t)
        @test integrator.u == u_before
    end

    @testset "accepts the Val{:false} the callback path passes" begin
        # A `Val` of the *Symbol*, so the flag has to be decoded by dispatch.
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.step!(integrator)
        tmid = (integrator.tprev + integrator.t) / 2
        SciMLBase.change_t_via_interpolation!(integrator, tmid, Val{:false}, nothing)
        @test integrator.t == tmid
    end

    @testset "modify_save_endpoint rewrites the saved endpoint" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, save_everystep = true)
        DiffEqBase.step!(integrator)
        nsaved = integrator.saveiter
        tmid = (integrator.tprev + integrator.t) / 2
        SciMLBase.change_t_via_interpolation!(integrator, tmid, Val{true})
        @test integrator.saveiter == nsaved + 1
        @test integrator.sol.t[integrator.saveiter] == tmid
    end
end

@testset "solution object plumbing" begin
    prob = make_prob()

    @testset "dense is accepted and does not reach the leaves" begin
        integrator = DiffEqBase.init(
            prob, ltg(); dt = 0.1, dense = true, save_everystep = true
        )
        sol = DiffEqBase.solve!(integrator)
        @test sol.retcode == ReturnCode.Success
        @test sol(0.35) ≈ (sol.u[4] .+ sol.u[5]) ./ 2
        # `dense` must be swallowed by `__init`; a leaf handed `dense = true` would
        # store interpolation data for stages that are not solution points.
        @test integrator.child_subintegrators[1].sol.dense == false
    end

    @testset "sol.stats is a real DEStats" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        @test integrator.sol.stats !== nothing
        @test integrator.sol.stats.ncondition == 0
    end

    @testset "postamble! is idempotent" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        sol = DiffEqBase.solve!(integrator)
        n = length(sol.t)
        SciMLBase.postamble!(integrator)
        SciMLBase.postamble!(integrator)
        @test length(integrator.sol.t) == n
    end

    @testset "postamble! does not run once per step" begin
        finalized = Ref(0)
        cb = DiscreteCallback(
            (u, t, integrator) -> false,
            integrator -> nothing;
            finalize = (c, u, t, integrator) -> (finalized[] += 1)
        )
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
        DiffEqBase.solve!(integrator)
        @test finalized[] == 1
    end
end

@testset "reinit!" begin
    prob = make_prob()

    @testset "restores the saved solution" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, saveat = [0.5])
        first_run = copy(DiffEqBase.solve!(integrator).t)
        DiffEqBase.reinit!(integrator)
        @test integrator.saveiter == 1
        @test integrator.sol.t[1] == 0.0
        DiffEqBase.solve!(integrator)
        @test integrator.sol.t == first_run
    end

    @testset "reinit_callbacks = false works without a saving callback" begin
        # Regression: this used to index an empty `discrete_callbacks` tuple.
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1)
        DiffEqBase.solve!(integrator)
        DiffEqBase.reinit!(integrator; reinit_callbacks = false)
        @test integrator.t == 0.0
        @test integrator.saveiter == 1
        @test DiffEqBase.solve!(integrator).retcode == ReturnCode.Success
    end

    @testset "a new u0 reaches the whole tree" begin
        # Regression: a leaf's `reinit!` restores the `u0` slice captured when it was
        # built, and the palindromic schemes skip the forward sync of their first
        # child, so this used to integrate a stale child state.
        u0b = [10.0, 20.0, 30.0]
        probb = OperatorSplittingProblem(fsplit, copy(u0b), TSPAN)
        for alg in (
                ltg(),
                StrangMarchuk((Euler(), Euler())),
                StrangMarchuk((Tsit5(), Euler())),
                PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5())),
            )
            integrator = DiffEqBase.init(make_prob(), alg; dt = 0.1)
            DiffEqBase.solve!(integrator)
            DiffEqBase.reinit!(integrator, copy(u0b))
            @test integrator.child_subintegrators[1].u == u0b
            @test integrator.child_subintegrators[2].u == u0b[1:2]
            got = DiffEqBase.solve!(integrator).u[end]
            ref = DiffEqBase.solve!(DiffEqBase.init(probb, alg; dt = 0.1)).u[end]
            @test got ≈ ref rtol = 1.0e-12
        end
    end

    @testset "erase_sol clears the stored solution" begin
        integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, save_everystep = true)
        DiffEqBase.solve!(integrator)
        DiffEqBase.reinit!(integrator; erase_sol = true)
        @test integrator.saveiter == 1
        @test length(integrator.sol.t) == 1
    end
end

@testset "callbacks can be attached" begin
    # Regression: `init` used to throw as soon as any callback was passed.
    prob = make_prob()
    calls = Ref(0)
    cb = DiscreteCallback((u, t, integrator) -> false, integrator -> (calls[] += 1))
    integrator = DiffEqBase.init(prob, ltg(); dt = 0.1, callback = cb)
    @test integrator.derivative_discontinuity == false
    sol = DiffEqBase.solve!(integrator)
    @test sol.retcode == ReturnCode.Success
    @test calls[] == 0

    integrator = DiffEqBase.init(
        prob, ltg(); dt = 0.1,
        callback = CallbackSet(cb, cb)
    )
    @test DiffEqBase.solve!(integrator).retcode == ReturnCode.Success
end
