using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase: ReturnCode
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

# ---------------------------------------------------------------------------
# A nested splitting function:  f = (f1, (f3, f3))
# ---------------------------------------------------------------------------
ode1(du, u, p, t) = @. du = -0.1u
function ode3(du, u, p, t)
    du[1] = -0.005u[2]
    return du[2] = -0.005u[1]
end
f1 = ODEFunction(ode1)
f3 = ODEFunction(ode3)

f_inner = GenericSplitFunction((f3, f3), ([1, 2], [1, 2]))
f_flat = GenericSplitFunction((f1, f3), ([1, 2, 3], [1, 3]))
f_nested = GenericSplitFunction((f1, f_inner), ([1, 2, 3], [1, 3]))

@testset "SplitNode" begin
    @testset "minting" begin
        @test f_nested[].path === ()
        @test f_nested[].object === f_nested
        @test f_nested[1].path === (1,)
        @test f_nested[1].object === f1
        @test f_nested[2, 1].path === (2, 1)
        @test f_nested[2, 1].object === f3

        # stepwise descent and varargs descent agree
        @test f_nested[2][1].path === f_nested[2, 1].path
        @test f_nested[][2, 1].path === f_nested[2, 1].path
    end

    @testset "invalid addresses" begin
        @test_throws ArgumentError f_nested[3]           # out of range
        @test_throws ArgumentError f_nested[0]
        @test_throws ArgumentError f_nested[1, 1]        # descends into a leaf
        @test_throws ArgumentError f_nested[2, 3]
    end

    @testset "resolution against the mirroring trees" begin
        alg = LieTrotterGodunov((Euler(), StrangMarchuk((Tsit5(), Euler()))))

        @test f_nested[f_nested[2, 1]] === f3
        @test f_nested[f_nested[2]] === f_inner
        @test f_nested[f_nested[]] === f_nested
        @test OS.get_operator(f_nested, f_nested[2, 1]) === f3

        @test alg[f_nested[1]] === alg.inner_algs[1]
        @test alg[f_nested[2]] === alg.inner_algs[2]
        @test alg[f_nested[2, 1]] === alg.inner_algs[2].inner_algs[1]
        @test alg[2, 1] === alg.inner_algs[2].inner_algs[1]

        u0 = [0.7, 0.9, 0.5]
        prob = OperatorSplittingProblem(f_nested, u0, (0.0, 0.1))
        integrator = DiffEqBase.init(prob, alg; dt = 0.01, adaptive = false)

        @test integrator[f_nested[]] === integrator
        @test integrator[f_nested[1]] === integrator.child_subintegrators[1]
        @test integrator[f_nested[2]] === integrator.child_subintegrators[2]
        @test integrator[f_nested[2, 1]] ===
            integrator.child_subintegrators[2].child_subintegrators[1]

        # SciMLBase's symbolic indexing must keep working: integrator[i] is the
        # i-th state component, not the i-th subintegrator.
        @test integrator[1] == integrator.u[1]
    end

    @test sprint(show, f_nested[2, 1]) == "f[2, 1]"
end

@testset "TreeOption" begin
    @testset "construction mirrors the function tree" begin
        opt = TreeOption(f_nested, 1.0e-2)
        @test opt[] == 1.0e-2
        @test opt[1] == 1.0e-2
        @test opt[2] == 1.0e-2
        @test opt[2, 1] == 1.0e-2
        @test opt[2, 2] == 1.0e-2
        @test opt isa TreeOption{Float64}

        @test OS.structure_matches(opt, f_nested)
        @test !OS.structure_matches(opt, f_flat)
        @test OS.structure_matches(TreeOption(f_flat, 1.0e-2), f_flat)
    end

    @testset "plain assignment writes a single node" begin
        opt = TreeOption(f_nested, 1.0e-2)
        opt[2] = 1.0e-4
        @test opt[] == 1.0e-2
        @test opt[1] == 1.0e-2
        @test opt[2] == 1.0e-4
        @test opt[2, 1] == 1.0e-2      # untouched
        @test opt[2, 2] == 1.0e-2

        opt[2, 1] = 1.0e-5
        @test opt[2, 1] == 1.0e-5
        @test opt[2, 2] == 1.0e-2

        opt[] = 0.5
        @test opt[] == 0.5
        @test opt[1] == 1.0e-2
    end

    @testset "addressing by SplitNode" begin
        opt = TreeOption(f_nested, 1.0e-2)
        opt[f_nested[2, 1]] = 1.0e-5
        @test opt[2, 1] == 1.0e-5
        @test opt[f_nested[2, 1]] == 1.0e-5

        opt[f_nested[2]] .= 3.0e-4
        @test opt[2] == 3.0e-4
        @test opt[2, 1] == 3.0e-4
        @test opt[2, 2] == 3.0e-4
        @test opt[1] == 1.0e-2
    end

    @testset "broadcast assignment writes the subtree" begin
        opt = TreeOption(f_nested, 1.0e-2)
        opt[2] .= 1.0e-4
        @test opt[] == 1.0e-2
        @test opt[1] == 1.0e-2
        @test opt[2] == 1.0e-4
        @test opt[2, 1] == 1.0e-4
        @test opt[2, 2] == 1.0e-4

        # a more specific write after a subtree write survives
        opt[2, 1] = 1.0e-5
        @test opt[2, 1] == 1.0e-5
        @test opt[2, 2] == 1.0e-4

        # ... and a subtree write after it clobbers it again (order matters)
        opt[2] .= 2.0e-4
        @test opt[2, 1] == 2.0e-4

        opt .= 7.0
        @test opt[] == 7.0
        @test opt[1] == 7.0
        @test opt[2, 2] == 7.0
    end

    @testset "rejected broadcasts" begin
        opt = TreeOption(f_nested, 1.0e-2)

        # Update-assignment reads one node but writes a subtree; there is no
        # unsurprising reading of that, so it must not silently do something.
        @test_throws ArgumentError opt[2] .*= 2
        @test_throws ArgumentError opt[2] .+= 1
        @test_throws ArgumentError opt .*= 2
        @test_throws ArgumentError opt[2] .= [1.0, 2.0]
        @test_throws ArgumentError opt[2] .= (1.0, 2.0)

        # nothing was written by the rejected calls
        @test opt[2] == 1.0e-2
        @test opt[2, 1] == 1.0e-2

        # the undotted right hand side the error message suggests does work
        opt[2] .= 2 * opt[2]
        @test opt[2] == 2.0e-2
        @test opt[2, 1] == 2.0e-2
    end

    @testset "element type" begin
        opt = TreeOption(f_nested, true)
        @test opt isa TreeOption{Bool}
        opt[2] = false
        @test opt[2] === false
        @test_throws ArgumentError opt[2] = "yes"

        # widening conversions that Julia performs anyway are still fine
        num = TreeOption(f_nested, 1.0)
        num[1] = 2
        @test num[1] === 2.0

        # mixed types need the explicit form
        ctrl = TreeOption{Any}(f_nested, nothing)
        @test ctrl isa TreeOption{Any}
        @test ctrl[1] === nothing
        ctrl[1] = :something
        @test ctrl[1] === :something
        ctrl[2] .= nothing
        @test ctrl[2, 1] === nothing
    end

    @testset "invalid addresses" begin
        opt = TreeOption(f_nested, 1.0e-2)
        @test_throws ArgumentError opt[3]
        @test_throws ArgumentError opt[1, 1]       # [1] mirrors a leaf
        @test_throws ArgumentError opt[2, 3]
        @test_throws ArgumentError opt[3] = 1.0
        @test_throws ArgumentError opt[1, 1] .= 1.0
    end

    @testset "show" begin
        opt = TreeOption(f_nested, 1.0e-2)
        opt[2] .= 1.0e-4
        str = sprint(show, MIME"text/plain"(), opt)
        @test occursin("TreeOption{Float64}", str)
        @test occursin("[] => 0.01", str)
        @test occursin("[2, 1] => 0.0001", str)
    end
end

# ---------------------------------------------------------------------------
# Per-node settings reaching the integrator tree
#
# f1 + f3 + f3 == f1 + f2 is linear, so the exact solution is available for
# checking that a multi-rate configuration still integrates the right problem.
# ---------------------------------------------------------------------------
trueA = [-0.1 0.0 0.0; 0.0 -0.1 0.0; 0.0 0.0 -0.1]
trueB = [0.0 0.0 -0.01; 0.0 0.0 0.0; -0.01 0.0 0.0]
u0 = [0.7611944793397108, 0.9059606424982555, 0.5755174199139956]
tspan = (0.0, 1.0)
trueu = exp((tspan[2] - tspan[1]) * (trueA + trueB)) * u0

prob = OperatorSplittingProblem(f_nested, u0, tspan)
# A power of two, so that the subcycled step sizes below are exactly
# representable and the sub-step counts are deterministic.
dt_outer = 2.0^-7
nsteps = round(Int, (tspan[2] - tspan[1]) / dt_outer)

@testset "per-node configuration" begin
    @testset "uniform dt is unchanged" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        scalar = DiffEqBase.init(prob, alg; dt = dt_outer, adaptive = false)
        tree = DiffEqBase.init(
            prob, alg; dt = TreeOption(f_nested, dt_outer), adaptive = false
        )
        DiffEqBase.solve!(scalar)
        DiffEqBase.solve!(tree)
        @test scalar.u == tree.u
        @test scalar.iter == tree.iter
    end

    @testset "multi-rate: a subtree subcycles" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        dt = TreeOption(f_nested, dt_outer)
        dt[f_nested[2]] .= dt_outer / 4       # the nested split and its two leaves

        integrator = DiffEqBase.init(prob, alg; dt, adaptive = false)

        @test integrator.dt == dt_outer
        @test integrator[f_nested[1]].dt == dt_outer
        @test integrator[f_nested[2]].dt == dt_outer / 4
        @test integrator[f_nested[2, 1]].dt == dt_outer / 4

        DiffEqBase.solve!(integrator)
        @test integrator.sol.retcode == ReturnCode.Success
        @test integrator.t ≈ tspan[2]
        @test integrator.iter == nsteps

        # The outer integrator hands the nested node an interval of dt_outer, which
        # it covers in four sub-steps -- and its own children follow along.
        @test integrator[f_nested[1]].iter == nsteps
        @test integrator[f_nested[2]].iter == 4 * nsteps
        @test integrator[f_nested[2, 1]].iter == 4 * nsteps

        # Everything still lands on the same time points.
        @test integrator[f_nested[1]].t ≈ tspan[2]
        @test integrator[f_nested[2]].t ≈ tspan[2]
        @test integrator[f_nested[2, 1]].t ≈ tspan[2]

        @test isapprox(integrator.u, trueu, atol = 1.0e-4)
    end

    @testset "multi-rate: a single leaf subcycles" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        dt = TreeOption(f_nested, dt_outer)
        dt[2, 1] = dt_outer / 8               # this leaf only

        integrator = DiffEqBase.init(prob, alg; dt, adaptive = false)
        DiffEqBase.solve!(integrator)

        @test integrator.iter == nsteps
        @test integrator[f_nested[2]].iter == nsteps
        @test integrator[f_nested[2, 1]].iter == 8 * nsteps
        @test integrator[f_nested[2, 2]].iter == nsteps
        @test isapprox(integrator.u, trueu, atol = 1.0e-4)
    end

    @testset "a sub-dt that does not divide the interval still lands on it" begin
        # 1/3 of the outer step is not representable, so the leaf covers the last
        # sliver of each interval with an extra ulp-sized step. That costs steps but
        # must not cost accuracy or leave the tree out of sync.
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        dt = TreeOption(f_nested, dt_outer)
        dt[f_nested[2]] .= dt_outer / 3

        integrator = DiffEqBase.init(prob, alg; dt, adaptive = false)
        DiffEqBase.solve!(integrator)

        @test integrator.sol.retcode == ReturnCode.Success
        @test integrator.iter == nsteps
        @test integrator.t ≈ tspan[2]
        @test integrator[f_nested[2]].t ≈ tspan[2]
        @test integrator[f_nested[2, 1]].t ≈ tspan[2]
        @test integrator[f_nested[2]].iter ≥ 3 * nsteps
        @test isapprox(integrator.u, trueu, atol = 1.0e-4)
    end

    @testset "mixed adaptivity" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Tsit5(), Tsit5()))))
        adaptive = TreeOption(f_nested, false)
        adaptive[2, 1] = true                 # leaves only, the splitting nodes
        adaptive[2, 2] = true                 # stay non-adaptive

        integrator = DiffEqBase.init(prob, alg; dt = dt_outer, adaptive)

        @test integrator.opts.adaptive == false
        @test integrator[f_nested[1]].opts.adaptive == false
        @test integrator[f_nested[2]].opts.adaptive == false
        @test integrator[f_nested[2, 1]].opts.adaptive == true
        @test integrator[f_nested[2, 2]].opts.adaptive == true

        DiffEqBase.solve!(integrator)
        @test integrator.sol.retcode == ReturnCode.Success
        @test integrator.t ≈ tspan[2]
        @test integrator.iter == nsteps       # the splitting step is untouched
        @test isapprox(integrator.u, trueu, atol = 1.0e-4)
    end

    @testset "inner integrator options travel to the leaves" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Tsit5(), Tsit5()))))
        adaptive = TreeOption(f_nested, false)
        adaptive[2, 1] = true
        adaptive[2, 2] = true
        reltol = TreeOption(f_nested, 1.0e-3)
        reltol[2, 1] = 1.0e-9

        integrator = DiffEqBase.init(
            prob, alg; dt = dt_outer, adaptive, reltol, dtmin = 1.0e-12
        )

        # `reltol` means nothing to a splitting node, so it is handed to the leaves
        @test integrator[f_nested[2, 1]].opts.reltol == 1.0e-9
        @test integrator[f_nested[2, 2]].opts.reltol == 1.0e-3
        # ... while `dtmin` is understood at every level
        @test integrator.opts.dtmin == 1.0e-12
        @test integrator[f_nested[2]].opts.dtmin == 1.0e-12

        DiffEqBase.solve!(integrator)
        @test integrator.sol.retcode == ReturnCode.Success
        @test isapprox(integrator.u, trueu, atol = 1.0e-4)
    end

    @testset "asking a non-adaptive splitting node to be adaptive warns" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        adaptive = TreeOption(f_nested, false)
        adaptive[2] = true                    # a splitting node, and LTG is not adaptive
        @test_logs (:warn, r"operator \[2\] is not adaptive") DiffEqBase.init(
            prob, alg; dt = dt_outer, adaptive
        )
    end

    @testset "rejected configurations" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))

        # built for a differently shaped splitting function
        wrong = TreeOption(f_flat, dt_outer)
        @test_throws ArgumentError DiffEqBase.init(prob, alg; dt = wrong)

        negative = TreeOption(f_nested, dt_outer)
        negative[2, 1] = -1.0e-3
        @test_throws ErrorException DiffEqBase.init(prob, alg; dt = negative)
    end

    @testset "reinit! keeps the per-node configuration" begin
        alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
        dt = TreeOption(f_nested, dt_outer)
        dt[f_nested[2]] .= dt_outer / 4

        integrator = DiffEqBase.init(prob, alg; dt, adaptive = false)
        DiffEqBase.solve!(integrator)
        ufinal = copy(integrator.u)

        DiffEqBase.reinit!(integrator)
        @test integrator.dt == dt_outer
        @test integrator[f_nested[2]].dt == dt_outer / 4
        @test integrator[f_nested[2, 1]].dt == dt_outer / 4
        DiffEqBase.solve!(integrator)
        @test integrator.u ≈ ufinal
        @test integrator[f_nested[2]].iter == 4 * nsteps

        # An explicit scalar reconfigures the whole tree, as it would at `init`.
        DiffEqBase.reinit!(integrator; dt = dt_outer)
        @test integrator.dt == dt_outer
        @test integrator[f_nested[2]].dt == dt_outer
        @test integrator[f_nested[2, 1]].dt == dt_outer
    end
end
