using OrdinaryDiffEqOperatorSplitting
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
import SciMLBase: ReturnCode
using OrdinaryDiffEqTsit5

# Same non-commuting pair as in test/adaptivity.jl. Integrating the exact solution
# at t = 1 backward to t = 0 has to recover u0.
M = [-1.0 0.5; 0.5 -2.0]
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)
fA = ODEFunction(odeA)
fB = ODEFunction(odeB)

dofs = [1, 2]
u0 = [1.0, 1.0]
uT = exp(M) * u0
fsplit = GenericSplitFunction((fA, fB), (dofs, dofs))

PPLTG = PalindromicPairLieTrotterGodunov

@testset "Backward-in-time integration" begin
    @testset "Fixed-step consistency | $(nameof(typeof(alg)))" for (alg, atol) in (
            (LieTrotterGodunov((Tsit5(), Tsit5())), 1.0e-2),
            (StrangMarchuk((Tsit5(), Tsit5())), 1.0e-4),
            (PPLTG((Tsit5(), Tsit5())), 1.0e-4),
        )
        prob = OperatorSplittingProblem(fsplit, copy(uT), (1.0, 0.0))
        integ = DiffEqBase.init(prob, alg; dt = 0.01, adaptive = false)
        @test integ.dt < 0
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.t ≈ 0.0
        @test integ.iter == 100
        @test maximum(abs, integ.u .- u0) < atol

        # A reinitialized backward solve is deterministic.
        ufinal = copy(integ.u)
        DiffEqBase.reinit!(integ)
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.u == ufinal
    end

    @testset "Forward-backward roundtrip" begin
        prob_fwd = OperatorSplittingProblem(fsplit, copy(u0), (0.0, 1.0))
        fwd = DiffEqBase.init(prob_fwd, PPLTG((Tsit5(), Tsit5())); dt = 0.01, adaptive = false)
        DiffEqBase.solve!(fwd)
        @test fwd.sol.retcode == ReturnCode.Success

        prob_bwd = OperatorSplittingProblem(fsplit, copy(fwd.u), (1.0, 0.0))
        bwd = DiffEqBase.init(prob_bwd, PPLTG((Tsit5(), Tsit5())); dt = 0.01, adaptive = false)
        DiffEqBase.solve!(bwd)
        @test bwd.sol.retcode == ReturnCode.Success
        @test maximum(abs, bwd.u .- u0) < 1.0e-4
    end

    @testset "Adaptivity works backward" begin
        prob = OperatorSplittingProblem(fsplit, copy(uT), (1.0, 0.0))
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5()));
            dt = 0.5, reltol = 1.0e-6, abstol = 1.0e-8
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.t ≈ 0.0
        # The controller both rejected oversized steps and kept dt pointing backward.
        @test integ.stats.nreject ≥ 1
        @test integ.dt < 0
        @test maximum(abs, integ.u .- u0) < 1.0e-5
    end

    @testset "Nested splitting works backward" begin
        odeB1(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.0; nothing)
        odeB2(du, u, p, t) = (du[1] = 0.0; du[2] = 0.5 * u[1]; nothing)
        f_nested = GenericSplitFunction(
            (fA, GenericSplitFunction((ODEFunction(odeB1), ODEFunction(odeB2)), (dofs, dofs))),
            (dofs, dofs)
        )
        prob = OperatorSplittingProblem(f_nested, copy(uT), (1.0, 0.0))
        alg = PPLTG((Tsit5(), PPLTG((Tsit5(), Tsit5()))))
        integ = DiffEqBase.init(prob, alg; dt = 0.1, reltol = 1.0e-6, abstol = 1.0e-8)
        sub = integ.child_subintegrators[2]
        @test sub.tdir < 0
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.t ≈ 0.0
        @test sub.t ≈ 0.0
        @test maximum(abs, integ.u .- u0) < 1.0e-5
    end

    @testset "reinit! cannot flip the direction" begin
        prob = OperatorSplittingProblem(fsplit, copy(uT), (1.0, 0.0))
        integ = DiffEqBase.init(prob, PPLTG((Tsit5(), Tsit5())); dt = 0.01, adaptive = false)
        @test_throws ErrorException DiffEqBase.reinit!(integ; t0 = 0.0, tf = 1.0)
    end
end
