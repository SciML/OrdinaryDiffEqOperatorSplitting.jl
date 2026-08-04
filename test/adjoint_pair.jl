using OrdinaryDiffEqOperatorSplitting
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
import SciMLBase: ReturnCode
using OrdinaryDiffEqTsit5

using OrdinaryDiffEqOperatorSplitting: order, coefficients

M = [-1.0 0.5; 0.5 -2.0]
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)
dofs = [1, 2]
u0 = [1.0, 1.0]
fsplit = GenericSplitFunction((ODEFunction(odeA), ODEFunction(odeB)), (dofs, dofs))
uexact = exp(M) * u0

function splitting_error(alg, dt)
    prob = OperatorSplittingProblem(fsplit, copy(u0), (0.0, 1.0))
    integ = DiffEqBase.init(
        prob, alg; dt = dt, adaptive = false, abstol = 1.0e-14, reltol = 1.0e-14
    )
    DiffEqBase.solve!(integ)
    @assert integ.sol.retcode == ReturnCode.Success
    return maximum(abs, integ.u .- uexact)
end

@testset "Adjoint pairs" begin
    @testset "the base scheme must be of odd order" begin
        # Odd p is what makes the averaging work: the adjoint's leading error is
        # (-1)^p C h^(p+1), so the signs oppose and cancel. For even p they are equal,
        # the average stays order p, and the difference is no longer an error estimate.
        @test AdjointPair(Ruth3((Tsit5(), Tsit5()))) isa AdjointPair
        @test_throws ArgumentError AdjointPair(StrangMarchuk((Tsit5(), Tsit5())))
    end

    @testset "the pair raises the order by one" begin
        alg = AdjointPair(Ruth3((Tsit5(), Tsit5())))
        @test order(alg) == 4
        # The estimate measures the *base* scheme's leading error, so that is the
        # order the controller sees.
        @test OrdinaryDiffEqOperatorSplitting.alg_adaptive_order(alg) == 3
        @test SciMLBase.isadaptive(alg)
    end

    @testset "AdjointPair(LieTrotterGodunov) reproduces PalindromicPairLieTrotterGodunov" begin
        # PPLTG is exactly this construction at p = 1, hand-written. Agreement to
        # machine precision tests the whole generic path -- table execution, the
        # flat-sequence adjoint, averaging, rollback -- against trusted code.
        prob_pair = OperatorSplittingProblem(fsplit, copy(u0), (0.0, 1.0))
        pair = DiffEqBase.init(
            prob_pair, AdjointPair(LieTrotterGodunov((Tsit5(), Tsit5())));
            dt = 0.1, adaptive = false, abstol = 1.0e-14, reltol = 1.0e-14
        )
        DiffEqBase.solve!(pair)

        prob_ppltg = OperatorSplittingProblem(fsplit, copy(u0), (0.0, 1.0))
        ppltg = DiffEqBase.init(
            prob_ppltg, PalindromicPairLieTrotterGodunov((Tsit5(), Tsit5()));
            dt = 0.1, adaptive = false, abstol = 1.0e-14, reltol = 1.0e-14
        )
        DiffEqBase.solve!(ppltg)

        @test pair.u == ppltg.u
    end

    @testset "the pair is more accurate than its base scheme" begin
        # The order itself is checked in test/convergence.jl. What matters here is that
        # averaging buys accuracy rather than merely costing two sequences.
        @test splitting_error(AdjointPair(Ruth3((Tsit5(), Tsit5()))), 0.1) <
            splitting_error(Ruth3((Tsit5(), Tsit5())), 0.1)
    end

    @testset "the pair difference drives adaptive stepping" begin
        prob = OperatorSplittingProblem(fsplit, copy(u0), (0.0, 1.0))
        integ = DiffEqBase.init(
            prob, AdjointPair(Ruth3((Tsit5(), Tsit5())));
            dt = 0.5, abstol = 1.0e-10, reltol = 1.0e-8
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.t ≈ 1.0
        @test isfinite(integ.EEst)
        @test maximum(abs, integ.u .- uexact) < 1.0e-6
    end
end
