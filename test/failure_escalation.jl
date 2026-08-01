using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
import SciMLBase: ReturnCode
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5

# Failure protocol: a failing *adaptive* node is fatal (it exhausted its own
# adaptation); a failing non-adaptive node escalates to the nearest adaptive
# ancestor, which retries on a failfactor-shrunken interval; a non-adaptive root
# stops with the escalated retcode.

M = [-1.0 0.5; 0.5 -2.0]
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
fA = ODEFunction(odeA)
dofs = [1, 2]
u0 = [1.0, 1.0]
trueu = exp(M) * u0

# The coupling operator B (or a part of it), but producing NaNs for a few
# evaluations once `t` first crosses `tfail`: a transient inner failure in the
# middle of the solve. Time-triggered (leaf integrators evaluate `f` at init, so a
# purely count-based trigger would be consumed before any stepping), and emitting
# several NaNs (a single one lands in the FSAL cache, which the forward sync's
# u_modified re-evaluation heals without the step ever failing).
mutable struct TransientFailure{F}
    countdown::Int
    triggered::Bool
    tfail::Float64
    f::F
end
TransientFailure(tfail, f) = TransientFailure(3, false, tfail, f)
function (tf::TransientFailure)(du, u, p, t)
    tf.triggered |= t >= tf.tfail
    if tf.triggered && tf.countdown > 0
        tf.countdown -= 1
        du .= NaN
    else
        tf.f(du, u, p, t)
    end
    return nothing
end
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)
odeB1(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.0; nothing)
odeB2(du, u, p, t) = (du[1] = 0.0; du[2] = 0.5 * u[1]; nothing)
ode_nan(du, u, p, t) = (du .= NaN; nothing)

PPLTG = PalindromicPairLieTrotterGodunov

@testset "Failure escalation" begin
    @testset "Transient non-adaptive failure recovers via the adaptive root" begin
        ffail = TransientFailure(0.35, odeB)
        f = GenericSplitFunction((fA, ODEFunction(ffail)), (dofs, dofs))
        prob = OperatorSplittingProblem(f, copy(u0), (0.0, 1.0))
        # Per-node defaults: adaptive PPLTG root, non-adaptive Euler leaf for B.
        integ = DiffEqBase.init(prob, PPLTG((Tsit5(), Euler())); dt = 0.1)
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.nreject ≥ 1    # the failed attempt was retried
        @test ffail.triggered               # ... and the failure was actually hit
        @test maximum(abs, integ.u .- trueu) < 0.05
    end

    @testset "Adaptive child failure is fatal, without retries" begin
        f = GenericSplitFunction((fA, ODEFunction(ode_nan)), (dofs, dofs))
        prob = OperatorSplittingProblem(f, copy(u0), (0.0, 1.0))
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Tsit5())); dt = 0.1, verbose = false
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode ∈
            (ReturnCode.Unstable, ReturnCode.DtNaN, ReturnCode.DtLessThanMin)
        # The fatal branch stops immediately instead of shrinking dt to dtmin.
        @test integ.iter ≤ 2
    end

    @testset "Failure escalates past a non-adaptive intermediate node" begin
        ffail = TransientFailure(0.35, odeB2)
        f_nested = GenericSplitFunction(
            (fA, GenericSplitFunction((ODEFunction(odeB1), ODEFunction(ffail)), (dofs, dofs))),
            (dofs, dofs)
        )
        prob = OperatorSplittingProblem(f_nested, copy(u0), (0.0, 1.0))
        # The inner LieTrotterGodunov node cannot adapt, so the failure of its
        # Euler leaf has to bubble up to the PPLTG root.
        alg = PPLTG((Tsit5(), LieTrotterGodunov((Euler(), Euler()))))
        integ = DiffEqBase.init(prob, alg; dt = 0.1)
        sub = integ.child_subintegrators[2]
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.stats.nreject ≥ 1
        @test ffail.triggered
        @test SciMLBase.successful_retcode(sub.status.retcode)
        @test maximum(abs, integ.u .- trueu) < 0.05
    end

    @testset "A transient failure under a non-adaptive root still stops" begin
        ffail = TransientFailure(0.35, odeB)
        f = GenericSplitFunction((fA, ODEFunction(ffail)), (dofs, dofs))
        prob = OperatorSplittingProblem(f, copy(u0), (0.0, 1.0))
        integ = DiffEqBase.init(
            prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.1, adaptive = false
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.Unstable
        @test ffail.triggered
        @test integ.t < 1.0  # aborted mid-solve, nobody could retry
    end

    @testset "Rollback clears sticky Newton-style leaf failure flags" begin
        # SciMLBase's generic check_error re-derives ConvergenceFailure from
        # `last_stepfail` on non-adaptive leaves even after the retcode is reset,
        # so the escalation retry has to clear the flag during rollback or it dies
        # again immediately. An implicit inner solver sets the flag organically on
        # a Newton failure; it is injected here because the test dependencies are
        # explicit solvers only.
        f = GenericSplitFunction((fA, ODEFunction(odeB)), (dofs, dofs))
        prob = OperatorSplittingProblem(f, copy(u0), (0.0, 1.0))
        integ = DiffEqBase.init(prob, PPLTG((Tsit5(), Euler())); dt = 0.1, verbose = false)
        DiffEqBase.step!(integ)

        leaf = integ.child_subintegrators[2]
        leaf.last_stepfail = true
        @test SciMLBase.check_error(leaf) == ReturnCode.ConvergenceFailure
        @test OS._child_failed(leaf)     # the eager detection sees it ...

        OS.reject_step!(integ)           # ... and the retry's rollback clears it
        @test !leaf.last_stepfail
        @test !OS._child_failed(leaf)
        @test SciMLBase.check_error(leaf) == ReturnCode.Success
    end

    @testset "Persistent non-adaptive failure exhausts the adaptive root" begin
        f = GenericSplitFunction((fA, ODEFunction(ode_nan)), (dofs, dofs))
        prob = OperatorSplittingProblem(f, copy(u0), (0.0, 1.0))
        integ = DiffEqBase.init(
            prob, PPLTG((Tsit5(), Euler())); dt = 0.1, verbose = false
        )
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == ReturnCode.DtLessThanMin
    end
end
