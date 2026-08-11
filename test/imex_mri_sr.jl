using OrdinaryDiffEqOperatorSplitting
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
using OrdinaryDiffEqLowOrderRK: Heun, BS3

using OrdinaryDiffEqOperatorSplitting: MRISRCoefficients,
    IMEX_MRI_SR2_COEFFICIENTS, IMEX_MRI_SR3_COEFFICIENTS

# ---------------------------------------------------------------------------
# The Kvaerno-Prothero-Robinson problem, [FisReyRob:2023:iem] eq. (6.1), with the
# three-way partitioning of eq. (6.3). Nonlinear, non-autonomous, stiff, multirate, and
# with an analytic solution, which is why the paper uses it to measure convergence.
# ---------------------------------------------------------------------------

const λF = -10.0
const λS = -1.0
const ε = 0.1
const α = 1.0
const β = 20.0

const Λ = [
    λF                        (1 - ε) / α * (λF - λS)
    -α * ε * (λF - λS)            λS
]

kpr_exact(t) = [sqrt(3 + cos(β * t)), sqrt(2 + cos(t))]

# The bracketed vector shared by all three partitions of (6.3).
@inline function kpr_inner(u, t)
    return (
        (-3 + u[1]^2 - cos(β * t)) / (2 * u[1]),
        (-2 + u[2]^2 - cos(t)) / (2 * u[2]),
    )
end

# f^{F} = Λ[1 0; 0 0] * inner
function kpr_fast!(du, u, p, t)
    g1, g2 = kpr_inner(u, t)
    du[1] = Λ[1, 1] * g1
    du[2] = Λ[2, 1] * g1
    return nothing
end

# f^{E} = [0 ; sin(t) / (2 v)] with a minus sign, from (6.1)/(6.3)
function kpr_expl!(du, u, p, t)
    du[1] = 0.0
    du[2] = -sin(t) / (2 * u[2])
    return nothing
end

# f^{I} = Λ[0 0; 0 1] * inner, minus the β sin(βt)/(2u) term of (6.1)
function kpr_impl!(du, u, p, t)
    g1, g2 = kpr_inner(u, t)
    du[1] = Λ[1, 2] * g2 - β * sin(β * t) / (2 * u[1])
    du[2] = Λ[2, 2] * g2
    return nothing
end

function kpr_split_function()
    dofs = 1:2
    return GenericSplitFunction(
        (ODEFunction(kpr_fast!), ODEFunction(kpr_expl!), ODEFunction(kpr_impl!)),
        (dofs, dofs, dofs),
    )
end

# Solve KPR with outer step H and fast step H/10, as in §6.1, and return the maximum
# error over ten equally spaced output points.
function kpr_max_error(alg_ctor, fast_alg, H; tend = 5π / 2, adaptive = false, kwargs...)
    f = kpr_split_function()
    u0 = kpr_exact(0.0)
    prob = OperatorSplittingProblem(f, copy(u0), (0.0, tend))

    dt = TreeOption(f, H)
    dt[f[1]] = H / 10

    saveat = collect(range(0.0, tend; length = 11))
    integrator = DiffEqBase.init(
        prob, alg_ctor(fast_alg);
        dt, saveat, adaptive, kwargs...
    )
    DiffEqBase.solve!(integrator)
    sol = integrator.sol

    err = 0.0
    for t in saveat
        err = max(err, maximum(abs, sol(t) .- kpr_exact(t)))
    end
    return err
end

# Least squares slope of log(error) against log(H).
function observed_order(errors, steps)
    x = log.(steps)
    y = log.(errors)
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

@testset "IMEX-MRI-SR" begin
    # The order of the method as a whole is the real check on both the coefficient
    # tables and the step: a wrong coefficient costs an order.
    @testset "convergence order on KPR (§6.1)" begin
        # H = π/2^k as in the paper. Both methods approach their design order from
        # above and are asymptotic by k ≈ 9: measured local orders are 2.14, 2.07,
        # 2.04, 2.02 (SR2) and 3.22, 3.12, 3.06, 3.03 (SR3) for k = 9, 10, 11, 12.
        # Coarser steps are still in the transition regime -- at k = 5 SR2 measures
        # nearly 3 -- and at H ≈ 0.2 the stage solves stop converging altogether.
        ks = 9:11
        steps = [π / 2^k for k in ks]

        for (name, ctor, fast_alg, expected) in (
                ("IMEXMRISR2", IMEXMRISR2, Heun(), 2),
                ("IMEXMRISR3", IMEXMRISR3, BS3(), 3),
            )
            errors = [kpr_max_error(ctor, fast_alg, H) for H in steps]
            @test all(isfinite, errors)
            @test issorted(errors; rev = true)          # error falls as H falls
            p = observed_order(errors, steps)
            @test isapprox(p, expected; atol = 0.2)
        end
    end

    @testset "adaptive stepping tracks the tolerance" begin
        f = kpr_split_function()
        u0 = kpr_exact(0.0)
        tend = 5π / 2

        prev_err = Inf
        for tol in (1.0e-4, 1.0e-6, 1.0e-8)
            prob = OperatorSplittingProblem(f, copy(u0), (0.0, tend))
            dt = TreeOption(f, 0.05)
            dt[f[1]] = 0.005
            integrator = DiffEqBase.init(
                prob, IMEXMRISR3(BS3());
                dt, adaptive = true, abstol = tol, reltol = tol,
            )
            DiffEqBase.solve!(integrator)
            @test SciMLBase.successful_retcode(integrator.sol)
            err = maximum(abs, integrator.u .- kpr_exact(tend))
            @test err < prev_err                 # tighter tolerance, smaller error
            prev_err = err
        end
    end

    @testset "the step is adaptive and reports its orders" begin
        @test SciMLBase.isadaptive(IMEXMRISR2(Heun()))
        @test SciMLBase.isadaptive(IMEXMRISR3(BS3()))
        @test OrdinaryDiffEqOperatorSplitting.order(IMEXMRISR2(Heun())) == 2
        @test OrdinaryDiffEqOperatorSplitting.order(IMEXMRISR3(BS3())) == 3
        # The estimate comes from the embedding, one order below.
        @test OrdinaryDiffEqOperatorSplitting.alg_adaptive_order(IMEXMRISR2(Heun())) == 1
        @test OrdinaryDiffEqOperatorSplitting.alg_adaptive_order(IMEXMRISR3(BS3())) == 2
    end

    @testset "reinit! reproduces the same solution" begin
        f = kpr_split_function()
        u0 = kpr_exact(0.0)
        prob = OperatorSplittingProblem(f, copy(u0), (0.0, 1.0))
        dt = TreeOption(f, 0.05)
        dt[f[1]] = 0.005

        integrator = DiffEqBase.init(prob, IMEXMRISR3(BS3()); dt, adaptive = false)
        DiffEqBase.solve!(integrator)
        first_u = copy(integrator.u)

        DiffEqBase.reinit!(integrator, copy(u0))
        DiffEqBase.solve!(integrator)
        @test integrator.u ≈ first_u
    end

    # The interface errors are worth testing directly: they are the guard rails around
    # reusing GenericSplitFunction for an *additive* partition, which the type system
    # cannot express.
    @testset "misuse is rejected" begin
        dofs = 1:2
        fast, expl, impl = ODEFunction(kpr_fast!), ODEFunction(kpr_expl!), ODEFunction(kpr_impl!)
        u0 = kpr_exact(0.0)

        two_ops = GenericSplitFunction((fast, impl), (dofs, dofs))
        @test_throws ArgumentError DiffEqBase.init(
            OperatorSplittingProblem(two_ops, copy(u0), (0.0, 1.0)),
            IMEXMRISR2(Heun()); dt = 0.1,
        )

        # Disjoint slices: the classic splitting layout, which is wrong here.
        sliced = GenericSplitFunction((fast, expl, impl), (1:1, 2:2, dofs))
        @test_throws ArgumentError DiffEqBase.init(
            OperatorSplittingProblem(sliced, copy(u0), (0.0, 1.0)),
            IMEXMRISR2(Heun()); dt = 0.1,
        )

        # A nested fast operator cannot carry the stage forcing.
        nested_fast = GenericSplitFunction(
            (GenericSplitFunction((fast, fast), (dofs, dofs)), expl, impl),
            (dofs, dofs, dofs),
        )
        @test_throws ArgumentError DiffEqBase.init(
            OperatorSplittingProblem(nested_fast, copy(u0), (0.0, 1.0)),
            IMEXMRISR2(LieTrotterGodunov((Heun(), Heun()))); dt = 0.1,
        )
    end

    # The tables themselves are validated by the constructor, which an order test cannot
    # reach; this checks that the validation actually rejects bad tables.
    @testset "malformed coefficient tables are rejected" begin
        good_Ω = Rational{Int}[
            0 0 0
            1 // 2 0 0
            -1 // 2 3 // 2 0
            0 1 0
        ]
        good_Γ = Rational{Int}[
            0 0 0
            -1 // 2 1 // 2 0
            0 -1 // 2 1 // 2
            1 -1 0
        ]
        good_c = Rational{Int}[0, 1 // 2, 1]
        @test MRISRCoefficients((good_Ω,), good_Γ, good_c) isa MRISRCoefficients

        # eq. (2.11a): row 2 of Ω^{0} no longer sums to c[2].
        bad = copy(good_Ω); bad[2, 1] = 1 // 3
        @test_throws ArgumentError MRISRCoefficients((bad,), good_Γ, good_c)

        # eq. (2.11c): Γ's rows must sum to zero.
        bad = copy(good_Γ); bad[2, 1] = -1 // 3
        @test_throws ArgumentError MRISRCoefficients((good_Ω,), bad, good_c)

        # Ω^{0} must be strictly lower triangular.
        bad = copy(good_Ω); bad[2, 2] = bad[2, 1]; bad[2, 1] = 0
        @test_throws ArgumentError MRISRCoefficients((bad,), good_Γ, good_c)

        # γ̂_S ≠ 0 would demand an implicit solve for the embedding.
        bad = copy(good_Γ); bad[4, 3] = 1; bad[4, 2] = -2
        @test_throws ArgumentError MRISRCoefficients((good_Ω,), bad, good_c)

        # A zero abscissa past the first stage.
        @test_throws ArgumentError MRISRCoefficients(
            (good_Ω,), good_Γ, Rational{Int}[0, 0, 1]
        )

        # No embedding row.
        @test_throws ArgumentError MRISRCoefficients((good_Ω[1:3, :],), good_Γ, good_c)
    end
end
