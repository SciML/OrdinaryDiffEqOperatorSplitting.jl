using OrdinaryDiffEqOperatorSplitting
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
import SciMLBase: ReturnCode
using OrdinaryDiffEqTsit5

using OrdinaryDiffEqOperatorSplitting: SplittingCoefficients, coefficients, order

# Non-commuting pair with a known exact solution, as in test/backward.jl.
M = [-1.0 0.5; 0.5 -2.0]
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)
dofs = [1, 2]
u0 = [1.0, 1.0]
fsplit = GenericSplitFunction((ODEFunction(odeA), ODEFunction(odeB)), (dofs, dofs))

# Convergence orders live in test/convergence.jl, which runs every algorithm over two
# and three operators, nested, forward and backward. What is left here is what is
# specific to building a scheme out of a coefficient table.
@testset "Coefficient-table splitting schemes" begin
    @testset "each operator's coefficients must sum to one" begin
        # Ruth's third order table (Ruth, 1983): both columns sum to 1.
        @test SplittingCoefficients((2 // 3, 7 // 24), (-2 // 3, 3 // 4), (1 // 1, -1 // 24)) isa
            SplittingCoefficients
        # Operator 1 sums to 0.9, so the scheme is not even consistent.
        @test_throws ArgumentError SplittingCoefficients((0.5, 0.5), (0.4, 0.5))
    end

    @testset "Ruth3 exposes its table and order" begin
        alg = Ruth3((Tsit5(), Tsit5()))
        @test order(alg) == 3
        @test coefficients(alg) isa SplittingCoefficients
    end

    @testset "Yoshida4's merged table carries a zero coefficient" begin
        # Merging the adjacent flows the triple jump leaves next to each other is what
        # brings it down to eight evaluations, and it zeroes the last stage's second
        # coefficient. Its convergence in test/convergence.jl is therefore also the
        # coverage for the zero-coefficient skip path.
        table = coefficients(Yoshida4((Tsit5(), Tsit5()))).a
        @test order(Yoshida4((Tsit5(), Tsit5()))) == 4
        @test length(table) == 4
        @test count(iszero, Iterators.flatten(table)) == 1
        @test iszero(table[end][end])
    end

    @testset "tables stay compile-time constants" begin
        # The stage count is a type parameter derived from `length`, and `@unroll` can
        # only unroll the stage loop if that survives inference. Lie-Trotter's table is
        # the one actually built at run time, from the length of `inner_algs`.
        @test (@inferred coefficients(Ruth3((Tsit5(), Tsit5())))) isa
            SplittingCoefficients{3, 2}
        @test (@inferred coefficients(LieTrotterGodunov((Tsit5(), Tsit5())))) isa
            SplittingCoefficients{1, 2}
    end

    @testset "the table is re-applied faithfully on every step" begin
        # A table ends on the last operator, so at the next step's start operator 1's
        # buffer is stale and the `mark_next_sync_continuous` shortcut is invalid for
        # it. That bug leaves the *first* step exact and corrupts every later one, so
        # a single-step check cannot see it: compare several steps against the
        # scheme's own composition of exact linear flows.
        A = [-1.0 0.0; 0.0 -2.0]
        B = [0.0 0.5; 0.5 0.0]
        a = coefficients(Ruth3((Tsit5(), Tsit5()))).a

        h = 0.1
        reference = copy(u0)
        for _ in 1:4, stage in a
            reference = exp(B * (float(stage[2]) * h)) * (exp(A * (float(stage[1]) * h)) * reference)
        end

        prob = OperatorSplittingProblem(fsplit, copy(u0), (0.0, 4h))
        integ = DiffEqBase.init(
            prob, Ruth3((Tsit5(), Tsit5())); dt = h, abstol = 1.0e-14, reltol = 1.0e-14
        )
        DiffEqBase.solve!(integ)

        @test integ.iter == 4
        @test integ.u ≈ reference atol = 1.0e-12
    end
end
