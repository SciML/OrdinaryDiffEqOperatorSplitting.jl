using OrdinaryDiffEqOperatorSplitting
using Test

import DiffEqBase: DiffEqBase, ODEFunction
import SciMLBase
using OrdinaryDiffEqTsit5

# Convergence orders of every splitting algorithm, in both integration directions,
# on pairwise non-commuting operators so the splitting error dominates. The inner
# solves use adaptive Tsit5, which resolves the sub-problems well below the
# splitting error at these step sizes.
#
#   A = diag(-1, -2), and B = 0.5·offdiag split once more into its strictly upper
#   and lower triangles Bu, Bl: [A, B], [A, Bu], [A, Bl], [Bu, Bl] are all nonzero.
M = [-1.0 0.5; 0.5 -2.0]
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)
odeBu(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.0; nothing)
odeBl(du, u, p, t) = (du[1] = 0.0; du[2] = 0.5 * u[1]; nothing)
fA = ODEFunction(odeA)
fB = ODEFunction(odeB)
fBu = ODEFunction(odeBu)
fBl = ODEFunction(odeBl)

dofs = [1, 2]
u0 = [1.0, 1.0]
uT = exp(M) * u0

f_two = GenericSplitFunction((fA, fB), (dofs, dofs))
f_three = GenericSplitFunction((fA, fBu, fBl), (dofs, dofs, dofs))
f_nested = GenericSplitFunction(
    (fA, GenericSplitFunction((fBu, fBl), (dofs, dofs))), (dofs, dofs)
)

function convergence_rates(f, alg, tspan, ustart, utarget; dts = (0.1, 0.05, 0.025))
    errs = map(dts) do dt
        prob = OperatorSplittingProblem(f, copy(ustart), tspan)
        integ = DiffEqBase.init(prob, alg; dt, adaptive = false, verbose = false)
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == SciMLBase.ReturnCode.Success
        maximum(abs, integ.u .- utarget)
    end
    return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
end

@testset "Convergence order" begin
    for (AlgType, expected_order) in (
            (LieTrotterGodunov, 1),
            (StrangMarchuk, 2),
            (PalindromicPairLieTrotterGodunov, 2),
        )
        cases = (
            ("two operators", f_two, AlgType((Tsit5(), Tsit5()))),
            ("three operators", f_three, AlgType((Tsit5(), Tsit5(), Tsit5()))),
            ("nested", f_nested, AlgType((Tsit5(), AlgType((Tsit5(), Tsit5()))))),
        )
        directions = (
            ("forward", (0.0, 1.0), u0, uT),
            ("backward", (1.0, 0.0), uT, u0),
        )
        @testset "$(nameof(AlgType)) | $case | $dir (order $expected_order)" for
            (case, f, alg) in cases, (dir, tspan, ustart, utarget) in directions

            rates = convergence_rates(f, alg, tspan, ustart, utarget)
            for rate in rates
                @test rate ≈ expected_order atol = 0.3
            end
        end
    end
end
