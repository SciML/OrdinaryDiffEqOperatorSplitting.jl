using OrdinaryDiffEqOperatorSplitting, OrdinaryDiffEq, BenchmarkTools

const SUITE = BenchmarkGroup()

# Split: u' = A u + B u  with  A = diag decay, B = skew coupling
odeA(du, u, p, t) = (du[1] = -u[1]; du[2] = -2 * u[2]; nothing)
odeB(du, u, p, t) = (du[1] = 0.5 * u[2]; du[2] = 0.5 * u[1]; nothing)

dofs = [1, 2]
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)

fsplit = GenericSplitFunction((ODEFunction(odeA), ODEFunction(odeB)), (dofs, dofs))
prob = OperatorSplittingProblem(fsplit, u0, tspan)

# Larger split problem (10 dofs per subproblem)
N = 20
odeA_big(du, u, p, t) = (du .= -u; nothing)
function odeB_big(du, u, p, t)
    @inbounds for i in 1:(length(u) - 1)
        du[i] = 0.1 * (u[i + 1] - u[i])
    end
    du[end] = 0.1 * (u[1] - u[end])
    return nothing
end
fsplit_big = GenericSplitFunction(
    (ODEFunction(odeA_big), ODEFunction(odeB_big)), (1:N, 1:N)
)
prob_big = OperatorSplittingProblem(fsplit_big, ones(N), tspan)

# =============================================================================
# Construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["split_function"] = @benchmarkable GenericSplitFunction(
    ($(ODEFunction(odeA)), $(ODEFunction(odeB))), ($dofs, $dofs)
)
SUITE["construct"]["problem"] = @benchmarkable OperatorSplittingProblem(
    $fsplit, $u0, $tspan
)

# =============================================================================
# Solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["lie_trotter"] = @benchmarkable solve(
    $prob, LieTrotterGodunov((Tsit5(), Tsit5())); dt = 0.1
)
SUITE["solve"]["strang"] = @benchmarkable solve(
    $prob, StrangMarchuk((Tsit5(), Tsit5())); dt = 0.1
)
SUITE["solve"]["strang_big"] = @benchmarkable solve(
    $prob_big, StrangMarchuk((Tsit5(), Tsit5())); dt = 0.1
)
