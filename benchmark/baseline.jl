using BenchmarkTools, TimerOutputs, SciMLBase
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqOperatorSplitting
using LinearAlgebra

mutable struct CR{F}; f::F; n::Int; end
(c::CR)(du,u,p,t) = (c.n += 1; c.f(du,u,p,t))

ssprk33!(u,u1,du,rhs!,h) = begin
    rhs!(du,u, nothing,0); @. u1 = u + h*du
    rhs!(du,u1,nothing,0); @. u1 = 0.75u + 0.25*(u1 + h*du)
    rhs!(du,u1,nothing,0); @. u  = (1/3)*u + (2/3)*(u1 + h*du)
end
strang_hand!(u,u1,du,h,fA,fB) = (ssprk33!(u,u1,du,fA,h/2);
                                 ssprk33!(u,u1,du,fB,h);
                                 ssprk33!(u,u1,du,fA,h/2))

N, dt = 100_000, 1e-3
u0 = ones(N); dofs = collect(1:N)
A = Tridiagonal(-ones(N-1), 2ones(N), -ones(N-1))
fA() = CR((du,u,p,t)->(du .= -0.1 .* A * u), 0)
fB() = CR((du,u,p,t)->(du .= -u .+ 0.01/length(u) .* sum(abs2,u)), 0)

function sciml_setup(inner_B, inner_A = SSPRK33())
    a, b = fA(), fB()
    fsplit = GenericSplitFunction((ODEFunction(a), ODEFunction(b)), (dofs, dofs))
    alg    = StrangMarchuk((inner_A, inner_B))
    integ  = init(OperatorSplittingProblem(fsplit, copy(u0), (0.0,1e6)), alg;
                  dt=dt, adaptive=false, alias_u0=false, verbose=false)
    return integ
end
hand_step() = (a = fA(); b = fB(); u=copy(u0); u1=similar(u); du=similar(u);
               strang_hand!(u,u1,du,dt,a,b); a.n = 0; b.n = 0;
               strang_hand!(u,u1,du,dt,a,b); (a.n, b.n))

b_hand  = @benchmark strang_hand!(u,u1,du,dt,a,b) setup=(a = fA(); b = fB(); u=copy(u0); u1=similar(u); du=similar(u))
b_ssprk = @benchmark SciMLBase.step!(integ) setup=(integ=sciml_setup(SSPRK33()))
b_eul   = @benchmark SciMLBase.step!(integ) setup=(integ=sciml_setup(Euler()))

TimerOutputs.enable_debug_timings(OrdinaryDiffEqOperatorSplitting)

integ=sciml_setup(SSPRK33(), SSPRKMSVS32())
step!(integ);
reset_timer!(); step!(integ); print_timer()
