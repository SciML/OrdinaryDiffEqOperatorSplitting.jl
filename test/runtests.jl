using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import UnPack: @unpack

import SciMLBase: SciMLBase, ReturnCode
import DiffEqBase: DiffEqBase, ODEFunction, ODEProblem


# For testing purposes taken from https://github.com/SciML/SimpleDiffEq.jl
struct SimpleEuler end

################################################################################
#                                  Solution
################################################################################

struct DummyODESolution <: SciMLBase.AbstractODESolution{Float64,2,Vector{Float64}}
    retcode::SciMLBase.ReturnCode.T
end
DummyODESolution() = DummyODESolution(SciMLBase.ReturnCode.Default)
function SciMLBase.solution_new_retcode(sol::DummyODESolution, retcode)
    return DiffEqBase.@set sol.retcode = retcode
end

mutable struct SimpleEulerIntegrator{IIP, S, T, P, F} <:
               DiffEqBase.AbstractODEIntegrator{SimpleEuler, IIP, S, T}
    f::F             # ..................................... Equations of motion
    uprev::S         # .......................................... Previous state
    u::S             # ........................................... Current state
    tmp::S           #  Auxiliary variable similar to state to avoid allocations
    tprev::T         # ...................................... Previous time step
    t::T             # ....................................... Current time step
    t0::T            # ........... Initial time step, only for re-initialization
    dt::T            # ............................................... Step size
    tdir::T          # ...................................... Not used for Euler
    p::P             # .................................... Parameters container
    u_modified::Bool # ..... If `true`, then the input of last step was modified
    sol::DummyODESolution
end

const SEI = SimpleEulerIntegrator

SciMLBase.isautodifferentiable(alg::SimpleEuler) = true
SciMLBase.allows_arbitrary_number_types(alg::SimpleEuler) = true
SciMLBase.allowscomplex(alg::SimpleEuler) = true
SciMLBase.isadaptive(alg::SimpleEuler) = false

# If `true`, then the equation of motion format is `f!(du,u,p,t)` instead of
# `du = f(u,p,t)`.
DiffEqBase.isinplace(::SEI{IIP}) where {IIP} = IIP


################################################################################
#                                Initialization
################################################################################

function DiffEqBase.__init(prob::ODEProblem, alg::SimpleEuler;
    dt = error("dt is required for this algorithm"))
    simpleeuler_init(prob.f,
        DiffEqBase.isinplace(prob),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p)
end

function DiffEqBase.__solve(prob::ODEProblem, alg::SimpleEuler;
    dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = _copy(u0)

    integ = simpleeuler_init(prob.f, DiffEqBase.isinplace(prob), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n - 1)
        step!(integ)
        us[i + 1] = _copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol;
            timeseries_errors = true,
            dense_errors = false)

    return sol
end

@inline function simpleeuler_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
    p::P) where
    {F, P, T, S}
    integ = SEI{IIP, S, T, P, F}(f,
        _copy(u0),
        _copy(u0),
        _copy(u0),
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true)

    return integ
end

################################################################################
#                                   Stepping
################################################################################

@inline function DiffEqBase.step!(integ::SEI{true, S, T}) where {T, S}
    integ.uprev .= integ.u
    tmp = integ.tmp
    f! = integ.f
    p = integ.p
    t = integ.t
    dt = integ.dt
    uprev = integ.uprev
    u = integ.u

    f!(u, uprev, p, t)
    @. u = uprev + dt * u

    integ.tprev = t
    integ.t += dt

    return nothing
end

@inline function DiffEqBase.step!(integ::SEI{false, S, T}) where {T, S}
    integ.uprev = integ.u
    f = integ.f
    p = integ.p
    t = integ.t
    dt = integ.dt
    uprev = integ.uprev

    k = f(uprev, p, t)
    integ.u = uprev + dt * k
    integ.tprev = t
    integ.t += dt

    return nothing
end

################################################################################
#                                Interpolation
################################################################################

@inline function (integ::SEI)(t::T) where {T}
    t₁, t₀, dt = integ.t, integ.tprev, integ.dt

    y₀ = integ.uprev
    y₁ = integ.u
    Θ = (t - t₀) / dt

    # Hermite interpolation.
    @inbounds if !isinplace(integ)
        u = (1 - Θ) * y₀ + Θ * y₁
        return u
    else
        for i in 1:length(u)
            u = @. (1 - Θ) * y₀ + Θ * y₁
        end
        return u
    end
end


################################################################################
#                            OperatorSplitting API Compat
################################################################################

SciMLBase.set_proposed_dt!(integrator::SEI, dt) = integrator.dt = dt
DiffEqBase.has_reinit(integrator::SEI) = false # FIXME Ignored :)
function DiffEqBase.reinit!(
    integrator::SEI,
    u0 ;# = integrator.sol.prob.u0;
    t0 ,# = integrator.sol.prob.tspan[1],
    tf ,# = integrator.sol.prob.tspan[2],
    dt0 = tf-t0,
    erase_sol = false,
    tstops = nothing, # integrator.opts.tstops_cache,
    saveat = nothing, #  integrator.opts.saveat_cache,
    d_discontinuities = nothing,# = integrator.opts.d_discontinuities_cache,
    reinit_callbacks = true,
    reinit_retcode = true,
    reinit_cache = true,
)
    SciMLBase.recursivecopy!(integrator.u, u0)
    SciMLBase.recursivecopy!(integrator.uprev, integrator.u)
    integrator.t = t0
    integrator.tprev = t0

    # integrator.iter = 0
    integrator.u_modified = false

    # integrator.stats.naccept = 0
    # integrator.stats.nreject = 0

    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    # if reinit_callbacks
    #     DiffEqBase.initialize!(integrator.opts.callback, u0, t0, integrator)
    # else # always reinit the saving callback so that t0 can be saved if needed
    #     saving_callback = integrator.opts.callback.discrete_callbacks[end]
    #     DiffEqBase.initialize!(saving_callback, u0, t0, integrator)
    # end
    if reinit_retcode
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Default)
    end

    # tType = typeof(integrator.t)
    # tspan = (tType(t0), tType(tf))
    # integrator.opts.tstops = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, tspan)
    # integrator.opts.saveat = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, tspan)
    # integrator.opts.d_discontinuities = OrdinaryDiffEqCore.initialize_d_discontinuities(tType,
    #     d_discontinuities,
    #     tspan)

    if reinit_cache
        #DiffEqBase.initialize!(integrator, integrator.cache)
    end
end

DiffEqBase.add_tstop!(integrator::SEI, tnext) = nothing

SciMLBase.isadaptive(integrator::SEI) = false

# Not compatible when `opts` not present...
function SciMLBase.check_error(integrator::SimpleEulerIntegrator)
    if integrator.sol.retcode ∉ (ReturnCode.Success, ReturnCode.Default)
        return integrator.sol.retcode
    end
    return ReturnCode.Success
end


# Custom __init to make an algorithm compatible with the operator splitting API
function OS.build_subintegrator_tree_with_cache(
    f::F,
    alg::SimpleEuler, p::P,
    uprevouter::S, uouter::S,
    solution_indices,
    t0::T, dt::T, tf::T,
    tstops, saveat, d_discontinuities, callback,
    adaptive, verbose,
    save_end=false,
    controller=nothing,
) where {S, T, P, F}
    uprev = @view uprevouter[solution_indices]
    u = @view uouter[solution_indices]

    integrator = SEI{true, S, T, P, F}(f,
        copy(uprev),
        copy(u),
        copy(u),
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true,
        DummyODESolution()
    )

    return integrator, integrator
end

################################################################################
#                                  Test
################################################################################

@testset "Operator Splitting API" begin
    # Reference
    function ode_true(du, u, p, t)
        du .= -0.1u
        du[1] += 0.01u[3]
        du[3] += 0.01u[1]
    end

    # Setup individual functions
    # Diagonal components
    function ode1(du, u, p, t)
        @. du = -0.1u
    end
    # Offdiagonal components
    function ode2(du, u, p, t)
        du[1] = 0.01u[2]
        du[2] = 0.01u[1]
    end

    f1 = ODEFunction(ode1)
    f2 = ODEFunction(ode2)

    # Here we describe index sets f1dofs and f2dofs that map the
    # local indices in f1 and f2 into the global problem. Just put
    # ode_true and ode1/ode2 side by side to see how they connect.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    fsplit1 = GenericSplitFunction((f1,f2), (f1dofs, f2dofs))

    # Now the usual setup just with our new problem type.
    # u0 = rand(3)
    u0 = [0.7611944793397108
        0.9059606424982555
        0.5755174199139956]
    tspan = (0.0,100.0)
    prob = OperatorSplittingProblem(fsplit1, u0, tspan)

    # Now some recursive splitting
    function ode3(du, u, p, t)
        du[1] = 0.005u[2]
        du[2] = 0.005u[1]
    end
    f3 = ODEFunction(ode3)
    # The time stepper carries the individual solver information.

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    f3dofs = [1,3]
    fsplit2_inner = GenericSplitFunction((f2,f3), (f3dofs, f3dofs))
    fsplit2_outer = GenericSplitFunction((f1,fsplit2_inner), (f1dofs, f2dofs))

    prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)

    function ode_NaN(du, u, p, t)
        du[1] = NaN
        du[2] = 0.01u[1]
    end

    f_NaN = ODEFunction(ode_NaN)
    f_NaN_dofs = f3dofs
    fsplit_NaN = GenericSplitFunction((f1,f_NaN), (f1dofs, f_NaN_dofs))
    prob_NaN = OperatorSplittingProblem(fsplit_NaN, u0, tspan)

    function ode2_force_half(du, u, p, t)
        du[1] = 0.5
        du[2] = 0.5
    end

    f2half = ODEFunction(ode2_force_half)
    fsplit_force_half = GenericSplitFunction((f1,f2half), (f1dofs, f2dofs))
    prob_force_half = OperatorSplittingProblem(fsplit_force_half, u0, tspan)

    dt = 0.01π
    @testset "OperatorSplitting" begin
        for TimeStepperType in (LieTrotterGodunov,)
            timestepper = TimeStepperType(
                (SimpleEuler(), SimpleEuler())
            )
            timestepper_inner = TimeStepperType(
                (SimpleEuler(), SimpleEuler())
            )
            timestepper2 = TimeStepperType(
                (SimpleEuler(), timestepper_inner)
            )

            for (tstepper1, tstepper_inner, tstepper2) in (
                    (timestepper, timestepper_inner, timestepper2),
                    )
                # The remaining code works as usual.
                integrator = DiffEqBase.init(prob, tstepper1, dt=dt, verbose=true, alias_u0=false)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal = copy(integrator.u)
                @test ufinal ≉ u0 # Make sure the solve did something

                # DiffEqBase.reinit!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0)
                # end
                # @test  isapprox(ufinal, integrator.u, atol=1e-8)

                # DiffEqBase.reinit!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
                # end
                # @test  isapprox(ufinal, integrator.u, atol=1e-8)

                # DiffEqBase.reinit!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # DiffEqBase.solve!(integrator)
                # @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success

                integrator2 = DiffEqBase.init(prob2, tstepper2, dt=dt, verbose=true, alias_u0=false)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal2 = copy(integrator2.u)
                @test ufinal2 ≉ u0 # Make sure the solve did something

                # DiffEqBase.reinit!(integrator2)
                # @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                # for (u, t) in DiffEqBase.TimeChoiceIterator(integrator2, 0.0:5.0:100.0)
                # end
                # @test isapprox(ufinal2, integrator2.u, atol=1e-8)

                # DiffEqBase.reinit!(integrator2)
                # @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                # DiffEqBase.solve!(integrator2)
                # @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Success
                @testset "NaNs" begin
                    integrator_NaN = DiffEqBase.init(prob_NaN, tstepper1, dt=dt, verbose=true, alias_u0=false)
                    @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                    DiffEqBase.solve!(integrator_NaN)
                    @test integrator_NaN.sol.retcode ∈ (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
                end
            end
        end
    end
end
