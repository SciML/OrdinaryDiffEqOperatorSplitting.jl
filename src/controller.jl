@inline OrdinaryDiffEqCore.ispredictive(::AbstractOperatorSplittingAlgorithm) = false
@inline OrdinaryDiffEqCore.isstandard(::AbstractOperatorSplittingAlgorithm) = false
function OrdinaryDiffEqCore.beta2_default(alg::AbstractOperatorSplittingAlgorithm)
    isadaptive(alg) ? 2 // (5alg_order(alg)) : 0 // 1
end
function OrdinaryDiffEqCore.beta1_default(alg::AbstractOperatorSplittingAlgorithm, beta2)
    isadaptive(alg) ? 7 // (10alg_order(alg)) : 0 // 1
end

function OrdinaryDiffEqCore.qmin_default(alg::AbstractOperatorSplittingAlgorithm)
    isadaptive(alg) ? 1 // 5 : 0 // 1
end
OrdinaryDiffEqCore.qmax_default(alg::AbstractOperatorSplittingAlgorithm) = 10 // 1
function OrdinaryDiffEqCore.gamma_default(alg::AbstractOperatorSplittingAlgorithm)
    isadaptive(alg) ? 9 // 10 : 0 // 1
end
OrdinaryDiffEqCore.qsteady_min_default(alg::AbstractOperatorSplittingAlgorithm) = 1 // 1
OrdinaryDiffEqCore.qsteady_max_default(alg::AbstractOperatorSplittingAlgorithm) = 1 // 1

mutable struct PIController{T} <: OrdinaryDiffEqCore.AbstractController
    qmin::T
    qmax::T
    qsteady_min::T
    qsteady_max::T
    qoldinit::T
    beta1::T
    beta2::T
    gamma::T
    # Internal
    q11::T
    qold::T
    q::T
end
PIController(; qmin, qmax, qsteady_min, qsteady_max, qoldinit, beta1, beta2, gamma, q11) = PIController(qmin, qmax, qsteady_min, qsteady_max, qoldinit, beta1, beta2, gamma, q11, qoldinit, qoldinit)

function default_controller(alg, cache)
    @assert isadaptive(alg)

    beta2 = OrdinaryDiffEqCore.beta2_default(alg)
    beta1 = OrdinaryDiffEqCore.beta1_default(alg, beta2)
    qmin = OrdinaryDiffEqCore.qmin_default(alg)
    qmax = OrdinaryDiffEqCore.qmax_default(alg)
    gamma = OrdinaryDiffEqCore.gamma_default(alg)
    qsteady_min = OrdinaryDiffEqCore.qsteady_min_default(alg)
    qsteady_max = OrdinaryDiffEqCore.qsteady_max_default(alg)
    qoldinit = 1 // 10^4
    q11 = 1 // 1
    PIController(;
        beta1, beta2,
        qmin, qmax,
        gamma,
        qsteady_min, qsteady_max,
        qoldinit, q11
    )
end

@inline DiffEqBase.isadaptive(::AbstractOperatorSplittingAlgorithm) = false

@inline function stepsize_controller!(integrator::OperatorSplittingIntegrator, controller::PIController, alg)
    (; qold, qmin, qmax, gamma) = controller
    (; beta1, beta2)            = controller
    EEst = DiffEqBase.value(integrator.EEst)

    if iszero(EEst)
        q = inv(qmax)
    else
        q11 = OrdinaryDiffEqCore.fastpower(EEst, convert(typeof(EEst), beta1))
        q = q11 / OrdinaryDiffEqCore.fastpower(qold, convert(typeof(EEst), beta2))
        controller.q11 = q11
        @fastmath q = max(inv(qmax), min(inv(qmin), q / gamma))
    end
    controller.q = q # Return Q for temporary compat with OrdinaryDiffEqCore
end

function step_accept_controller!(integrator::OperatorSplittingIntegrator, controller::PIController, alg)
    (; q, qsteady_min, qsteady_max, qoldinit) = controller
    EEst = DiffEqBase.value(integrator.EEst)

    if qsteady_min <= q <= qsteady_max
        q = one(q)
    end
    controller.qold = max(EEst, qoldinit)
    integrator.dt /= q
    return nothing
end

function step_reject_controller!(integrator::OperatorSplittingIntegrator, controller::PIController, alg)
    (; q11, qmin, gamma) = controller
    integrator.dt /= min(inv(qmin), q11 / gamma)
    return nothing
end

@inline function should_accept_step(integrator, controller::OrdinaryDiffEqCore.AbstractController)
    return integrator.EEst <= 1
end
