# ---------------------------------------------------------------------------
# Implicit-explicit multirate infinitesimal stage-restart (IMEX-MRI-SR) methods
#
# [FisReyRob:2023:iem](@cite). Unlike every other scheme in this package, these are not
# sequences of flows over disjoint slices of the state: the three operators are an
# *additive* partition of one full-state right-hand side,
#
#     y' = f^{F}(t, y) + f^{E}(t, y) + f^{I}(t, y),
#
# into fast, slow-explicit and slow-implicit parts. A step evolves a sequence of forced
# fast initial value problems, each restarted from y_n -- hence "stage-restart" -- and
# follows each with an implicit solve at the slow time scale.
# ---------------------------------------------------------------------------

"""
    AbstractIMEXMRISR <: AbstractOperatorSplittingAlgorithm

Supertype of the IMEX-MRI-SR methods of [FisReyRob:2023:iem](@cite).

An algorithm in this family expects exactly three operators, in this order:

| position | operator | role                                                   |
|:---------|:---------|:-------------------------------------------------------|
| 1        | `f^{F}`  | fast; integrated by the inner algorithm, with forcing  |
| 2        | `f^{E}`  | slow explicit; only ever evaluated                     |
| 3        | `f^{I}`  | slow implicit; evaluated and solved against            |

All three must act on the whole state, so all three `solution_indices` of the
[`GenericSplitFunction`](@ref) span it.
"""
abstract type AbstractIMEXMRISR <: AbstractOperatorSplittingAlgorithm end

# Only operator 1 gets a child integrator; f^{E} and f^{I} are evaluated directly.
child_node_count(::AbstractIMEXMRISR, f) = 1

# `inner_algs` is `(fast_alg, nothing, nothing)`, so the generic definition -- which
# maps over every entry -- would trip over the two placeholders.
@inline isdtchangeable(alg::AbstractIMEXMRISR) = isdtchangeable(alg.inner_algs[1])

@inline SciMLBase.isadaptive(::AbstractIMEXMRISR) = true

# The estimate comes from the embedding, whose order is one below the primary method's.
alg_adaptive_order(alg::AbstractIMEXMRISR) = order(alg) - 1

Base.show(io::IO, alg::AbstractIMEXMRISR) =
    print(io, _imex_mri_sr_name(alg), " (", sprint(Base.show, alg.inner_algs[1]), ")")

"""
    IMEXMRISR2(fast_alg; nlsolve = NewtonRaphson())

Second order IMEX-MRI-SR method with a first order embedding, `IMEX-MRI-SR2(1)` of
[FisReyRob:2023:iem](@cite), §4.1. Four stages, `nΩ = 1`, three slow nonlinear solves
per step.

`fast_alg` integrates the forced fast initial value problems; `nlsolve` solves the slow
implicit stages. See [`AbstractIMEXMRISR`](@ref) for the required operator order.
"""
struct IMEXMRISR2{AlgTupleType <: Tuple, NL} <: AbstractIMEXMRISR
    inner_algs::AlgTupleType    # (fast_alg, nothing, nothing)
    nlsolve::NL
end

"""
    IMEXMRISR3(fast_alg; nlsolve = NewtonRaphson())

Third order IMEX-MRI-SR method with a second order embedding, `IMEX-MRI-SR3(2)` of
[FisReyRob:2023:iem](@cite), §4.2. Five stages, `nΩ = 2`, four slow nonlinear solves
per step.

`fast_alg` integrates the forced fast initial value problems; `nlsolve` solves the slow
implicit stages. See [`AbstractIMEXMRISR`](@ref) for the required operator order.
"""
struct IMEXMRISR3{AlgTupleType <: Tuple, NL} <: AbstractIMEXMRISR
    inner_algs::AlgTupleType
    nlsolve::NL
end

for (T, name, ord, table) in (
        (:IMEXMRISR2, "IMEX-MRI-SR2(1)", 2, :IMEX_MRI_SR2_COEFFICIENTS),
        (:IMEXMRISR3, "IMEX-MRI-SR3(2)", 3, :IMEX_MRI_SR3_COEFFICIENTS),
    )
    @eval begin
        $T(fast_alg; nlsolve = NewtonRaphson()) =
            $T((fast_alg, nothing, nothing), nlsolve)
        _imex_mri_sr_name(::$T) = $name
        order(::$T) = $ord
        coefficients(::$T) = $table
    end
end

# ---------------------------------------------------------------------------
# Forcing of the fast initial value problems
# ---------------------------------------------------------------------------

"""
    MRIForcing

The forcing term `g(θ)` currently applied to the fast right-hand side, where
`θ = t - tn` is the elapsed fast time within a stage.

Definition 1.1 writes the forcing of stage `i` as a sum over previous stages,

```
g_i(θ) = (1/c_i) Σ_{j<i} ω_{i,j}(θ/(c_i H)) (f_j^{E} + f_j^{I}),
```

but `ω_{i,j}` is a polynomial in the normalized time `τ = θ/(c_i H)`, so the sum over
stages can be collapsed once per stage into one vector per power of `τ`:

```
g_i(θ) = Σ_k τ^k G^{k},   G^{k} = (1/c_i) Σ_{j<i} ω_{i,j}^{k} (f_j^{E} + f_j^{I}).
```

Each fast right-hand side evaluation then costs `nΩ` axpys instead of a sum over
stages. `G` is mutated in place between stages; the fast integrator sees the change
because it holds this same object.
"""
mutable struct MRIForcing{T, uType, NΩ}
    tn::T                      # start of the step; θ = t - tn
    scale::T                   # τ = θ / scale
    const G::NTuple{NΩ, uType} # coefficient of τ^k is G[k+1]
end

"""
    ForcedFastFunction(f, forcing)

The fast right-hand side `f^{F}` plus the current [`MRIForcing`](@ref), which is what
an IMEX-MRI-SR stage actually integrates. In-place only.
"""
struct ForcedFastFunction{F, C <: MRIForcing}
    f::F
    forcing::C
end

function (w::ForcedFastFunction)(du, u, p, t)
    w.f(du, u, p, t)
    return _add_forcing!(du, w.forcing, t)
end

function _add_forcing!(du, forcing::MRIForcing, t)
    τ = (t - forcing.tn) / forcing.scale
    return _accumulate_forcing!(du, forcing.G, τ, one(τ))
end

# Recursion over the tuple rather than a loop, so the powers of τ are unrolled and no
# power is recomputed.
_accumulate_forcing!(du, ::Tuple{}, τ, τpow) = nothing
function _accumulate_forcing!(du, G::Tuple, τ, τpow)
    g = first(G)
    if isone(τpow)
        @. du += g
    else
        @. du += τpow * g
    end
    return _accumulate_forcing!(du, Base.tail(G), τ, τpow * τ)
end

# ---------------------------------------------------------------------------
# The slow implicit stage system
# ---------------------------------------------------------------------------

# Parameters of the stage system, mutated between stages so that one nonlinear solver
# cache serves the whole step.
#
# `p` is f^{I}'s parameter object, carried here so the residual can reach it through the
# NonlinearProblem's parameters. It is refreshed from the integrator every stage rather
# than captured once: a callback may reassign `integrator.p`, and a stale copy would
# silently solve the stage against the old parameters.
mutable struct MRIStageParams{T, uType, P}
    p::P
    t::T            # tn + c_i H
    γH::T           # H γ_{i,i}
    const z::uType  # everything in the stage that does not depend on Y_i
end

# Y_i - z_i - H γ_{i,i} f^{I}(t_i, Y_i) = 0
struct MRIStageResidual{F}
    f_impl::F
end

function (r::MRIStageResidual)(res, Y, sp::MRIStageParams)
    r.f_impl(res, Y, sp.p, sp.t)
    @. res = Y - sp.z - sp.γH * res
    return nothing
end

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

struct IMEXMRISRCache{
        uType, uprevType, coeffType, forcingType, YType, spType, nlType,
    } <: AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    coeffs::coeffType
    forcing::forcingType
    Y::YType            # stage solutions Y_i
    fE::YType           # f^{E}(tn + c_i H, Y_i)
    fI::YType           # f^{I}(tn + c_i H, Y_i)
    uembed::uType       # embedded solution; reused as the residual buffer
    stage_params::spType
    nlcache::nlType
end

function init_cache_with_parameters(
        f::GenericSplitFunction, alg::AbstractIMEXMRISR, p;
        uprev::AbstractArray, u::AbstractVector,
    )
    _check_imex_mri_sr_function(f)

    T = eltype(u)
    coeffs = convert_coefficients(coefficients(alg), T)
    S = nstages(coeffs)
    NΩ = num_omega(coeffs)

    forcing = MRIForcing(zero(T), one(T), ntuple(_ -> zero(u), NΩ))

    # One nonlinear solver cache serves every stage of every step: only `t`, `γH` and
    # `z` change between stages, and those live in the stage parameters.
    stage_params = MRIStageParams(p[3], zero(T), zero(T), zero(u))
    nlprob = SciMLBase.NonlinearProblem(
        SciMLBase.NonlinearFunction{true}(MRIStageResidual(get_operator(f, 3))),
        zero(u), stage_params,
    )
    nlcache = SciMLBase.init(nlprob, alg.nlsolve)

    return IMEXMRISRCache(
        u, uprev, coeffs, forcing,
        ntuple(_ -> zero(u), S), ntuple(_ -> zero(u), S), ntuple(_ -> zero(u), S),
        zero(u), stage_params, nlcache,
    )
end

function _check_imex_mri_sr_function(f::GenericSplitFunction)
    n = num_operators(f)
    n == 3 || throw(
        ArgumentError(
            "an IMEX-MRI-SR method needs exactly three operators (f^{F}, f^{E}, f^{I}) \
             but the split function has $n. This family is an additive partition of one \
             full-state right-hand side, not a splitting over disjoint state slices."
        )
    )
    get_operator(f, 1) isa GenericSplitFunction && throw(
        ArgumentError(
            "the fast operator of an IMEX-MRI-SR method cannot itself be a \
             GenericSplitFunction: its right-hand side is wrapped to carry the stage \
             forcing, which a nested splitting node has no way to apply."
        )
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Tree construction
#
# Only the fast operator becomes a child integrator, and it integrates f^{F} wrapped
# around the cache's forcing term rather than f^{F} itself.
# ---------------------------------------------------------------------------

function build_subintegrators(
        prob::OperatorSplittingProblem,
        alg::AbstractIMEXMRISR,
        uprevouter::AbstractVector,
        uouter::AbstractVector,
        u_master::AbstractVector,
        solution_indices,
        t0, tf,
        tstops, saveat, d_discontinuities, callback,
        config::ConfigTree,
        cache::IMEXMRISRCache,
    )
    (; f, p) = prob
    _check_imex_mri_sr_indices(f, length(uouter))

    f_fast = get_operator(f, 1)
    forced = SciMLBase.ODEFunction{true}(
        ForcedFastFunction(f_fast, cache.forcing);
        mass_matrix = _fast_mass_matrix(f_fast),
    )

    child = _build_child(
        prob, alg.inner_algs[1], forced, p[1],
        uprevouter, uouter, u_master,
        get_solution_indices(f, 1),
        t0, tf,
        tstops, saveat, d_discontinuities, callback,
        config.children[1],
    )
    return (child,)
end

_fast_mass_matrix(f::SciMLBase.AbstractDiffEqFunction) = f.mass_matrix
_fast_mass_matrix(_) = SciMLBase.I

# These methods are an additive partition of one full-state right-hand side, so every
# operator must see the whole state in its natural order.
function _check_imex_mri_sr_indices(f::GenericSplitFunction, n)
    for i in 1:3
        idxs = get_solution_indices(f, i)
        collect(idxs) == collect(1:n) || throw(
            ArgumentError(
                "operator $i of an IMEX-MRI-SR split function has solution_indices \
                 $idxs, but all three operators must span the whole state as 1:$n. \
                 f^{F}, f^{E} and f^{I} are an additive partition of one right-hand \
                 side, not a splitting over disjoint slices of the state."
            )
        )
    end
    return nothing
end

# An IMEX-MRI-SR node reads f^{E}, f^{I} and their parameters off the integrator, which
# only the root carries, so it cannot sit inside another splitting.
function _build_child(
        ::OperatorSplittingProblem,
        alg::AbstractIMEXMRISR,
        ::GenericSplitFunction,
        ::Any,
        ::AbstractVector, ::AbstractVector, ::AbstractVector,
        _, _, _, _, _, _, _, ::ConfigTree
    )
    return throw(
        ArgumentError(
            "$(_imex_mri_sr_name(alg)) can only be the outermost algorithm of an \
             operator splitting problem, not an inner node of another splitting."
        )
    )
end

# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

function _perform_step!(
        parent,
        children::Tuple,
        cache::IMEXMRISRCache,
        dt
    )
    (; coeffs, Y) = cache
    S = nstages(coeffs)
    H = dt
    tn = parent.t
    child = children[1]

    # Stage 1 is Y_1 = y_n at c_1 = 0: no fast solve, no implicit solve.
    Y[1] .= parent.uprev
    _eval_slow!(cache, parent, 1, tn)

    for i in 2:S
        _mri_stage_fast!(parent, child, cache, i, tn, H)
        parent.force_stepfail && return

        # parent.u now holds v_i(c_i H); add the explicit part of the implicit stage.
        _mri_stage_solve!(parent, cache, i, tn, H)
        parent.force_stepfail && return

        # The final stage's slow tendencies are never read, so they are not computed.
        # Ω is strictly lower triangular and Γ lower triangular, so stage i only ever
        # uses stages j < i -- the j = i term of Γ acts on the trial value inside the
        # nonlinear solve, not on a stored tendency -- and the embedding's ω̂_S and γ̂_S
        # are both zero, which the table constructor enforces. Skipping this saves two
        # full right-hand side evaluations per step.
        i < S && _eval_slow!(cache, parent, i, tn + coeffs.c[i] * H)
    end

    # First-same-as-last: the last stage *is* the step's solution.
    parent.u .= Y[S]

    if parent.controller_cache !== nothing
        _mri_embedded_step!(parent, child, cache, tn, H)
        parent.force_stepfail && return
        _mri_error_estimate!(parent, cache, dt)
    end
    return
end

# Evaluate and store the slow tendencies of stage `i`.
function _eval_slow!(cache::IMEXMRISRCache, parent, i, t)
    (; Y, fE, fI) = cache
    f = parent.f
    p = parent.p
    get_operator(f, 2)(fE[i], Y[i], p[2], t)
    get_operator(f, 3)(fI[i], Y[i], p[3], t)
    return nothing
end

# Solve the forced fast IVP of stage `i` over θ ∈ [0, c_i H], restarted from y_n.
function _mri_stage_fast!(parent, child, cache::IMEXMRISRCache, i, tn, H)
    (; coeffs, Y) = cache
    ci = coeffs.c[i]

    _set_stage_forcing!(cache, i, ci * H, tn, ci)

    # Stage restart: every stage begins from y_n at t = tn, so the child is re-anchored
    # there rather than continued. `rollback_child!` is exactly that re-anchoring, and
    # it also clears a transient failure left by an earlier stage.
    parent.u .= Y[1]
    idxs = parent.child_solution_indices[1]
    rollback_child!(child, parent.u, idxs, tn)

    return _advance_child!(parent, child, 1, ci * H)
end

# G^{k} = (1/c_i) Σ_{j<i} ω_{i,j}^{k} (f_j^{E} + f_j^{I})
function _set_stage_forcing!(cache::IMEXMRISRCache, i, scale, tn, ci)
    (; coeffs, forcing, fE, fI) = cache
    forcing.tn = tn
    forcing.scale = scale
    for k in eachindex(forcing.G)
        Gk = forcing.G[k]
        fill!(Gk, false)
        for j in 1:(i - 1)
            w = coeffs.Ω[k][i][j] / ci
            iszero(w) && continue
            @. Gk += w * (fE[j] + fI[j])
        end
    end
    return nothing
end

# Y_i = v_i(c_i H) + H Σ_{j≤i} γ_{i,j} f_j^{I}, the j = i term making it implicit.
function _mri_stage_solve!(parent, cache::IMEXMRISRCache, i, tn, H)
    (; coeffs, Y, fI, stage_params) = cache
    z = stage_params.z

    z .= parent.u
    for j in 1:(i - 1)
        γ = coeffs.Γ[i][j]
        iszero(γ) && continue
        @. z += (H * γ) * fI[j]
    end

    γii = coeffs.Γ[i][i]
    if iszero(γii)
        Y[i] .= z
        return nothing
    end

    stage_params.p = parent.p[3]
    stage_params.t = tn + coeffs.c[i] * H
    stage_params.γH = H * γii

    # `z` is the natural initial guess: the stage is Y_i = z + H γ_{i,i} f^{I}(t_i, Y_i),
    # so z is one fixed point iteration in. The previous stage would be a worse guess --
    # the abscissae are not sorted, so Y_{i-1} can sit further from Y_i than z does.
    SciMLBase.reinit!(cache.nlcache, z; p = stage_params)
    sol = SciMLBase.solve!(cache.nlcache)
    if !SciMLBase.successful_retcode(sol)
        _is_verbose(parent.opts.verbose) && @warn "IMEX-MRI-SR stage $i nonlinear solve \
            failed with retcode $(sol.retcode); rejecting the step."
        parent.force_stepfail = true
        return nothing
    end
    Y[i] .= sol.u
    return nothing
end

# The embedding of (1.5d): one fast solve over the whole step, forced by ω̂, followed by
# an explicit update. Both shipped methods have γ̂_S = 0, which the table check
# enforces, so no extra implicit solve is needed here.
function _mri_embedded_step!(parent, child, cache::IMEXMRISRCache, tn, H)
    (; coeffs, Y, fI, uembed) = cache
    S = nstages(coeffs)

    # ĉ = 1, so the interval is the whole step and there is no 1/c_i factor.
    _set_stage_forcing!(cache, S + 1, H, tn, one(H))

    parent.u .= Y[1]
    idxs = parent.child_solution_indices[1]
    rollback_child!(child, parent.u, idxs, tn)
    _advance_child!(parent, child, 1, H)
    parent.force_stepfail && return nothing

    uembed .= parent.u
    for j in 1:S
        γ = coeffs.Γ[S + 1][j]
        iszero(γ) && continue
        @. uembed += (H * γ) * fI[j]
    end

    # `_advance_child!` left the fast solve's end state in `parent.u`; the step's
    # solution has to be put back.
    parent.u .= Y[S]
    return nothing
end

function _mri_error_estimate!(parent, cache::IMEXMRISRCache, dt)
    (; uembed) = cache
    (; abstol, reltol, internalnorm) = parent.opts
    @. uembed = (parent.u - uembed) /
        (abstol + max(abs(parent.u), abs(parent.uprev)) * reltol)
    OrdinaryDiffEqCore.set_EEst!(parent, internalnorm(uembed, parent.t + dt))
    return nothing
end
