# Adaptive time stepping

Two independent layers of a splitting tree can adapt their step sizes:

- **Leaf solvers** adapt their own internal steps within each sub-solve, exactly as
  they would outside of this package. Their behavior is governed by the tolerances
  and options they receive.
- **Splitting nodes** adapt the splitting step itself. This requires an algorithm
  that produces an error estimate of the splitting error;
  [`PalindromicPairLieTrotterGodunov`](@ref) is currently the only one. It advances
  with the average of the two mutually reversed Lie-Trotter sequences and uses half
  their difference as the local error estimate.

By default every node adapts exactly if its own algorithm can: for

```julia
alg = PalindromicPairLieTrotterGodunov((Tsit5(), Euler()))
integrator = init(prob, alg; dt = 0.1)
```

the splitting node runs its step size controller, the `Tsit5` leaf adapts its inner
steps, and the `Euler` leaf steps fixed. Passing `adaptive = false` (or a
per-node [`TreeOption`](@ref)) overrides this.

## The two tolerance layers

`abstol` and `reltol` are understood at *every* node:

- At a **splitting node** they scale the splitting error estimate: the node accepts
  a step when `‖(u₁ - u₂)/2 ./ (abstol .+ max.(|u|) .* reltol)‖ ≤ 1`, where
  `u₁, u₂` are the results of the two sequences of the palindromic pair.
- At a **leaf** they control the accuracy of the inner solves.

A scalar passed to `init` configures the whole tree with the same value. A
[`TreeOption`](@ref) configures each node individually:

```julia
reltol = TreeOption(f, 1.0e-8)   # splitting node(s): tight
reltol[1] = 1.0e-10              # leaf 1: tighter still
reltol[2] = 1.0e-10              # leaf 2: tighter still
integrator = init(prob, alg; dt, reltol)
```

## Rule: inner tolerances must be tighter than the splitting tolerances

The palindromic error estimator measures the **difference of two inner-solved
sequences**. Whatever error the inner solvers commit enters that difference as
noise the estimator cannot distinguish from splitting error. Two consequences:

1. **The estimator is blind to inner error.** With coarse fixed-step inner solvers
   (say `Euler()` stepping at the splitting step size) the overall method degrades
   to the inner order while the controller happily reports small `EEst`.
2. **Loose inner tolerances choke the controller.** When a sub-problem has fast
   internal dynamics (the typical reason for splitting it off), its solver commits
   error on the order of its tolerance in *every* sub-solve, independent of the
   splitting step size. If that exceeds the splitting tolerance — e.g. an inner
   `abstol` 10× *larger* than the splitting `abstol` — then `EEst > 1` no matter
   how small the splitting `dt` becomes. The controller keeps rejecting and
   shrinking `dt` without the estimate improving, until the solve aborts with
   `ReturnCode.DtLessThanMin` (or, with a tiny `dtmin`, grinds down to absurdly
   small splitting steps).

As a rule of thumb, keep the leaf tolerances at least **one to two orders of
magnitude tighter** than the splitting tolerances (adaptive leaves), or the fixed
inner steps well below the splitting step. The safe default is the scalar spread —
identical tolerances everywhere are already borderline; never configure leaves
*looser* than their splitting node.

## Symptoms and causes

| Symptom | Likely cause |
|---|---|
| `ReturnCode.DtLessThanMin`, `dt` collapsed, solution up to that point looks fine | Inner tolerances looser than the splitting tolerances (noise floor in the estimator), or genuinely unreachable tolerances |
| Splitting `dt` grows to the full interval immediately | The operators (nearly) commute, the splitting error is ≈ 0; harmless — the inner solvers carry the accuracy |
| Result visibly less accurate than the splitting tolerances suggest | Inner solves under-resolved: the estimator cannot see inner error (see rule above) |
| A few rejections right after `init` or `reinit!` | Initial `dt` too large for the tolerance; harmless, the controller recovers |
| Immediate abort with `DtLessThanMin` although tolerances look consistent | A scalar `dtmin` travels to the leaves too and may forbid the sub-steps they need. Restrict it to the splitting node with a `TreeOption` (`dtmin = TreeOption(f, 0.0); dtmin[] = 1e-3`) |

## Controllers

An adaptive splitting node runs an `OrdinaryDiffEqCore` step size controller
(default: `IController`). The standard knobs (`qmin`, `qmax`, `gamma`,
`qsteady_min`, `qsteady_max`, `failfactor`) can be passed to `init` — they are
folded into the default controller — or a controller object can be passed
explicitly via `controller` (per node via a `TreeOption`), e.g. a `PIController`
whose memory smooths the step size sequence.

## Failure handling

A failing *adaptive* node (leaf or splitting node) is fatal: it already exhausted
its own step size adaptation, and its return code propagates to the root. A failing
*non-adaptive* node escalates the failure to the nearest adaptive ancestor, which
rolls the whole subtree back and retries with a `failfactor`-shrunken step —
shrinking the effective step of every non-adaptive descendant — until it either
succeeds or falls below `dtmin`. Without any adaptive ancestor the integration
stops with the escalated return code.
