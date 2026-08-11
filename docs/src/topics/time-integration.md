## [Operator Splitting Theory](@id theory_operator-splitting)

For operator splitting procedures we assume that we have some time-dependent
problem with initial condition $u_0 := u(t_0)$ and an operator $F$ describing
the right hand side. We assume that $F$ can be additively split into $N$
suboperators $F_i$. This can be formally written as

```math
d_t u(t) = F(u(t), p, t) = F_1(u(t), p, t) + ... + F_N(u(t), p, t) \, .
```

We call $t$ time and $u(t)$ the *state* of the system. This way we can
define subproblems

```math
\begin{aligned}
    d_t u(t) &= F_1(u(t), p, t) \\
             & \vdots \\
    d_t u(t) &= F_N(u(t), p, t)
\end{aligned}
```

Now, the key idea of operator splitting methods is that solving the subproblems
can be easier, and hopefully more efficient, than solving the full problem.
Arguably the easiest algorithm to advance the solution from $t_0$ to some time
point $t_1 > t_0$ is the Lie-Trotter-Godunov operator splitting [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
Here the subproblems are solved consecutively, where the solution of one
subproblem is taken as the initial guess for the next subproblem, until we have
solved all subproblems. In this case we have constructed an _approximation_
for $u(t_1)$.

More formally we can write the Lie-Trotter-Godunov scheme [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite) as follows:

```math
\begin{aligned}
    \text{Solve} \quad d_t u^1(t) &= F_1(u^1(t), p, t) & & \quad \text{on} \; [t_0, t_1] \; \text{with} \; u^1(t_0) = u_0 \\
    \text{Solve} \quad d_t u^2(t) &= F_2(u^2(t), p, t) & & \quad \text{on} \; [t_0, t_1] \; \text{with} \; u^2(t_0) = u^1(t_1) \\
             & \vdots & & \\
    \text{Solve} \quad d_t u^N(t) &= F_N(u^N(t), p, t) & & \quad \text{on} \; [t_0, t_1] \; \text{with} \; u^N(t_0) = u^{N-1}(t_1)
\end{aligned}
```

Such that we obtain the approximation $u(t_1) \approx u^{N-1}(t_1)$. The
approximation is first order in time, as we will show in the next section.

Probably the most widely spread application for operator splitting schemes is
the solution of reaction diffusion systems. These have the form

```math
d_t u(t) = Lu + R(u)
```

where $L$ is some linear operator, usually coming from the linearization of
diffusion operators and a nonlinear reaction part $R$ which has some interesting
locality properties. This locality property usually tells us that the time
evolution of $R$ naturally decouples into many small blocks. This way we only
have to solve for the time evolution of a linear problem $d_t u(t) = Lu$ and a
set of many very small nonlinear problems $d_t u(t) = R(u)$.

### Analysis of Lie-Trotter-Godunov

It should be noted that even if we solve all subproblems analytically, then
operator splitting schemes themselves almost always come with their own
approximation error, which is simply called the splitting error. For linear
problems this error can vanish if all suboperators $F_i$ commute, i.e. if
$F_j \cdot F_i = F_i \cdot F_j$ for all $1 \leq i,j \leq N$, which can be shown
with the Baker-Campbell-Hausdorff formula. Let us investigate the convergence
order for two bounded linear operators $L_1$ and $L_2$, i.e. on the following
system of ODEs

```math
d_t u = L_1 u + L_2 u \, .
```

Here the exact solution $u$ at time point $t$ for some initial condition at $t_0 = 0$ is

```math
u(t) = e^{(L_1 + L_2)t} u_0 \, ,
```

while the solution for the Lie-Trotter-Godunov scheme is

```math
\tilde{u}(t) = e^{L_1t}e^{L_2t} u_0 \, .
```

The local truncation error can be written as

```math
\epsilon(t) = ||e^{L_1t}e^{L_2t} - e^{(L_1 + L_2)t}|| \, ||u_0||
```

if we now replace the exponentials with their definitions we obtain for the first norm

```math
\begin{aligned}
&||(I + tL_1 + \frac{h^2}{2}L_1^2 + ...)(I + tL_2 + \frac{h^2}{2}L_2^2 + ...) - (I + t(L_1 + L_2) + \frac{h^2}{2}(L_1+L_2)^2 + ...)||\\
=& ||\frac{h^2}{2} (L_1 L_2 - L_2 L_1) + ... || \leq \frac{h^2}{2} || (L_1 L_2 - L_2 L_1) || + O(h^3)
\end{aligned}
```

This shows that the local truncation error is O(h^2) and hence the scheme is first order accurate.

Showing stability is also straight forward. We assumed that $L_1$ and $L_2$ are
bounded, so we obtain for all time points $t' < t$ and all repeated subdivisions
$n \in \mathbb{N}$ the following bound

```math
||(e^{L_1\frac{t'}{n}}e^{L_2\frac{t'}{n}})^n||
\leq ||e^{L_1\frac{t'}{n}}e^{L_2\frac{t'}{n}}||^n
\leq ||e^{L_1\frac{t'}{n}}||^n ||e^{L_2\frac{t'}{n}}||^n
\leq e^{||L_1||t'} e^{||L_2||t'}
\leq e^{||L_1||t} e^{||L_2||t}
\leq C < \infty
```

which implies stability of the scheme.

### Strang-Marchuk Splitting

A natural way to improve the accuracy of operator splitting is to symmetrize the
scheme. The Strang-Marchuk splitting [Str:1968:ccd,Mar:1971:tsm](@cite) achieves
second-order accuracy for $N$ operators by performing a palindromic sweep

```math
F_1(\Delta t/2) \to \cdots \to F_{N-1}(\Delta t/2) \to F_N(\Delta t) \to F_{N-1}(\Delta t/2) \to \cdots \to F_1(\Delta t/2)
```

More formally, for the simplest case of two operators $F_1$ and $F_2$

```math
\begin{aligned}
    \text{Solve} \quad d_t u^1(t) &= F_1(u^1(t), p, t) & & \quad \text{on} \; [t_0, t_0 + \Delta t/2] \; \text{with} \; u^1(t_0) = u_0 \\
    \text{Solve} \quad d_t u^2(t) &= F_2(u^2(t), p, t) & & \quad \text{on} \; [t_0, t_0 + \Delta t] \; \text{with} \; u^2(t_0) = u^1(t_0 + \Delta t/2) \\
    \text{Solve} \quad d_t u^3(t) &= F_1(u^3(t), p, t) & & \quad \text{on} \; [t_0 + \Delta t/2, t_0 + \Delta t] \; \text{with} \; u^3(t_0 + \Delta t/2) = u^2(t_0 + \Delta t)
\end{aligned}
```

yielding $u(t_0 + \Delta t) \approx u^3(t_0 + \Delta t)$.

### Analysis of Strang-Marchuk

We show the second-order accuracy for two bounded linear operators $L_1$ and
$L_2$. The Strang-Marchuk approximation reads

```math
\tilde{u}(t) = e^{L_1 t/2} \, e^{L_2 t} \, e^{L_1 t/2} \, u_0 \, .
```

Expanding the exponentials:

```math
\begin{aligned}
e^{L_1 t/2} \, e^{L_2 t} \, e^{L_1 t/2}
&= \bigl(I + \tfrac{t}{2}L_1 + \tfrac{t^2}{8}L_1^2 + \cdots\bigr)
   \bigl(I + t L_2 + \tfrac{t^2}{2}L_2^2 + \cdots\bigr)
   \bigl(I + \tfrac{t}{2}L_1 + \tfrac{t^2}{8}L_1^2 + \cdots\bigr) \\
&= I + t(L_1 + L_2) + \tfrac{t^2}{2}(L_1 + L_2)^2 + O(t^3)
\end{aligned}
```

which matches the Taylor expansion of $e^{(L_1+L_2)t}$ through the $t^2$ term.
The symmetry of the scheme causes the first-order commutator term
$[L_1, L_2] = L_1 L_2 - L_2 L_1$ to cancel, leaving a local truncation error
of $O(t^3)$ and hence second-order global accuracy. The same argument extends to
the general $N$-operator palindromic scheme.

## [Higher order splittings](@id theory_higher-order)

Lie-Trotter-Godunov and Strang-Marchuk are both instances of a more general
construction. A splitting scheme applies the sub-flows in a fixed sequence, each for
a fixed fraction of the step, so for $N$ operators it is completely described by an
$S \times N$ table of coefficients $a_{ji}$: stage $j$ advances operator $i$ by
$a_{ji} \Delta t$. In the linear two-operator case,

```math
\mathcal{S}(\Delta t) = \prod_{j=1}^{S} e^{a_{jN} \Delta t L_N} \cdots e^{a_{j1} \Delta t L_1} \, .
```

Lie-Trotter-Godunov is the one-stage table $a = (1, 1)$ and Strang-Marchuk is the
two-stage table $a = \bigl((\tfrac{1}{2}, 1), (\tfrac{1}{2}, 0)\bigr)$.

Each operator's coefficients must sum to one,

```math
\sum_{j=1}^{S} a_{ji} = 1 \quad \text{for every } i \, ,
```

since otherwise the scheme does not even advance every sub-problem by $\Delta t$.
This is the consistency condition, and it is the only one this package checks when a
table is constructed. Attaining order $p$ imposes further conditions, one per
independent commutator up to order $p$, obtained by matching the
Baker-Campbell-Hausdorff expansion of the product above against that of
$e^{\Delta t (L_1 + L_2)}$ — the same computation as in the two analyses above,
carried further.

### Composition: the triple jump

Solving the order conditions directly gets unpleasant quickly. A cheaper route is
*composition*: build a higher-order scheme out of a symmetric one of lower order.
If $\mathcal{S}_2$ is any symmetric second-order scheme, then

```math
\mathcal{S}_4(\Delta t) = \mathcal{S}_2(w_1 \Delta t) \, \mathcal{S}_2(w_0 \Delta t) \, \mathcal{S}_2(w_1 \Delta t)
```

is symmetric for any weights, and hence of even order. It is of order four as soon
as the weights satisfy

```math
2 w_1 + w_0 = 1 \, , \qquad 2 w_1^3 + w_0^3 = 0 \, ,
```

the first being consistency and the second the cancellation of the third-order term.
The real solution is $w_1 = 1/(2 - 2^{1/3})$ and $w_0 = -2^{1/3} w_1$, giving
Yoshida's "triple jump" [Yos:1990:cho](@cite), implemented here as
[`Yoshida4`](@ref). Writing the three Strang steps out as a flat sequence of flows
and merging the adjacent flows of the same operator that the composition leaves next
to each other collapses nine flows to eight — which is exactly the four-stage table
`Yoshida4` carries, its last stage having a zero second coefficient.

### The order barrier and negative coefficients

Note that $w_0 < 0$ above. This is not an artifact of the construction: no splitting
scheme of order greater than two has all coefficients positive
[She:1989:slp,Suz:1991:gtf](@cite). Any third- or higher-order splitting therefore
integrates some sub-problem *backward in time* during part of every step, which has
two practical consequences.

First, the sub-problems must admit a backward flow. For a parabolic sub-problem —
diffusion, say — the backward evolution is ill-posed and the negative sub-steps are
violently unstable, so on a reaction-diffusion system the higher-order schemes here
are not usable on the diffusion operator, however attractive their order. This is
the reason Strang-Marchuk remains the workhorse despite being only second order.

Second, the implementation has to actually run its sub-integrators backwards. An
inner integrator fixes its direction of integration at construction, so a negative
sub-step temporarily reverses it; see the developer documentation for the details.

### Adjoint pairs

The *adjoint* of a scheme is

```math
\mathcal{S}^*(\Delta t) = \mathcal{S}(-\Delta t)^{-1} \, ,
```

which for a splitting scheme is simply its whole sequence of flows run in reverse
order, every coefficient keeping its sign and its operator. A scheme is symmetric
exactly when $\mathcal{S}^* = \mathcal{S}$, which is why Strang-Marchuk — a
palindrome — gains an order over Lie-Trotter-Godunov.

If $\mathcal{S}$ has order $p$ with leading local error $C \Delta t^{p+1}$, then
$\mathcal{S}^*$ has the same order with leading error $(-1)^p C \Delta t^{p+1}$. For
**odd** $p$ the two signs oppose, so running the pair from the same initial value
gives, at twice the cost of one scheme,

```math
\frac{\mathcal{S} + \mathcal{S}^*}{2} \quad \text{of order } p+1 \, ,
\qquad
\frac{\mathcal{S} - \mathcal{S}^*}{2} \quad \text{an estimate of the local error of } \mathcal{S} \, ,
```

the latter being asymptotically correct as $\Delta t \to 0$
[AuzHofKetKoc:2017:psm](@cite). This is the Milne device applied to a scheme and its
adjoint, and it is what makes the splitting error itself estimable and hence the
splitting step adaptive — see [Adaptive time stepping](@ref). The construction is
[`AdjointPair`](@ref); at $p = 1$, with Lie-Trotter-Godunov as the base, it is the
pair of mutually reversed sequences implemented directly as
[`PalindromicPairLieTrotterGodunov`](@ref).

For even $p$ the two leading terms are *equal* rather than opposite: averaging
cancels nothing and the difference is not an error estimate, which is why
[`AdjointPair`](@ref) rejects an even-order base scheme.

## [Implicit-explicit multirate methods](@id theory_imex-multirate)

Everything above splits the *state*: each operator owns a slice of the solution vector
and a step is a sequence of flows, coupled only through the initial condition each flow
is handed. That weak coupling is what limits Lie-Trotter-Godunov and Strang-Marchuk to
first and second order no matter how accurately the sub-problems are solved
[FisReyRob:2023:iem](@cite).

The IMEX-MRI-SR methods of [FisReyRob:2023:iem](@cite) break out of that barrier by
splitting the *right-hand side* instead. One additive partition,

```math
y'(t) = f^{\{F\}}(t, y) + f^{\{E\}}(t, y) + f^{\{I\}}(t, y),
```

separates rapidly evolving dynamics ($F$) from slow dynamics that are in turn split in
an implicit-explicit fashion: $f^{\{I\}}$ is stiff and solved implicitly, $f^{\{E\}}$ is
non-stiff and treated explicitly. All three act on the *whole* state.

A step from $t_n$ to $t_n + H$ evolves a sequence of *forced* fast initial value
problems, each restarted from $y_n$ — hence "stage restart" — and follows each with one
implicit solve at the slow time scale:

```math
\begin{aligned}
Y_1 &= y_n, \\
v_i'(\theta) &= f^{\{F\}}(t_n + \theta, v_i(\theta)) + g_i(\theta),
  \quad \theta \in [0, c_i H], \quad v_i(0) = y_n, \\
Y_i &= v_i(c_i H) + H \sum_{j \le i} \gamma_{i,j} f_j^{\{I\}},
\end{aligned}
```

with $y_{n+1} = Y_{s}$, and the forcing built from the slow tendencies of the previous
stages,

```math
g_i(\theta) = \frac{1}{c_i} \sum_{j < i}
  \omega_{i,j}\!\left(\frac{\theta}{c_i H}\right)
  \left(f_j^{\{E\}} + f_j^{\{I\}}\right).
```

The forcing is what supplies the strong coupling: information from the slow operators
enters the fast sub-problem throughout its evolution, not merely through its initial
condition. Because $\omega_{i,j}$ is a polynomial in the normalized fast time, the sum
over stages collapses once per stage into one vector per power, so a fast right-hand side
evaluation stays cheap regardless of the stage count.

The fast sub-problem is solved by any ordinary `OrdinaryDiffEq` algorithm, given its own
step size through the per-node `dt` described under
[Multi-rate integration](@ref); the paper's experiments use $h = H/10$. The slow implicit
stages are nonlinear systems, solved by the algorithm's `nlsolve`.

Two methods are provided, [`IMEXMRISR2`](@ref) and [`IMEXMRISR3`](@ref), of order two
and three. Both carry an embedding one order lower, which gives a genuine local error
estimate and hence adaptivity — unlike the splitting schemes, this needs no second pass
over the step, only one extra fast solve over $[0, H]$. The paper's fourth order method
is deliberately not included: its joint stability region is empty and its embedding
gives poor estimates, which the authors report as getting "stuck" oscillating between
accepted and rejected steps.

Because the partition is additive rather than state-disjoint, these algorithms expect a
[`GenericSplitFunction`](@ref) whose three operators — in the order
$f^{\{F\}}, f^{\{E\}}, f^{\{I\}}$ — each span the whole state:

```julia
f = GenericSplitFunction(
    (f_fast, f_explicit, f_implicit),
    (1:n, 1:n, 1:n),      # additive: every operator sees the whole state
)

dt = TreeOption(f, H)
dt[f[1]] = H / 10          # the fast sub-problem's step size

solve(OperatorSplittingProblem(f, u0, tspan), IMEXMRISR3(BS3()); dt, adaptive = true)
```

## References

```@bibliography
Pages = ["time-integration.md"]
Canonical = false
```
