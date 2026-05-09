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
second-order accuracy for two operators $F_1$ and $F_2$ by performing

```math
\begin{aligned}
    \text{Solve} \quad d_t u^1(t) &= F_1(u^1(t), p, t) & & \quad \text{on} \; [t_0, t_0 + \Delta t/2] \; \text{with} \; u^1(t_0) = u_0 \\
    \text{Solve} \quad d_t u^2(t) &= F_2(u^2(t), p, t) & & \quad \text{on} \; [t_0, t_0 + \Delta t] \; \text{with} \; u^2(t_0) = u^1(t_0 + \Delta t/2) \\
    \text{Solve} \quad d_t u^3(t) &= F_1(u^3(t), p, t) & & \quad \text{on} \; [t_0 + \Delta t/2, t_0 + \Delta t] \; \text{with} \; u^3(t_0 + \Delta t/2) = u^2(t_0 + \Delta t)
\end{aligned}
```

yielding $u(t_0 + \Delta t) \approx u^3(t_0 + \Delta t)$.

### Analysis of Strang-Marchuk

For two bounded linear operators $L_1$ and $L_2$ the Strang-Marchuk approximation
reads

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
of $O(t^3)$ and hence second-order global accuracy.

## References

```@bibliography
Pages = ["time-integration.md"]
Canonical = false
```
