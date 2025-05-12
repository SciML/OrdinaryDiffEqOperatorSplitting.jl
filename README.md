# OrdinaryDiffEqOperatorSplitting.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sciml.github.io/OrdinaryDiffEqOperatorSplitting.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sciml.github.io/OrdinaryDiffEqOperatorSplitting.jl/dev/)
[![Build Status](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SciML/OrdinaryDiffEqOperatorSplitting.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/OrdinaryDiffEqOperatorSplitting.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

OrdinaryDiffEqOperatorSplitting.jl is a component package in the DifferentialEquations ecosystem. It holds
operator splitting solvers and utilities.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
OrdinaryDiffEqOperatorSplitting.jl in the standard way:

```julia
import Pkg;
Pkg.add("OrdinaryDiffEqOperatorSplitting");
```

## API

OrdinaryDiffEqOperatorSplitting.jl is part of the SciML common interface. The only requirement is that the user passes an algorithm
compatible to the corresponding function to `solve`. For example, we can solve a simple split problem using the `Euler()` algorithm
for each subproblem with the `LieTrotterGodunov` algorithm, by defining a problem tree and an analogue solver tree via tuples:

```julia
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqOperatorSplitting
# This is the true, full ODE.
function ode_true(du, u, p, t)
    du   .-= 0.1u
    du[1] -= 0.01u[3]
    du[3] -= 0.01u[1]
end

# This is the first operator of the ODE.
function ode1(du, u, p, t)
    @. du = -0.1u
end
f1 = ODEFunction(ode1)
f1dofs = [1,2,3]

# This is the second operator of the ODE.
function ode2(du, u, p, t)
    du[1] = -0.01u[2]
    du[2] = -0.01u[1]
end
f2 = ODEFunction(ode2)
f2dofs = [1,3]

# This defines the split of the ODE.
f = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))

# Next we can define the split problem.
u0 = [-1.0, 1.0, 0.0]
tspan = (0.0, 1.0)
prob = OperatorSplittingProblem(f, u0, tspan)

# And the time integration algorithm.
alg = LieTrotterGodunov(
    (Euler(), Euler())
)

# Right now OrdinaryDiffEqOperatorSplitting.jl does not implement the SciML solution interface,
# but we can only intermediate solutions via the iterator interface.
integrator = init(prob, alg, dt=0.1)
for (u, t) in TimeChoiceIterator(integrator, 0.0:0.5:1.0)
    @show t, u
end
```

## Available Solvers

For the list of available solvers, please refer to the [OrdinaryDiffEqOperatorSplitting.jl Solvers](https://sciml.github.io/OrdinaryDiffEqOperatorSplitting.jl/dev/api-reference/#Solvers) pages.
