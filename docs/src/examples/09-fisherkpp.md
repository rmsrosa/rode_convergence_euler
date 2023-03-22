```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/examples/09-fisherkpp.jl"
```

# Random Fisher-KPP partial differential equation

Here we simulate a Fisher-KPP equation with random boundary conditions, as inspired by the works of [Salako & Shen (2020)](https://doi.org/10.1007/s10884-020-09847-2) and [Freidlin & Wentzell (1992)](https://doi.org/10.1214/aop/1176989813)). The first work addresses the Fisher-KPP equation with a random reaction coefficient, while the second work considers more general reaction-diffusion equations but driven by random boundary conditions. Here, we consider the Fisher-KPP equation driven by random Neumann-type boundary conditions. The intent here is to illustrate the strong order 1 convergence rate on a nonlinear partial differential equation.

We use the method of lines (MOL), with finite differences in space, to approximate the random partial differential equation (PDE) by a system of random ODEs.

The equation is a nonlinear parabolic equation of reaction-diffusion type, modeling inhomogeneous population growth and displaying wave propagation. We force the system with random incoming/outcoming population fluctuations on the frontiers of the spatial domain.

## The equation

The equation takes the form

```math
  \frac{\partial u}{\displaystyle \partial t} = \frac{\partial^2 u}{\partial x^2} + u(1 - u), \quad (t, x) \in (0, \infty) \times (0, 1),
```
endowed with the boundary conditions

```math
  \frac{\partial u}{\partial x}(t, 0) = A_t, \quad \frac{\partial u}{\partial x}(t, 1) = - B_t
```

and a given a initial condition
```math
  u(0, x) = u_0(x).
```

The unknown $u(t, x)$ represents the population density at time $t$ and point $x$, relative to a given saturation value.

The initial condition is taken to be of the form
```math
  u_0(x) = 2(x-1)^2(x + 1/2) = 2x^3 -3x^2 + 1,
```
so that
```math
  u_0(0) = 1, \quad u_0(1) = 0, \quad u_0'(0) = u_0'(1) = 0.
```

````@example 09-fisherkpp
using Plots # hide
plot(0.0:0.01:1.0, x -> 2x^3 - 3x^2 + 1, title="Initial condition", titlefont=8, legend=nothing) # hide
````

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 09-fisherkpp
using Plots
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
using BenchmarkTools
````

Then we set up some variables as usual, starting with the random seed.

````@example 09-fisherkpp
rng = Xoshiro(123)
````

The time interval

````@example 09-fisherkpp
t0, tf = 0.0, 2.0
````

The discretization in space is made with `l` mesh points $x_j = (j-1) / (l-1)$, for $j = 1, \ldots, l$, corresponding to `l-1` mesh intervals of length $\Delta x = 1 / (l-1)$. The points $x_1 = 0$ and $x_l = 1$ are the boundary points. We set `l` to

````@example 09-fisherkpp
l = 16
````

Notice that for the target solution we need a very fine *time* mesh, on top of having to repeat the simulation a number of times for the Monte-Carlo estimate. This is computationally demanding for large `l`, so we choose a moderate number just for illustration purpose.

The initial condition is

````@example 09-fisherkpp
u₀(x) = 2x^3 - 3x^2 + 1
````

The discretized initial condition is then

````@example 09-fisherkpp
u0law = product_distribution(Tuple(Dirac(u₀((j-1) / (l-1))) for j in 1:l)...)

plot(title="Discretized initial condition", titlefont=8, xlabel="\$x\$", ylabel="\$u\$")
plot!(0.0:0.01:1.0, u₀, label="initial condition")
scatter!((0:l-1) ./ (l-1), u₀, label="discretization")
````

For the discretization of the equation we use finite differences with a centered second-order discretization of the second derivative:

```math
  \frac{\partial^2 u}{\partial x^2}(t, x_j) \approx \frac{u(t, x_{j+1}) - 2u(t, x_j) + u(t, x_{j-1})}{\Delta x^2}, \quad j = 1, \ldots, l
```

Notice this goes up to the boundary points $j=1$ and $j=l$, corresponding to $x=0$ and $x=1$, and depends on the "ghost" points $x_0 = -\Delta x$ and $x_{l+1} = 1 + \Delta x$. These points are solved for by using the Neumann boundary conditions and a centered second-order finite difference approximation of the first derivative, namely

```math
  \frac{\partial u}{\partial x}(t, x_j) = \frac{u(t, x_{j+1} - u(t, x_{j-1}))}{\Delta x},
```

on the boundary points $j=1$ and $j=l$, so that

```math
  u(t, x_0) = u(t, x_2) - A_t \Delta x, \qquad u(t, x_{l}) = u(t, x_{l-2}) + B_t \Delta x.
```

These points are plugged into the second-order approximation of the second derivatives at the boundary points.

This yields the following in-place formulation for the right-hand side of the MOL Random ODE approximation of the Random PDE.

````@example 09-fisherkpp
function f!(du, t, u, y)
    axes(u, 1) isa Base.OneTo || error("indexing of `x` should be Base.OneTo")

    l = length(u)
    dx = 1.0 / (l - 1)
    dx² = dx ^ 2
````

interior points

````@example 09-fisherkpp
    for j in 2:l-1
        du[j] = (u[j-1] - 2u[j] + u[j+1]) / dx² + u[j] * (1.0 - u[j])
    end
````

ghost points

````@example 09-fisherkpp
    gh1 = u[2] - 2dx * y[1]
    gh2 = u[l-1] - 2dx * y[2]
````

boundary points

````@example 09-fisherkpp
    du[1] = ( u[2] - 2u[1] + gh1 ) / dx² + u[1] * ( 1.0 - u[1] )
    du[l] = ( gh2 - 2u[l] + u[l-1] ) / dx² + u[l] * ( 1.0 - u[l] )
    return nothing
end
````

Now we make sure this is non-allocating. We use a finer spatial mesh for testing.

````@example 09-fisherkpp
xx = 0.0:0.01:1.0
u = sin.(π * xx) .^ 2
du = similar(u)
y = [0.0, 0.0]
t = 0.0
f!(du, t, u, y)

plot(xx, u, label="u")
plot!(xx, du, label="du/dt")
````

````@example 09-fisherkpp
@btime f!($du, $t, $u, $y)
nothing # hide
````

The noise is an approximation of a white noise

````@example 09-fisherkpp
y0 = 0.0
ν = 200.0 # = 1 / 0.005 => time-scale = 0.005
σ = 10.0 # variance σ^2 / 2ν = 0.25
noise = ProductProcess(OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ), OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ))
````

````@example 09-fisherkpp
ntgt = 2^22
ns = 2 .^ (9:11)

ntgt = 2^15 * 3^3 * 5
ns = [2^10, 2^7 * 3^2, 2^8 * 5, 2^9 * 3, 2^7 * 3 * 5, 2^11]
all(mod(ntgt, n) == 0 for n in ns)
ntgt ≥ last(ns)^2
m = 1_000
````

And add some information about the simulation:

````@example 09-fisherkpp
info = (
    equation = "Fisher-KPP equation",
    noise = "Orstein-Uhlenbeck modulated by a transport process",
    ic = "\$X_0 = 2(x - 1)^2(x + 1/2)\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 09-fisherkpp
target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))
````

### Order of convergence

With all the parameters set up, we build the convergence suite:

````@example 09-fisherkpp
suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors:

````@example 09-fisherkpp
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 09-fisherkpp
table = generate_error_table(result, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 09-fisherkpp
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")
````

### Plots

We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 09-fisherkpp
plot(result)
````

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide

For the sake of illustration, we plot a sample of an approximation of a target solution:

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

