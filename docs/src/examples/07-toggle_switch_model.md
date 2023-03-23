```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/examples/07-toggle_switch_model.jl"
```

# A toggle-switch model for gene expression with compound Poisson external activation process

Here, we consider the toggle-switch model in Section 7.8 of [Asai (2016)](https://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/40146), originated from [Verd, Crombach & Jaeger (2014)](https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-8-43). See also [Strasser, Theis & Marr (2012)](https://doi.org/10.1016/j.bpj.2011.11.4000).

Toogle switches in gene expression consist of genes that mutually repress each other and exhibit two stable steady states of ON and OFF gene expression. It is a regulatory mechanism which is active during cell differentiation and is believed to act as a memory device, able to choose and maintain cell fate decisions.

## The equation

We consider the following simple model as discussed in [Asai (2016)](https://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/40146), of two interacting genes, $X$ and $Y$, with the concentration of their corresponding protein products at each time $t$ denoted by $X_t$ and $Y_t$. These are stochastic processes defined by the system of equations
```math
  \begin{cases}
  \frac{\displaystyle \mathrm{d}X_t}{\displaystyle \mathrm{d} t} = \left( A_t + \frac{\displaystyle X_t^4}{\displaystyle a^4 + X_t^4}\right)\left(\frac{\displaystyle b^4}{\displaystyle b^4 + Y_t^4}\right) - \mu X_t, \\ \\
  \frac{\displaystyle \mathrm{d}Y_t}{\displaystyle \mathrm{d} t} = \left( B_t + \frac{\displaystyle Y_t^4}{\displaystyle c^4 + Y_t^4}\right)\left(\frac{\displaystyle d^4}{\displaystyle d^4 + X_t^4}\right) - \nu Y_t, \\ \\
  \left. X_t \right|_{t = 0} = X_0, \\ \\
  \left. Y_t \right|_{t = 0} = Y_0,
  \end{cases}
```

where $\{A_t\}_{t\geq 0}$ and $\{B_t\}_{t\geq 0}$ are given stochastic process representing the external activation on each gene; $a$ and $c$ determine the auto-activation thresholds; $b$ and $d$ determine the thresholds for mutual repression; and $\mu$ and $\nu$ are protein decay rates. In this model, the external activations $A_t$ and $B_t$ are taken to be two independent compound Poisson processes.

In the simulations below, we use the same parameters as in [Asai (2016)](https://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/40146): We fix $a = c = 0.25$; $b = d = 0.4$; and $\mu = \nu = 1.25$. The initial conditions are set to $X_0 = Y_0 = 10.0$. The external activations are compound Poisson process with Poisson rate $\lambda = 5.0$ and jumps uniformly distributed on $[0.0, 0.5]$.


We don't have an explicit solution for the equation so we just use as target for the convergence an approximate solution via Euler method at a much higher resolution.


## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 07-toggle_switch_model
using Plots
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
````

Then we define the random seed

````@example 07-toggle_switch_model
rng = Xoshiro(123)
````

The evolution law

````@example 07-toggle_switch_model
function f!(dx, t, x, y)
    a⁴ = c⁴ = 0.25 ^ 4
    b⁴ = d⁴ = 0.4 ^ 4
    μ = ν = 1.25
    α, β = y
    x₁⁴ = x[1]^4
    x₂⁴ = x[2]^4
    dx[1] = ( α + x₁⁴ / (a⁴  + x₁⁴) ) * ( b⁴ / ( b⁴ + x₂⁴)) - μ * x[1]
    dx[2] = ( β + x₂⁴ / (c⁴  + x₂⁴) ) * ( d⁴ / ( d⁴ + x₁⁴)) - ν * x[1]
    return dx
end
````

The time interval

````@example 07-toggle_switch_model
t0, tf = 0.0, 4.0
````

The law for the initial condition

````@example 07-toggle_switch_model
x0 = 10.0
y0 = 10.0
x0law = product_distribution(Dirac(x0), Dirac(y0))
````

The compound Poisson processes for the source terms

````@example 07-toggle_switch_model
λ = 5.0
ylaw = Uniform(0.0, 0.5)
noise = ProductProcess(CompoundPoissonProcess(t0, tf, λ, ylaw), CompoundPoissonProcess(t0, tf, λ, ylaw))
````

The resolutions for the target and approximating solutions, as well as the number of simulations for the Monte-Carlo estimate of the strong error

````@example 07-toggle_switch_model
ntgt = 2^18
ns = 2 .^ (4:9)
nsample = ns[[1, 2, 3, 4]]
m = 1_000
````

And add some information about the simulation:

````@example 07-toggle_switch_model
info = (
    equation = "toggle-switch model of gene regulation",
    noise = "compound Poisson process noises",
    ic = "\$X_0 = $x0; Y_0 = $y0\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 07-toggle_switch_model
target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))
````

### Order of convergence

With all the parameters set up, we build the [`ConvergenceSuite`](@ref):

````@example 07-toggle_switch_model
suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors via [`solve`](@ref):

````@example 07-toggle_switch_model
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 07-toggle_switch_model
table = generate_error_table(result, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 07-toggle_switch_model
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")
nothing # hide
````

### Plots

We plot the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 07-toggle_switch_model
plt = plot(result)
````

And we save the convergence plot for inclusion in the article.

````@example 07-toggle_switch_model
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "order_toggleswitch.png"))
nothing # hide
````

For the sake of illustration of the behavior of the system, we rebuild the problem with a longer time step and do a single run with it, for a single sample solution.

````@example 07-toggle_switch_model
t0, tf = 0.0, 10.0
m = 1
ns = 2 .^ (6:9)
suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
@time result = solve(rng, suite)
nothing # hide
````

We visualize the pair of solutions

````@example 07-toggle_switch_model
plt = plot(suite, xshow=1, ns=nothing, label="X_t", linecolor=1)
plot!(plt, suite, xshow=2, ns=nothing, label="Y_t", linecolor=2, ylabel="concentration")
````

and save it for display in the article

````@example 07-toggle_switch_model
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "evolution_toggleswitch.png"))
nothing # hide
````

We also illustrate the convergence to say the expression of the X gene:

````@example 07-toggle_switch_model
plt = plot(suite)
````

and save it:

````@example 07-toggle_switch_model
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "approximation_toggleswitch.png"))
nothing # hide
````

We can also visualize the noises associated with this sample solution:

````@example 07-toggle_switch_model
plot(suite, xshow=false, yshow=true, label=["\$A_t\$" "\$B_t\$"])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
