```@meta
EditURL = "../../literate/examples/06-popdyn.jl"
```

# Population dynamics with sin of gBm growth and step process harvest

This time we consider a population dynamics model with two types of noise, a geometric Brownian motion process affecting the growth rate and a point Poisson step process affecting the harvest.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \Lambda_t X_t (r - X_t) - \alpha H_t\frac{2X_t}{r + X_t}, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
with
```math
  \Lambda_t = \lambda(1 + \epsilon\sin(G_t))
```
where $\{G_t\}_{t\geq 0}$ is a geometric Brownian motion process and $\{H_t\}_{t \geq 0}$ is a point Poisson step process with Beta-distributed steps.

We fix $\lambda = 1.0$, $\epsilon = 0.3$, $r = 1.0$, and $\alpha = 0.5$. Notice the critical value for the bifurcation oscilates between $\lambda (1 - \epsilon) / 4$ and $\lambda (1 + \epsilon) / 4$, while the harvest term oscillates between 0 and $\alpha$, and we choose $\alpha = \lambda / 2$ so it oscillates below and above the critical value.

We choose a Beta distribution as the step law, with mean a little below $1/2$, so it stays mostly below the critical value, but often above it.

The geometric Brownian motion process is chosen with drift $\mu = 1$, diffusion $\sigma = 0.8$ and initial value $y_0 = 1.0$.

The Poisson counter for the point Poisson step process is chosen with rate 15.0, while the time interval is chosen with unit time span.

As for the initial condition, we also choose a Beta distribution, so it stays within the growth region, and with the same parameters as for the steps, just for the sake of simplicity.

We do not have an explicit solution for the equation so we use as target for the convergence an approximate solution via Euler method at a much higher resolution.

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 06-popdyn
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up the problem parameters.

We set the seed:

````@example 06-popdyn
rng = Xoshiro(123)
````

The right hand side of the evolution equation:

````@example 06-popdyn
function f(t, x, y)
    γ = 0.8
    ϵ = 0.3
    r = 1.0
    α = γ * r^2
    dx = x > zero(x) ? γ * (1 + ϵ * sin(y[1])) * x * (1 - x / r) - α * y[2] * x / (r + x) : zero(x)
    return dx
end
````

The time interval:

````@example 06-popdyn
t0, tf = 0.0, 1.0
````

The law for the initial condition:

````@example 06-popdyn
x0law = Beta(7.0, 5.0)
````

The noise parameters:

````@example 06-popdyn
μ = 1.0
σ = 0.8
y0 = 1.0
noise1 = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)

λ = 15.0
steplaw = Beta(5.0, 7.0)
noise2 = PoissonStepProcess(t0, tf, λ, steplaw)

noise = ProductProcess(noise1, noise2)
````

The mesh resolution:

````@example 06-popdyn
ntgt = 2^18
ns = 2 .^ (4:9)
nsample = ns[[1, 2, 3, 4]]
````

The number of samples for the Monte-Carlo estimate:

````@example 06-popdyn
m = 200
````

And add some information about the simulation:

````@example 06-popdyn
info = (
    equation = "population dynamics",
    noise = "gBm and step process noises",
    ic = "\$X_0 \\sim \\mathrm{Beta}($(x0law.α), $(x0law.β))\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 06-popdyn
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the [`ConvergenceSuite`](@ref):

````@example 06-popdyn
suite = ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors via [`solve`](@ref):

````@example 06-popdyn
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 06-popdyn
table = generate_error_table(result, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 06-popdyn
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide
````

### Plots

We illustrate the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 06-popdyn
plt = plot(result)
````

````@example 06-popdyn
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "order_popdyn_gBmPoisson.png")) # hide
nothing # hide
````

For the sake of illustration, we plot some approximations of a sample target solution:

````@example 06-popdyn
plt = plot(suite, ns=nsample)
````

````@example 06-popdyn
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "sample_popdyn_gBmPoisson.png")) # hide
nothing # hide
````

We can also visualize the noises associated with this sample solution:

````@example 06-popdyn
plot(suite, xshow=false, yshow=true, label=["Z_t" "H_t"], linecolor=:auto)
````

The gBm noises enters the equation via $G_t = \gamma(1 + \epsilon\sin(Z_t))$. Using the chosen parameters, this noise can be visualized below

````@example 06-popdyn
plot(suite, xshow=false, yshow= y -> 0.8 + 0.3sin(y[1]), label="\$G_t\$")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

