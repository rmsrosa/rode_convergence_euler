```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/examples/04-popdyn_gBmPoisson.jl"
```

# Population dynamics with sin(gBm) growth and step process harvest

Now we consider a population dynamics model with a coupled noise term, a geometric Brownian motion process affecting the growth and a point Poisson step process affecting the harvest

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \lambda(1 + \epsilon\sin(G_t)) X_t (r - X_t) - \alpha H_t, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{G_t\}_{t\geq 0}$ is a geometric Brownian motion process and $\{H_t\}_{t \geq 0}$ is a point Poisson step process.

For the sake of simplicity, we fix $\lambda = 10.0$, $\epsilon = 0.3$, $r = 1.0$, and $\alpha = 0.05$. Notice the critical value for the bifurcation oscilates between $\lambda (1 - \epsilon) / 4$ and $\lambda (1 + \epsilon) / 4$, while the harvest term oscillates between 0 and $\alpha$, so we choose $\alpha = \lambda / 2$ so it oscillates below and above the critical value.
More precisely, we choose a Beta distribution as the step law, with mean a little below $1/2$, so it stays mostly below the critical value, but often above it.

The geometric Brownian motion process is chosen with drift $\mu = 1$, diffusion $\sigma = 0.2$ and initial value $y_0 = 0.1$.

The point Poisson step process

As for the initial condition, we choose a Beta distribution with shape parameter

We don't have an explicit solution for the equation so we just use as target for the convergence an approximate solution via Euler method at a much higher resolution.


## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 04-popdyn_gBmPoisson
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables

````@example 04-popdyn_gBmPoisson
rng = Xoshiro(123)

function f(t, x, y)
    λ = 1.0
    ϵ = 0.3
    r = 1.0
    α = λ / 2
    dx = λ * (1 + ϵ * sin(y[1])) * x * (r - x) - α * y[2]
    return dx
end

t0 = 0.0
tf = 1.0

α = 7.0
β = 5.0
x0law = Beta(α, β)

μ = 1.0
σ = 0.2
y0 = 0.1
noise1 = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)

λₚ = 15.0
steplaw = Beta(β, α)
noise2 = PoissonStepProcess(t0, tf, λₚ, steplaw)

noise = ProductProcess(noise1, noise2)

ntgt = 2^18
ns = 2 .^ (4:9)
nsample = ns[[1, 2, 3, 4]]
m = 1_000
````

And add some information about the simulation:

````@example 04-popdyn_gBmPoisson
info = (
    equation = "population dynamics",
    noise = "gBm and step process noises",
    ic = "\$X_0 \\sim \\mathcal{B}($(round(α, sigdigits=1)), $(round(β, sigdigits=1)))\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 04-popdyn_gBmPoisson
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the [`ConvergenceSuite`](@ref):

````@example 04-popdyn_gBmPoisson
suite = ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors:

````@example 04-popdyn_gBmPoisson
@time result = solve(rng, suite)
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 04-popdyn_gBmPoisson
table = generate_error_table(result, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 04-popdyn_gBmPoisson
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")
````

### Plots

We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 04-popdyn_gBmPoisson
plot(result)
````

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide

For the sake of illustration, we plot a sample of an approximation of a target solution:

````@example 04-popdyn_gBmPoisson
plot(suite, ns=nsample)
````

We can also visualize the noise associated with this sample solution:

````@example 04-popdyn_gBmPoisson
plot(suite, shownoise=true, showapprox=false, showtarget=false)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

