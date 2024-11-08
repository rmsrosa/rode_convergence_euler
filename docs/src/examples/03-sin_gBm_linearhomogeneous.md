```@meta
EditURL = "../../literate/examples/03-sin_gBm_linearhomogeneous.jl"
```

# Homogenous linear RODE with the sine of a Geometric Brownian motion coefficient

This time we take, as the coefficient of a homogeneous linear equation, the sine of a Geometric Brownian motion process. This is a multiplicative noise.

## The equation

We consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \sin(Y_t) X_t, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{Y_t\}_{t\geq 0}$ is a geometric Brownian motion process.
The explicit solution is
```math
  X_t = e^{\int_0^t \sin(Y_s) \;\mathrm{d}s} X_0.
```

## Computing a higher order approximation of the solution

As in the previous examples, the integral $\int_0^{t_j} \sin(Y_s)\;\mathrm{d}s$ and, hence, the exact solution, is not uniquely defined from the values $W_{t_j}$ of the noise on the mesh points. This time, an exact distribution for the collection of exact solutions conditioned on the mesh points is not available in closed form. Hence, we consider an approximation of an exact solution by solving the equation numerically, with the Euler method itself, but with a much higher mesh resolution.

Indeed, the convergence will be estimated from a set of discretizations with mesh points with time step $\Delta t_N = (t\_f - t\_0) / 2^N$, for $N = N_1 < N_2 < \ldots N_n$, for some $n\in \mathbb{N}$, by comparing the error of such solutions to an approximated solutions computed in a finer mesh with $\Delta t_{\textrm{fine}} = \Delta t_{N_n}^2$, hence with $N_\textrm{fine} = N_n^2$.

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 03-sin_gBm_linearhomogeneous
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some parameters, with a [Distributions.Normal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Normal) random variable as the initial condition.

````@example 03-sin_gBm_linearhomogeneous
rng = Xoshiro(123)

f(t, x, y, p) = sin(y) * x

params = nothing

t0, tf = 0.0, 1.0
x0law = Normal()
````

The geometric Brownian motion noise is defined via [`GeometricBrownianMotionProcess`](@ref), with initial value $y_0$, drift $\mu$, and dissipation $\sigma$ as given by

````@example 03-sin_gBm_linearhomogeneous
μ = 1.0
σ = 0.2
y0 = 1.0
noise = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)
````

The mesh parameters are

````@example 03-sin_gBm_linearhomogeneous
ntgt = 2^18
ns = 2 .^ (4:9)
````

and

````@example 03-sin_gBm_linearhomogeneous
nsample = ns[[1, 2, 3, 4]]
````

The number of simulations for the Monte Carlo estimate is set to

````@example 03-sin_gBm_linearhomogeneous
m = 200
nothing # hide
````

We add some information about the simulation, for the caption of the convergence figure.

````@example 03-sin_gBm_linearhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(Y_t) X_t\$",
    noise = "a geometric Brownian motion process noise \$\\{Y_t\\}_t\$ (ic=$y0, drift=$μ; diffusion=$σ)",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)
nothing # hide
````

We define the *target* solution as the approximation obtained by the Euler method in the much higher resolution `ntgt` of mesh points. The approximations are also obtained via the Euler method, in the coarser meshes defined by `ns`.

````@example 03-sin_gBm_linearhomogeneous
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the [`ConvergenceSuite`](@ref):

````@example 03-sin_gBm_linearhomogeneous
suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors via [`solve`](@ref):

````@example 03-sin_gBm_linearhomogeneous
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 03-sin_gBm_linearhomogeneous
table = generate_error_table(result, suite, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 03-sin_gBm_linearhomogeneous
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide
````

### Plots

We illustrate the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 03-sin_gBm_linearhomogeneous
plt = plot(result)
````

````@example 03-sin_gBm_linearhomogeneous
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/order_sin_gBm_linearhomogenous.pdf")) # hide
nothing # hide
````

For the sake of illustration, we plot some approximations of a sample target solution:

````@example 03-sin_gBm_linearhomogeneous
plot(suite, ns=nsample)
````

We can also visualize the noise associated with this sample solution,

````@example 03-sin_gBm_linearhomogeneous
plot(suite, xshow=false, yshow=true, label="gBm noise")
````

and the sine of the noise, which is the coefficient of the equation

````@example 03-sin_gBm_linearhomogeneous
plot(suite, xshow=false, yshow=sin, label="sin of gBm noise")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

