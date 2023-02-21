```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/03-sin_gBm_linearhomogeneous.jl"
```

# Homogenous linear RODE with the sine of a Geometric Brownian motion coefficient

Now we consider a homogeneous linear equation in which the coefficient depends on the sine of a Geometric Brownian motion process.

## The equation

More precisely, we consider the RODE
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

The integral $\int_0^{t_j} \sin(Y_s)\;\mathrm{d}s$ and, hence, the exact solution, is not uniquely defined from the values $W_{t_j}$ of the noise on the mesh points, no matter how fine it is, and an exact distribution for the collection of exact solutions conditioned on the mesh points is not available in closed form. Hence, we consider an approximation of an exact solution by solving the equation numerically, with the Euler method itself, but in a much higher resolution.

Indeed, the convergence will be estimated from a set of discretizations with mesh points with time step $\Delta t_N = N$, for $N = N_1 < N_2 < \ldots N_n$, for some $n\in \mathbb{N}$, by comparing the error of such solutions to an approximated solutions computed in a finer mesh with $\Delta t_{\textrm{fine}} = \Delta t_{N_n}^2$, hence with $N_\textrm{fine} = N_n^2$.

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 03-sin_gBm_linearhomogeneous
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables

````@example 03-sin_gBm_linearhomogeneous
rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
μ = 1.0
σ = 0.2
y0 = 1.0
noise! = GBM_noise(t0, tf, y0, μ, σ)
f(t, x, y) = sin(y) * x

Ntgt = 2^18
Ns = 2 .^ (4:9)
M = 1_000
````

And add some information about the simulation:

````@example 03-sin_gBm_linearhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(Y_t) X_t\$",
    noise = "a geometric Brownian motion process noise \$\\{Y_t\\}_t\$ (drift=$μ; diffusion=$σ)",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$",
    M = M,
    Ntgt = Ntgt,
    Ns = Ns,
    filename = "order_sin_gBm_linearhomogenous.png"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `Ntgt` of mesh points:

````@example 03-sin_gBm_linearhomogeneous
target! = solve_euler!
````

### An illustrative sample path

````@example 03-sin_gBm_linearhomogeneous
plt, plt_noise, = plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info)
nothing # hide
````

````@example 03-sin_gBm_linearhomogeneous
plt_noise
````

````@example 03-sin_gBm_linearhomogeneous
plt
````

### An illustrative ensemble of solutions

### Order of convergence

With everything set up, we compute the errors:

````@example 03-sin_gBm_linearhomogeneous
@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
nothing # hide
````

The computed strong errors are stored in `errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 03-sin_gBm_linearhomogeneous
table = generate_error_table(Ns, deltas, errors, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `p`:

````@example 03-sin_gBm_linearhomogeneous
println("Order of convergence `C Δtᵖ` with p = $(round(p, sigdigits=2))")
````

### Plots

We create a plot with the rate of convergence with the help of `plot_dt_vs_error`. This returns a handle for the plot and a title.

````@example 03-sin_gBm_linearhomogeneous
plt, title = plot_dt_vs_error(deltas, errors, lc, p, info)
nothing # hide
````

One can use that to plot the figure here:

````@example 03-sin_gBm_linearhomogeneous
plot(plt; title)
````

While for the article, you plot a figure without the title and use `title` to create the caption for the latex source:

````@example 03-sin_gBm_linearhomogeneous
plot(plt)

println(title)
````

````@example 03-sin_gBm_linearhomogeneous
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide
````

We can also plot the time-evolution of the strong errors along the time mesh, just for the sake of illustration:

````@example 03-sin_gBm_linearhomogeneous
plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
