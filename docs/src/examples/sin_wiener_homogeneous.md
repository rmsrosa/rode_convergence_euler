# Homogenous linear RODE with a Geometric Brownian motion coefficient

Now we consider a homogeneous linear equation in which the coefficient depends on the sine of a Geometric Brownian motion process.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \sin (W_t) X_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a Wiener process.

The explicit solution is
```math
X_t = e^{\int_0^t \sin(W_s) \;\mathrm{d}s} X_0.
```

## Numerical approximation

### Setting up the problem

First we load the necessary packages

```@example sinwienerhomogeneous
using Plots
using Random
using Distributions
using RODEConvergence
```

Then we set up some variables

```@example sinwienerhomogeneous
rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
μ = 1.0
σ = 0.2
y0 = 1.0
noise! = GBM_noise(t0, tf, y0, μ, σ)
f(t, x, y) = sin(y) * x

Ntgt = 2^20
Ns = 2 .^ (4:10)
M = 1_000
```

And add some information about the simulation:

```@example sinwienerhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(W_t) X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$",
    M = M,
    Ntgt = Ntgt,
    Ns = Ns
)
```

We define the *target* solution as the numerical solution via the Euler method, but that is computed in the finest mesh, to avoid spurious errors.

```@example sinwienerhomogeneous
target! = solve_euler!
```

### An illustrative sample path

```@example sinwienerhomogeneous
plt, plt_noise, = plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info)
nothing # hide
```

```@example sinwienerhomogeneous
plt_noise
```

```@example sinwienerhomogeneous
plt
```

### An illustrative ensemble of solutions

### Order of convergence

With everything set up, we compute the errors:

```@example sinwienerhomogeneous
@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
nothing # hide
```

The computed strong errors are stored in `errors`, and a raw LaTeX table can be displayed for inclusion in the article:

```@example sinwienerhomogeneous
table = generate_error_table(Ns, deltas, errors, info)

println(table) # hide
nothing # hide
```

The calculated order of convergence is given by `p`:

```@example sinwienerhomogeneous
println("Order of convergence `C Δtᵖ` with p = $(round(p, sigdigits=2))")
nothing # hide
```

### Plots

We create a plot with the rate of convergence with the help of `plot_dt_vs_error`. This returns a handle for the plot and a title.

```@example sinwienerhomogeneous
plt, title = plot_dt_vs_error(deltas, errors, lc, p, info)
nothing # hide
```

One can use that to plot the figure here:

```@example sinwienerhomogeneous
plot(plt; title)
```

While for the article, you plot a figure without the title and use `title` to create the caption for the latex source:

```@example sinwienerhomogeneous
plot(plt)
```

```@example sinwienerhomogeneous
println(title)
```

```@example sinwienerhomogeneous
filename = "../../../latex/img/order_sinwienerhomogenous.png" # hide
savefig(plt, filename) # hide
nothing # hide
```

We can also plot the time-evolution of the strong errors along the time mesh, just for the sake of illustration:

```@example sinwienerhomogeneous
plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
```
