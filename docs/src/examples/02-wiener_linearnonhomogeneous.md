```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/02-wiener_linearnonhomogeneous.jl"
```

# Non-homogenous linear RODE with a Wiener process noise coefficient

Next we consider another linear equation, but in which a Wiener process noise enters as the non-homogeneous term.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = - X_t + W_t, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a Wiener process.

The explicit solution is
```math
  X_t = e^{-t}X_0 + \int_0^t e^{-(t-s)}W_s\;\mathrm{d}s.
```

## Computing a solution with the exact distribution

The integral $\int_0^{t_j} e^s W_s\;\mathrm{d}s$ and, hence, the exact solution, is not uniquely defined from the values $W_{t_j}$ of the noise on the mesh points, no matter how fine it is. Hence, it makes no sense to compute the strong distance to "the exact solution". But we can estimate that by drawing sample solutions with the exact distribution conditioned on the mesh values.

We do that by first breaking down the sum into parts:
```math
\int_0^{t_j} e^s W_s\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} e^s W_s\;\mathrm{d}s.
```

On each mesh interval, we consider again the Brownian bridge
```math
  B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
on $[t_i, t_{i+1}]$, which is independent of $\{W_t\}_{t\geq 0}$.

Then,
```math
  \begin{align*}
      \int_{t_i}^{t_{i+1}} e^s W_s\;\mathrm{d}s & = \int_{t_i}^{t_{i+1}} e^s B_s^i\;\mathrm{d}s + \int_{t_i}^{t_{i+1}} e^s\left( W_{t_i} + \frac{s - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})\right)\;\mathrm{d}s \\
      & = W_{t_{i+1}}e^{t_{i+1}} - W_{t_i}e^{t_i} - \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(e^{t_{i+1}}-e^{t_i}\right) + Z_i,
  \end{align*}
```
where
```math
    Z_i = \int_{t_i}^{t_{i+1}} e^s B_s^i\;\mathrm{d}s.
```

As before, the term $Z_i$ is a Gaussian with zero mean, and we need to compute its variance to completely characterize it. By translation, it suffices to consider a Brownian bridge $\{B_t\}_{t\in [0, \tau]}$ on an interval $[0, \tau]$, with $\tau = \Delta t_N$. This is obtained from $B_t = W_t - (t/\tau)W_\tau$. We have, since $\mathbb{E}[W_tW_s] = \min\{t, s\}$, that
```math
   \mathbb{E}[B_tB_s] = \min\{t, s\} - \frac{ts}{\tau}.
```
Hence,
```math
      \begin{align*}
      \mathbb{E}\left[\left(\int_0^{\tau} e^s B_s\;\mathrm{d}s\right)^2\right] & = \mathbb{E}\left[\int_0^{\tau} \int_0^\tau e^s e^t B_sB_t\;\mathrm{d}s\;\mathrm{d}\right] \\
      & = \int_0^\tau \int_0^\tau e^s e^t \mathbb{E}[B_sB_t] \;\mathrm{d}s\;\mathrm{d}t \\
      & = \int_0^\tau \int_0^\tau e^s e^t\left(\min\{t, s\} - \frac{ts}{\tau}\right) \;\mathrm{d}s\;\mathrm{d}t \\
      & = \int_0^\tau \int_0^t e^s e^t s\;\mathrm{d}s\;\mathrm{d}t + \int_0^\tau \int_t^\tau e^s e^t t\;\mathrm{d}s\;\mathrm{d}t - \int_0^\tau \int_0^\tau e^s e^t \frac{ts}{\tau} \;\mathrm{d}s\;\mathrm{d}t \\
      & = \int_0^\tau e^t(te^t-e^t+1)\;\mathrm{d}t + \int_0^\tau te^t(e^\tau - e^t)\;\mathrm{d}t \\
      & \qquad - \int_0^\tau \frac{te^t}{\tau}\left(\tau e^\tau - e^\tau + 1\right)\;\mathrm{d}t \\
      & = \frac{\tau^3}{12}.
  \end{align*}
```

Back to $Z_i$, this means that
```math
     Z_i \sim \mathcal{N}\left(0, \frac{(t_{i+1}- t_i)^3}{12}\right) = \frac{\sqrt{(t_{i+1} - t_i)^3}}{\sqrt{12}}\mathcal{N}(0, 1).
```

Summing up the terms, we find that
```math
  \begin{align*}
      \int_0^{t_j} e^s W_s\;\mathrm{d}s & = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} e^s W_s\;\mathrm{d}s \\
      & = \sum_{i = 0}^{j-1} \left( W_{t_{i+1}}e^{t_{i+1}} - W_{t_i}e^{t_i} - \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(e^{t_{i+1}}-e^{t_i}\right) + Z_i\right) \\
      & = W_{t_j}e^{t_j} - \sum_{i = 0}^{j-1} \left( \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(e^{t_{i+1}}-e^{t_i}\right) + Z_i\right)
  \end{align*}
```

Thus, once an Euler approximation is computed, along with realizations $\{W_{t_i}\}_{i=0}^n$ of a sample path of the noise, we consider an exact sample solution given by
```math
  X_{t_j} = e^{-{t_j}}\left(X_0 - \sum_{i=0}^{j-1} \left(\frac{W_{t_{i+1}} - W_{t_i}}{t_{i+1}-t_i}\left(e^{t_{i+1}} - e^{t_i}\right) + Z_i\right)\right) + W_{t_j},
```
for realizations $Z_i$ drawn from a normal distribution and scaled by the standard deviation $\sqrt{(t_{i+1} - t_i)^3/12}$. This is implemented by computing the integral recursively, via
```math
    \begin{cases} \\
        I_j = I_{j-1} - \left(W_{t_{j-1}} + W_{t_j}\right)(t_{j} - t_{j-1}) + Z_j, \\
        Z_j = \sqrt{\frac{(t_{j} - t_{j-1})^3}{12}} R_j, \\
        R_j \sim \mathcal{N}(0, 1), \\
    \end{cases}
```
with $I_0 = 0$, and setting
```math
  X_{t_j} = e^{t_j}\left(X_0 + I_j\right) + W_{t_j}.
```

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 02-wiener_linearnonhomogeneous
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables

````@example 02-wiener_linearnonhomogeneous
rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
y0 = 0.0
noise! = Wiener_noise(t0, tf, y0)
f(t, x, y) = - x + y

Ntgt = 2^20
Ns = 2 .^ (4:10)
M = 1_000
````

And add some information about the simulation:

````@example 02-wiener_linearnonhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$",
    M = M,
    Ntgt = Ntgt,
    Ns = Ns,
    filename = "order_wiener_linearnonhomogenous.png"
)
````

We define the *target* solution as described above.

````@example 02-wiener_linearnonhomogeneous
target! = function (rng, Xt, t0, tf, x0, f, Yt)
    Ntgt = length(Yt)
    dt = (tf - t0) / (Ntgt - 1)
    Xt[1] = x0
    It = 0.0
    for n in 2:Ntgt
        It -= (Yt[n] + Yt[n-1]) * dt + randn(rng) * sqrt(dt^3 / 12)
        tn = (n-1) * dt
        Xt[n] = exp(-tn) * (x0 + It) + Yt[n]
    end
end
````

There is something wrong with the formula, just use the euler approximation for now:

````@example 02-wiener_linearnonhomogeneous
target! = solve_euler!
````

### An illustrative sample path

````@example 02-wiener_linearnonhomogeneous
plt, plt_noise, = plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info)
nothing # hide
````

````@example 02-wiener_linearnonhomogeneous
plt_noise
````

````@example 02-wiener_linearnonhomogeneous
plt
````

### An illustrative ensemble of solutions

### Order of convergence

With everything set up, we compute the errors:

````@example 02-wiener_linearnonhomogeneous
@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
nothing # hide
````

The computed strong errors are stored in `errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 02-wiener_linearnonhomogeneous
table = generate_error_table(Ns, deltas, errors, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `p`:

````@example 02-wiener_linearnonhomogeneous
println("Order of convergence `C Δtᵖ` with p = $(round(p, sigdigits=2))")
````

### Plots

We create a plot with the rate of convergence with the help of `plot_dt_vs_error`. This returns a handle for the plot and a title.

````@example 02-wiener_linearnonhomogeneous
plt, title = plot_dt_vs_error(deltas, errors, lc, p, info)
nothing # hide
````

One can use that to plot the figure here:

````@example 02-wiener_linearnonhomogeneous
plot(plt; title)
````

While for the article, you plot a figure without the title and use `title` to create the caption for the latex source:

````@example 02-wiener_linearnonhomogeneous
plot(plt)

println(title)
````

````@example 02-wiener_linearnonhomogeneous
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide
````

We can also plot the time-evolution of the strong errors along the time mesh, just for the sake of illustration:

````@example 02-wiener_linearnonhomogeneous
plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

