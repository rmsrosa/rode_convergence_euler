```@meta
EditURL = "../../literate/examples/02-wiener_linearnonhomogeneous.jl"
```

# Non-homogenous linear RODE with a Wiener process noise coefficient

In our second linear example, a Wiener process noise enters as the non-homogeneous term.

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

As in the first example, the integral $\int_0^{t_j} e^s W_s\;\mathrm{d}s$ and, hence, the exact solution, is not uniquely defined from the values $W_{t_j}$ of the noise on the mesh points, no matter how fine it is. Thus we estimate the strong error by drawing sample solutions with the exact distribution conditioned on the mesh values.

We do that by first breaking down the sum into parts:
```math
\int_0^{t_j} e^s W_s\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} e^s W_s\;\mathrm{d}s.
```

On each mesh interval, we consider again the Brownian bridge
```math
  B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
on $[t_i, t_{i+1}]$, which is independent of $W_{t_i}$ and $W_{t_{i+1}}$.

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
      & = W_{t_j}e^{t_j} - \sum_{i = 0}^{j-1} \left( \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(e^{t_{i+1}}-e^{t_i}\right) + Z_i\right).
  \end{align*}
```

Thus, once an Euler approximation is computed, along with realizations $\{W_{t_i}\}_{i=0}^n$ of a sample path of the noise, we consider an exact sample solution given by
```math
  X_{t_j} = e^{-{t_j}}\left(X_0 - \sum_{i=0}^{j-1} \left(\frac{W_{t_{i+1}} - W_{t_i}}{t_{i+1}-t_i}\left(e^{t_{i+1}} - e^{t_i}\right) + Z_i\right)\right) + W_{t_j},
```
for realizations $Z_i$ drawn from a normal distribution and scaled by the standard deviation $\sqrt{(t_{i+1} - t_i)^3/12}$. This is implemented by computing the integral recursively, via
```math
    \begin{cases}
        I_j = I_{j-1} + \frac{W_{t_{j-1}} + W_{t_j}}{t_{j} - t_{j-1}}\left(e^{t_{j} - e^{t_{j-1}}}\right) - Z_j, \\
        Z_j = \sqrt{\frac{(t_{j} - t_{j-1})^3}{12}} R_j, \\
        R_j \sim \mathcal{N}(0, 1), \\
    \end{cases}
```
with $I_0 = 0$, and setting
```math
  X_{t_j} = e^{t_j}\left(X_0 - I_j\right) + W_{t_j}.
```

## Numerical approximation

### Setting up the problem

We load the necessary packages

````@example 02-wiener_linearnonhomogeneous
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up the relevant variables, as in the first example:

````@example 02-wiener_linearnonhomogeneous
rng = Xoshiro(123)

f(t, x, y, p) = - x + y

t0, tf = 0.0, 1.0
x0law = Normal()

y0 = 0.0
noise = WienerProcess(t0, tf, y0)

params = nothing

ntgt = 2^16
ns = 2 .^ (4:14)
````

The numbers of mesh points for a visualization of one of the sample approximations

````@example 02-wiener_linearnonhomogeneous
nsample = ns[[1, 2, 3, 4]]
````

The number of simulations for the Monte Carlo estimate is set to

````@example 02-wiener_linearnonhomogeneous
m = 80
nothing # hide
````

and the info about the simulation, for the caption of the convergence figure.

````@example 02-wiener_linearnonhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)
nothing # hide
````

We define the *target* solution as described above.

````@example 02-wiener_linearnonhomogeneous
target_solver! = function (xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, params::Q, rng::AbstractRNG) where {T, F, Q}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )

    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    integral = zero(T)
    ti1 = zero(T)
    zscale = sqrt(dt^3 / 12)
    for i in Iterators.drop(eachindex(xt, yt), 1)
        ti = ti1 + dt
        integral += (yt[i] - yt[i1]) * (exp(ti) - exp(ti1)) / dt +  zscale * randn(rng)
        xt[i] = exp(-ti) * (x0 - integral) + yt[i]
        ti1 = ti
        i1 = i
    end
end
nothing # hide
````

and with that we construct the [`CustomMethod`](@ref) that solves the problem with this `target_solver!`:

````@example 02-wiener_linearnonhomogeneous
target = CustomUnivariateMethod(target_solver!, rng)
nothing # hide
````

The method for which want to estimate the rate of convergence is, naturally, the Euler method, implemented via [`RandomEuler`](@ref):

````@example 02-wiener_linearnonhomogeneous
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the [`ConvergenceSuite`](@ref):

````@example 02-wiener_linearnonhomogeneous
suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors via [`solve`](@ref):

````@example 02-wiener_linearnonhomogeneous
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 02-wiener_linearnonhomogeneous
table = generate_error_table(result, suite, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 02-wiener_linearnonhomogeneous
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide
````

### Plots

We draw a plot of the rate of convergence with the help of a plot recipe for [`ConvergenceResult`](@ref):

````@example 02-wiener_linearnonhomogeneous
plt = plot(result)
````

````@example 02-wiener_linearnonhomogeneous
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/order_wiener_linearnonhomogenous.pdf")) # hide
nothing # hide
````

For the sake of illustration, we plot an approximation of a sample target solution:

````@example 02-wiener_linearnonhomogeneous
plot(suite, ns=nsample)
````

We can also visualize the noise associated with this sample solution:

````@example 02-wiener_linearnonhomogeneous
plot(suite, xshow=false, yshow=true, label="Wiener noise")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

