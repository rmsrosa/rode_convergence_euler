# Homogenous linear RODE with a Wiener process noise coefficient

Now we consider a homogeneous linear equation in which the coefficient is a Wiener process.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = W_t X_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a Wiener process.

The explicit solution is
```math
X_t = e^{\int_0^t W_s \;\mathrm{d}s} X_0.
```

## Computing a solution with the exact distribution

The integral $\int_0^{t_j} W_s\;\mathrm{d}s$ and, hence, the exact solution, is not uniquely defined from the values $W_{t_j}$ of the noise on the mesh points, no matter how fine it is. Hence, it makes no sense to compute the strong distance to "the exact solution". But we can estimate that by drawing sample solutions with the exact distribution conditioned on the mesh values.

We do that by first breaking down the sum into parts:
```math
\int_0^{t_j} W_s\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} W_s\;\mathrm{d}s.
```

On each mesh interval, we use that
```math
B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
is a Brownian bridge on $[t_i, t_{i+1}]$, independent of $\{W_t\}_{t\geq 0}$.

Since
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t,
```
we obtain
```math
\begin{align*}
    \int_{t_i}^{t_{i+1}} W_s\;\mathrm{d}s & = \int_{t_i}^{t_{i+1}} B_s^i\;\mathrm{d}s + \int_{t_i}^{t_{i+1}} \left( W_{t_i} + \frac{s - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})\right)\;\mathrm{d}s \\
    & = \frac{1}{2}\left(W_{t_i} + W_{t_{i+1}}\right)(t_{i+1} - t_i) + Z_i,
\end{align*}
```
where
```math
    Z_i = \int_{t_i}^{t_{i+1}} B_s^i\;\mathrm{d}s.
```

Notice the first term is the trapezoidal rule while the second term is a Gaussian with zero mean. We need to compute the variance of $Z_i$ to completely characterize it. By translation, it suffices to consider a Brownian bridge $\{B_t\}_{t\in [0, \tau]}$ on an interval $[0, \tau]$, with $\tau = \Delta t_N$. This is obtained from $B_t = W_t - (t/\tau)W_\tau$. We have, since $\mathbb{E}[W_tW_s] = \min\{t, s\}$, that
```math
    \mathbb{E}[B_tB_s] = \min\{t, s\} - \frac{ts}{\tau}.
```
Hence,
```math
\begin{align*}
    \mathbb{E}\left[\left(\int_0^{\tau} B_s\;\mathrm{d}s\right)^2\right] & = \mathbb{E}\left[\int_0^{\tau} \int_0^\tau B_sB_t\;\mathrm{d}s\;\mathrm{d}\right] \\
    & = \int_0^\tau \int_0^\tau \mathbb{E}[B_sB_t] \;\mathrm{d}s\;\mathrm{d}t \\
    & = \int_0^\tau \int_0^\tau \left(\min\{t, s\} - \frac{ts}{\tau}\right) \;\mathrm{d}s\;\mathrm{d}t \\
    & = \int_0^\tau \int_0^t s\;\mathrm{d}s\;\mathrm{d}t + \int_0^\tau \int_t^\tau t\;\mathrm{d}s\;\mathrm{d}t - \int_0^\tau \int_0^\tau \frac{ts}{\tau} \;\mathrm{d}s\;\mathrm{d}t \\
    & = \int_0^\tau \frac{t^2}{2}\;\mathrm{d}t + \int_0^\tau t(\tau - t)\;\mathrm{d}t - \int_0^\tau \frac{t\tau^2}{2\tau}\;\mathrm{d}t \\
    & = \frac{\tau^3}{12}.
\end{align*}
```

Back to $Z_i$, this means that
```math
    Z_i \sim \mathcal{N}\left(0, \frac{(t_{i+1}- t_i)^3}{12}\right) = \sqrt{\frac{(t_{i+1} - t_i)^3}{12}}\mathcal{N}(0, 1).
```

For a normal variable $N \sim \mathcal{N}(\mu, \sigma)$, the expectation of the random variable $e^N$ is $\mathbb{E}[e^N] = e^{\mu + \sigma^2/2}$. Hence,
```math
    \mathbb{E}[e^{Z_i}] = e^{((t_{i+1}- t_i)^3)/24}.
```
This is the contribution of this random variable to the mean of the exact solution. But in the implementation we actually draw from $Z_i$, not from $e^{Z_i}$.

Thus, once an Euler approximation is computed, along with realizations $\{W_{t_i}\}_{i=0}^n$ of a sample path of the noise, we consider an exact sample solution given by
```math
    X_{t_j} = X_0 e^{\sum_{i = 0}^{j-1}\left(\frac{1}{2}\left(W_{t_i} + W_{t_{i+1}}\right)(t_{i+1} - t_i) + Z_i\right)},
```
for realizations $Z_i$ drawn from a normal distribution and scaled by the standard deviation $\sqrt{(t_{i+1} - t_i)^3/12}$. This is implemented by computing the integral recursively, via
```math
    \begin{cases} \\
        I_j = I_{j-1} + \frac{1}{2}\left(W_{t_{j-1}} + W_{t_j}\right)(t_{j} - t_{j-1}) + Z_j, \\
        Z_j = \sqrt{\frac{(t_{j} - t_{j-1})^3}{12}} R_j, \\
        R_j \sim \mathcal{N}(0, 1), \\
    \end{cases}
```
with $I_0 = 0$, and setting
```math
X_{t_j} = X_0 e^{I_j}.
```

## Numerical approximation

### Setting up the problem

First we load the necessary packages

```@example wienerhomogeneous
using Plots
using Random
using Distributions
using RODEConvergence
```

Then we set up some variables

```@example wienerhomogeneous
rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
y0 = 0.0
f(t, x, y) = y * x
noise! = Wiener_noise(t0, tf, y0)

Ntgt = 2^20
Ns = 2 .^ (4:10)
M = 1_000
```

We define the *target* solution as described above.

```@example wienerhomogeneous
target! = function (rng, Xt, t0, tf, x0, f, Yt)
    Ntgt = length(Yt)
    dt = (tf - t0) / (Ntgt - 1)
    Xt[1] = x0
    It = 0.0
    for n in 2:Ntgt
        It += (Yt[n] + Yt[n-1]) * dt / 2 + randn(rng) * sqrt(dt^3 / 12)
        Xt[n] = x0 * exp(It)
    end
end
```

And add some information about the simulation:

```@example wienerhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$",
    M = "$M sample paths"
)
```

### An illustrative sample path

```@example wienerhomogeneous
filename = nothing # hide
plt, plt_noise, = plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info, filename)
nothing # hide
```

```@example wienerhomogeneous
plt_noise
```

```@example wienerhomogeneous
plt
```

### An illustrative ensemble of solutions

### Order of convergence

With everything set up, we compute the errors:

```@example wienerhomogeneous
@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
nothing # hide
```

The computed strong errors are stored in `errors`, and a raw LaTeX table can be displayed for inclusion in the article:

```@example wienerhomogeneous
table = generate_error_table(Ns, deltas, errors)

println(table) # hide
nothing # hide
```

The calculated order of convergence is given by `p`:

```@example wienerhomogeneous
println("Order of convergence `C Δtᵖ` with p = $(round(p, sigdigits=2))")
nothing # hide
```

### Plots

```@example wienerhomogeneous
filename = nothing
plot_dt_vs_error(deltas, errors, lc, p; info, filename)
```

```@example wienerhomogeneous
plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
```

