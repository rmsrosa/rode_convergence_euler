# Homogenous linear RODE with the coefficient as the sine of a Wiener process

Now we consider a homogeneous linear equation in which the coefficient depends on the sine of a Wiener process.

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

## Computing the exact solution

As before, we cannot compute the integral $\int_0^{t_j} \sin(W_s)\;\mathrm{d}s$ exactly, just from the values $W_{t_j}$ of the noise, on the mesh points, but we can compute its expectation. We break down the sum into parts:
```math
\int_0^{t_j} \sin(W_s)\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s.
```

On each mesh interval, we use that
```math
B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
is a Brownian bridge on the interval $[t_i, t_{i+1}]$, independent of $\{W_t\}_{t\geq 0}$.

According to Itô's formula, for a smooth function $f=f(w)$, we have
```math
\mathrm{d}f(W_t) = \frac{1}{2}f''(W_t)\;\mathrm{d}t + f'(W_t)\;\mathrm{d}W_t.
```
Considering $f(w) = \sin(w)$, we find
```math
\mathrm{d}\sin(W_t) = -\frac{1}{2}\sin(W_t)\;\mathrm{d}t + \cos(W_t)\;\mathrm{d}W_t.
```

Since
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t,
```
we obtain
```math
\mathrm{d}\sin(W_t) = -\frac{1}{2}\sin(W_t)\;\mathrm{d}t + \cos(W_t)\left(\mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right),
```
which can be written as
```math
\sin(W_t)\;\mathrm{d}t = -2\mathrm{d}\sin(W_t) + 2\cos(W_t)\left(\mathrm{d}B_t + 2\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right).
```

Thus,
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s \\
& \qquad - 2\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}s.
\end{align*}
```

Analogously,
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}s & = -2\cos(W_{t_{i+1}}) + 2\cos(W_{t_i}) + 2\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s \\
& \qquad + 2\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s.
\end{align*}
```

Substituting,
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right) \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s\\
& \qquad - 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s.
\end{align*}
```

Therefore,
```math
\begin{align*}
\left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right) \\
& \qquad - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s.
\end{align*}
```

Thus, we can write that
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s & = \left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)^{-1} \\
& \qquad \left( -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right)\right) \\
& \qquad + Z_i,
\end{align*}
```
where
```math
Z_i = \left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)^{-1}\left( - 2\int_{t_i}^{t_{i+1}} \cos(W_s)\;\mathrm{d}B_s - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i} \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}B_s\right).
```

Hmm, $Z_i$ averages to zero, but without knowing more about it, we cannot get a more explicit form for the expected exact solution, which envolves the exponential of the above integral. This example will have to be on hold, for now.

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
y0 = 0.0
f(t, x, y) = sin(y) * x
noise! = Wiener_noise(t0, tf, y0)

Ntgt = 2^20
Ns = 2 .^ (4:10)
M = 1_000
```

We define the *target* solution.

```@example sinwienerhomogeneous
target! = function (rng, Xt, t0, tf, x0, f, Yt)
    Ntgt = length(Yt)
    dt = (tf - t0) / (Ntgt - 1)
    Xt[1] = x0
    It = 0.0
    for n in 2:Ntgt
        It += (Yt[n] + Yt[n-1]) * dt / 2 + randn(rng) * sqrt(dt^3) / 12
        Xt[n] = x0 * exp(It)
    end
end

target! = solve_euler!
```

And add some information about the simulation:

```@example sinwienerhomogeneous
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(W_t) X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$",
    M = "$M sample paths"
)
```

### An illustrative sample path

```@example sinwienerhomogeneous
filename = nothing # hide
plt, plt_noise, = plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info, filename)
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
table = generate_error_table(Ns, deltas, errors)

println(table) # hide
nothing # hide
```

The calculated order of convergence is given in `p`:

```@example sinwienerhomogeneous
p
```

### Plots

```@example sinwienerhomogeneous
filename = nothing
plot_dt_vs_error(deltas, errors, lc, p; info, filename)
```

```@example sinwienerhomogeneous
plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
```

