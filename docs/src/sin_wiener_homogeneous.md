# Homogenous linear RODE with sine of Wiener noise coefficient

Now we consider a homogeneous linear equation in which the coefficient depends on the sine of a Wiener process.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = (\mu + \sigma \sin W_t) X_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a Wiener process.

The explicit solution
```math
X_t = e^{\int_0^t (\mu + \sigma \sin(W_s)) \;\mathrm{d}s} X_0.
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

Taking the expectation, using that the expectation of an Itô integral with respect to a Brownian bridge with zero endpoints is zero, and using that the values at the mesh points are given, we find that
```math
\begin{align*}
\mathbb{E}\left[ \int_{t_i}^{t_{i+1}} \sin(W_s)\;\mathrm{d}s\right] & = \left(1 + 4\left(\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\right)^2 \right)^{-1} \\
& \qquad \left( -2\sin(W_{t_{i+1}}) + 2\sin(W_{t_i}) - 4\frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left(-\cos(W_{t_{i+1}}) + \cos(W_{t_i})\right)\right).
\end{align*}
```

Hence, given the realizations of a Wiener noise on the mesh points,
```math
\begin{align*}
\mathbb{E}[X_{t_j}] & = e^{-t_j}X_0 + \sum_{i=0}^{j-1} \left(e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i}\right) \\
& \qquad - \sum_{i=0}^{j-1} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
\end{align*}
```

The first summation telescopes out and, since $W_0 = 0$, we are left with
```math
\mathbb{E}[X_{t_j}] = e^{-t_j}X_0 + W_{t_j} - e^{-t_j}\sum_{i=0}^{j-1} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{t_{i+1}} - e^{-t_i}\right).
```

Thus, we estimate the error by calculating the difference from the numerical approximation to the above expectation.

The summation above can be computed recursively. Indeed, if
```math
I_j = \sum_{i=0}^{j-1} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{t_{i+1}} - e^{t_i}\right),
```
then
```math
I_0 = 0
```
and, for $j = 1, \ldots, n$,
```math
\begin{aligned}
I_j & = \sum_{i=0}^{j-1} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{t_{i+1}} - e^{t_i}\right) \\
& = \frac{W_{t_{j}}-W_{t_{j-1}}}{t_{j}-t_{j-1}}\left( e^{t_{j}} - e^{t_{j-1}}\right) + \sum_{i=0}^{j-2} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{t_{i+1}} - e^{t_i}\right),
\end{aligned}
```
so that
```math
I_j = \frac{W_{t_{j}}-W_{t_{j-1}}}{t_{j}-t_{j-1}}\left( e^{t_{j}} - e^{t_{j-1}}\right) + I_{j-1}.
```

## Numerical approximation

First we load the necessary packages

```julia linrode
using DiffEqNoiseProcess, StochasticDiffEq, Plots, DiffEqDevTools, Random
```

### Setting up the problem

Now we set up the RODE problem.

We define the equation in the form
```math
\frac{\mathrm{d}u}{\mathrm{d}t} = f(u, t, p, W)
```

so that the right hand side becomes
```math
f(u, t, p, W) = pu + W,
```
with $p = -1.0$.

```julia linrode
f(u, p, t, W) = p * u + W
p = -1.0
```

Next we define the function that yields the analytic solution for a given computed solution `sol`, that contains the noise `sol.W` and the info about the (still to be defined) RODE problem `prob`.

```julia linrode
function f_analytic!(sol)
    empty!(sol.u_analytic)

    u0 = sol.prob.u0
    p = sol.prob.p
    push!(sol.u_analytic, u0)

    ti1, Wi1 = sol.W.t[1], sol.W.W[1]
    integral = 0.0
    for i in 2:length(sol)
        ti, Wi = sol.W.t[i], sol.W.W[i]
        integral += - (Wi - Wi1) / (ti - ti1) * (exp(ti) - exp(ti1))
        push!(sol.u_analytic, Wi + exp(-ti) * (u0 + integral))
        ti1, Wi1 = ti, Wi
    end
end
```

With the right-hand-side and the analytic solutions defined, we construct the `RODEFunction` to be passed on to the RODE problem builder.

```julia linrode
ff = RODEFunction(
    f,
    analytic = f_analytic!,
    analytic_full=true
)
```

Now we set up the RODE problem, with initial condition `X0 = 1.0`, and time span `tspan = (0.0, 1.0)`:

```julia linrode
X0 = 1.0
tspan = (0.0, 1.0)

prob = RODEProblem(ff, X0, tspan, p)
```

### An illustrative sample path

Just for the sake of illustration, we solve for a solution path, using the Euler method, which in SciML is provided as `RandomEM()`, and with a fixed time step. We fix the `seed` just for the sake of reproducibility:
```julia linrode
sol = solve(prob, RandomEM(), dt = 1/100, seed = 123)
```

and display the result
```julia linrode
plot(
    sol,
    title = "Sample path of \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$",
    titlefont = 12,
    xaxis = "\$t\$",
    yaxis = "\$x\$",
    label="\$X_t\$"
)
```

### An illustrative ensemble of solutions

```julia linrode
ensprob = EnsembleProblem(prob)
```

```julia linrode
enssol = solve(ensprob, RandomEM(), dt = 1/100, trajectories=1000)
```

```julia linrode
enssumm = EnsembleSummary(enssol; quantiles=[0.25,0.75])
plot(enssumm,
    title = "Ensemble of solution paths of \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$\nwith 50% confidence interval",
    titlefont = 12,
    xaxis = "\$t\$",
    yaxis = "\$x\$"
)
```

## Order of convergence

Now, we use the development tools in [SciML/DiffEqDevTools](https://github.com/SciML/DiffEqDevTools.jl) to set up a set of ensemble solves and obtain the order of convergence from them.

The `WorkPrecisionSet` with `error_estimate=:l∞`, that we use, actually compute the strong norm in the form
```math
\mathbb{E}[\max_{i = 0, \ldots, n} |X_i - X_i^N|].
```
instead of the form
```math
\max_{i=0, \ldots, n}\mathbb{E}[|X_i - X_i^N|].
```

For that, we choose a sequence of time steps and relative and absolute tolerances to check how the error decays along the sequence.

```julia linrode
reltols = 1.0 ./ 10.0 .^ (1:5)
abstols = reltols
dts = 1.0./5.0.^((1:length(reltols)) .+ 1)
N = 1_000
```

With that, we set up and solve the `WorkPrecisionSet`:
```julia linrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
]

wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)
```

There is already a plot recipe for the result of a `WorkPrecisionSet` that displays the order of convergence:
```julia linrode
plot(wp, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12, legend=:topleft)
```

## Benchmark

We complement the above convergence order with a benchmark comparing the Euler method with the tamed Euler method and the Heun method. They all seem to achieve strong order 1, but with the Heun method being a bit more efficient.

```julia linrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
    Dict(:alg=>RandomTamedEM(), :dts => dts)
    Dict(:alg=>RandomHeun(), :dts => dts)
]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)
plot(wp, title="Benchmark with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12)
```

Built-in recipe for order of convergence.
```julia linrode
plot(wp, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12, legend=:topleft)
```

## More

```julia linrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
    Dict(:alg=>RandomTamedEM(), :dts => dts)
    Dict(:alg=>RandomHeun(), :dts => dts)
]
wps = WorkPrecisionSet(EnsembleProblem(prob),abstols,reltols,setups;trajectories=20, numruns=10,maxiters=1e7,error_estimate=:l∞)
plot(wps, title="Benchmark with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12)
```

Built-in recipe for order of convergence.
```julia linrode
plot(wps, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12, legend=:topleft)
```
