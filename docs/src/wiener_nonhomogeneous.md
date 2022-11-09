# Linear RODE with non-homogenous Wiener noise

We start by considering the Euler approximation of one of the simplest linear random ordinary differential equation, in which the noise is just a Wiener process, as the  nonhomogeneous term.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = - X_t + W_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0.
  \end{cases}
```
This is one of the simplest examples of RODEs and has the explicit solution
```math
X_t = e^{-t}X_0 + \int_0^t e^{-(t - s)} W_s \;\mathrm{d}s.
```

## Computing the exact solution

For estimating the order of convergence, we use the Monte Carlo method, computing a number of numerical approximations of pathwise solutions and taking the average of their absolute differences.

The computed solution is calculated from realizations $W_{t_j}(\omega_k)$, for samples paths $\{W_t(\omega_k)\}_{t\geq 0}$, with $k = 1, \ldots, K,$ and on the mesh points $t_0, \ldots, t_n$. We cannot compute the integral $\int_0^{t_j} e^{-(t_j - s)}W_s\;\mathrm{d}s$ exactly just from the values on the mesh points, but we can compute its expectation. First we break down the sum into parts:
```math
\int_0^{t_j} e^{-(t_j - s)}W_s\;\mathrm{d}s = \sum_{i = 0}^{j-1} \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s.
```
On each mesh interval, we use that
```math
B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
is a Brownian bridge on the interval $[t_i, t_{i+1}]$, independent of $\{W_t\}_{t\geq 0}$ (see [Almost Sure: Brownian Bridges](https://almostsuremath.com/2021/03/29/brownian-bridges/) for more on Brownian bridges). Notice that
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t.
```

Thus,
```math
\begin{align*}
\mathrm{d}(e^{-(t_j-t)}W_t) & = e^{-(t_j-t)}W_t\;\mathrm{d}t + e^{-(t_j-t)}\;\mathrm{d}W_t \\
& = e^{-(t_j-t)}W_t\;\mathrm{d}t + e^{-(t_j-t)} \left(\mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right),
\end{align*}
```
so that
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s & = e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i} - \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}\;\mathrm{d}B_s \\
& \qquad - \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} e^{-(t_j - s)}\;\mathrm{d}s.
\end{align*}
```

Taking the expectation, using that the expectation of an Itô integral with respect to a Brownian bridge with zero endpoints is zero, and using that the values at the mesh points are given, we find that
```math
\mathbb{E}\left[ \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s\right] = e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i} - \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
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

Thus, we estimate the error by calculating the difference from the numerical approximation to the above expectation. Another way of seeing this is that the exact solution is
```math
X_{t_j} = e^{-t_j}X_0 + W_{t_j} - e^{-t_j}\sum_{i=0}^{j-1} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{t_{i+1}} - e^{-t_i}\right) + Z_j,
```
where the random variables $Z_j$ average out to zero, so the strong error to $X_{t_j}$ is the same as that to $\mathbb{E}[X_{t_j}]$.

## Numerical approximation

First we load the necessary packages. We use [SciML/StochasticDiffEq.jl](https://github.com/SciML/StochasticDiffEq.jl) for solving the RODEs, [SciML/DiffEqDevTools.jl](https://github.com/SciML/DiffEqDevTools.jl) for facilitate the benchmark, [JuliaPlots/Plots.jl](https://github.com/JuliaPlots/Plots.jl) for plotting the results, and [Random](https://docs.julialang.org/en/v1/stdlib/Random/) for setting the seeds for reproducibility.

```@example nonhomlinrode
using StochasticDiffEq, DiffEqDevTools, Plots, Random
```

### Setting up the problem

Now we set up the RODE problem. In [SciML/DifferentialEquations](https://diffeq.sciml.ai/stable/), the [Rode Problems](https://diffeq.sciml.ai/stable/types/rode_types/) are given in the form
```math
\frac{\mathrm{d}u}{\mathrm{d}t} = f(u, t, p, W)
```
In our example, the right hand side takes the form
```@example nonhomlinrode
f(u, p, t, W) = u + W
```

Next we define the function that yields the (expected) analytic solution for a given computed solution `sol`, that contains the noise `sol.W` and the info about the (still to be defined) RODE problem `prob`.

In the numerical implementation of the expected exact solution, the summation in the expression can be computed recursively. Indeed, if
```math
I_j = \sum_{i=0}^{j-1} \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{t_{i+1}} - e^{t_i}\right),
```
then $I_0 = 0$ and, for $j = 1, \ldots, n$,
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

Then, we set $X_{t_j}$ to
```math
X_{t_j} = W_{t_j} + e^{-t_j}\left(X_0 + I_j\right).
```

This is implemented below.
```@example nonhomlinrode
function f_analytic!(sol)
    empty!(sol.u_analytic)

    u0 = sol.prob.u0
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

```@example nonhomlinrode
ff = RODEFunction(
    f,
    analytic = f_analytic!,
    analytic_full=true
)
```

Now we set up the RODE problem, with initial condition `X0 = 1.0`, and time span `tspan = (0.0, 1.0)`:

```@example nonhomlinrode
X0 = 1.0
tspan = (0.0, 1.0)

prob = RODEProblem(ff, X0, tspan)
```

### An illustrative sample path

Just for the sake of illustration, we solve for a solution path, using the Euler method, which in SciML is provided as `RandomEM()`, and with a fixed time step. We fix the `seed` just for the sake of reproducibility:
```@example nonhomlinrode
sol = solve(prob, RandomEM(), dt = 1/100, seed = 123)
```

and display the result
```@example nonhomlinrode
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

```@example nonhomlinrode
ensprob = EnsembleProblem(prob)
```

```@example nonhomlinrode
enssol = solve(ensprob, RandomEM(), dt = 1/100, trajectories=1000)
```

```@example nonhomlinrode
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

```@example nonhomlinrode
reltols = 1.0 ./ 10.0 .^ (1:5)
abstols = reltols
dts = 1.0./5.0.^((1:length(reltols)) .+ 1)
N = 1_000
```

With that, we set up and solve the `WorkPrecisionSet`:
```@example nonhomlinrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
]

wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)
```

There is already a plot recipe for the result of a `WorkPrecisionSet` that displays the order of convergence:
```@example nonhomlinrode
plot(wp, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12, legend=:topleft)
```

## Benchmark

We complement the above convergence order with a benchmark comparing the Euler method with the tamed Euler method and the Heun method. They all seem to achieve strong order 1, but with the Heun method being a bit more efficient.

```@example nonhomlinrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
    Dict(:alg=>RandomTamedEM(), :dts => dts)
    Dict(:alg=>RandomHeun(), :dts => dts)
]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)
plot(wp, title="Benchmark with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12)
```

Built-in recipe for order of convergence.
```@example nonhomlinrode
plot(wp, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12, legend=:topleft)
```

## More

```julia nonhomlinrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
    Dict(:alg=>RandomTamedEM(), :dts => dts)
    Dict(:alg=>RandomHeun(), :dts => dts)
]
wps = WorkPrecisionSet(EnsembleProblem(prob),abstols,reltols,setups;trajectories=20, numruns=10,maxiters=1e7,error_estimate=:l∞)
plot(wps, title="Benchmark with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12)
```

Built-in recipe for order of convergence.
```julia nonhomlinrode
plot(wps, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$", titlefont=12, legend=:topleft)
```
