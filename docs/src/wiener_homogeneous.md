# Homogenous linear RODE with a Wiener process as coefficient

Now we consider a homogeneous linear equation in which the coefficient is a Wiener process.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = W_t X_t, \qquad 0 \leq t \leq T, \\
    \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a Wiener process. The exact solution reads
```math
X_t = e^{\int_0^t W_s \;\mathrm{d}s}X_0.
```

## Computing the exact solution

Similarly to the example of [Nonhomogenous Wiener noise](wiener_nonhomogeneous.md), we cannot compute the integral $\int_0^{t_j} W_s\;\mathrm{d}s$ exactly just from the values $W_{t_j}$ of the noise on the mesh points, but we can compute its expectation.

As before, on each mesh interval, we consider the Brownian bridge on the interval $[t_i, t_{i+1}]$,
```math
B_t = W_t - W_{t_i} - \frac{t - t_i}{t_{i+1}-t_i}(W_{t_{i+1}} - W_{t_i})
```
which yields
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t.
```

This time, we use that
```math
\begin{align*}
\mathrm{d}(tW_t) & = W_t\;\mathrm{d}t + t\;\mathrm{d}W_t \\
& = W_t\;\mathrm{d}t + t \left(\mathrm{d}B_t + \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right),
\end{align*}
```
so that
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} W_s\;\mathrm{d}s & = t_{i+1}W_{t_{i+1}} - t_iW_{t_i} - \int_{t_i}^{t_{i+1}} s\;\mathrm{d}B_s - \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} s\;\mathrm{d}s \\
& = t_{i+1}W_{t_{i+1}} - t_iW_{t_i} - \int_{t_i}^{t_{i+1}} s\;\mathrm{d}B_s - \frac{W_{t_{i+1}}-W_{t_i}}{t_{i+1}-t_i}\frac{t_{i+1}^2 - t_i^2}{2} \\
& = \frac{1}{2}(W_{t_{i+1}}+W_{t_i})(t_{i+1}-t_i) + Z_i,
\end{align*}
```
where
```math
Z_i = - \int_{t_i}^{t_{i+1}} s\;\mathrm{d}B_s.
```
Notice the first term is precisely the trapezoidal rule. Morever, $Z_i$ is a normal variable with zero expectation and variance ... 

## Numerical approximation

As before, we first we load the necessary packages.
```@example homlinrode
using StochasticDiffEq, DiffEqDevTools, Plots, Random
```

### Setting up the problem

Now we set up the RODE problem. We define the right hand side of the equation as
```@example homlinrode
f(u, p, t, W) = W * u
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
```@example homlinrode
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

```@example homlinrode
ff = RODEFunction(
    f,
    analytic = f_analytic!,
    analytic_full=true
)
```

Now we set up the RODE problem, with initial condition `X0 = 1.0`, and time span `tspan = (0.0, 1.0)`:

```@example homlinrode
X0 = 1.0
tspan = (0.0, 1.0)

prob = RODEProblem(ff, X0, tspan)
```

### An illustrative sample path

Just for the sake of illustration, we solve for a solution path:
```@example homlinrode
sol = solve(prob, RandomEM(), dt = 1/100, seed = 123)
```
and display the result
```@example homlinrode
plot(
    sol,
    title = "Sample path of \$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    titlefont = 12,
    xaxis = "\$t\$",
    yaxis = "\$x\$",
    label="\$X_t\$"
)
```

### An illustrative ensemble of solutions

```@example homlinrode
ensprob = EnsembleProblem(prob)
```

```@example homlinrode
enssol = solve(ensprob, RandomEM(), dt = 1/100, trajectories=1000)
```

```@example homlinrode
enssumm = EnsembleSummary(enssol; quantiles=[0.25,0.75])
plot(enssumm,
    title = "Ensemble of solution paths of \$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$\nwith 50% confidence interval",
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

```@example homlinrode
reltols = 1.0 ./ 10.0 .^ (1:5)
abstols = reltols
dts = 1.0./5.0.^((1:length(reltols)) .+ 1)
N = 1_000
```

With that, we set up and solve the `WorkPrecisionSet`:
```@example homlinrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
]

wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)
```

There is already a plot recipe for the result of a `WorkPrecisionSet` that displays the order of convergence:
```@example homlinrode
plot(wp, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = W_tX_t\$", titlefont=12, legend=:topleft)
```

## Benchmark

We complement the above convergence order with a benchmark comparing the Euler method with the tamed Euler method and the Heun method. They all seem to achieve strong order 1, but with the Heun method being a bit more efficient.

```@example homlinrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
    Dict(:alg=>RandomTamedEM(), :dts => dts)
    Dict(:alg=>RandomHeun(), :dts => dts)
]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)
plot(wp, title="Benchmark with \$\\mathrm{d}X_t/\\mathrm{d}t = W_tX_t\$", titlefont=12)
```

Built-in recipe for order of convergence.
```@example homlinrode
plot(wp, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = W_tX_t\$", titlefont=12, legend=:topleft)
```

## More

```julia homlinrode
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
    Dict(:alg=>RandomTamedEM(), :dts => dts)
    Dict(:alg=>RandomHeun(), :dts => dts)
]
wps = WorkPrecisionSet(EnsembleProblem(prob),abstols,reltols,setups;trajectories=20, numruns=10,maxiters=1e7,error_estimate=:l∞)
plot(wps, title="Benchmark with \$\\mathrm{d}X_t/\\mathrm{d}t = W_tX_t\$", titlefont=12)
```

Built-in recipe for order of convergence.
```julia homlinrode
plot(wps, view=:dt_convergence,title="Strong convergence with \$\\mathrm{d}X_t/\\mathrm{d}t = W_tX_t\$", titlefont=12, legend=:topleft)
```
