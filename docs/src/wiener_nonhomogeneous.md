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
is a Brownian bridge on the interval $[t_i, t_{i+1}]$, independent of $\{W_t\}_{t\geq 0}$. Notice that
```math
\mathrm{d}W_t = \mathrm{d}B_t + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t.
```

Thus,
```math
\begin{align*}
\mathrm{d}(e^{-(t_j-t)}W_t) & = e^{-(t_j-t)}W_t\;\mathrm{d}t + e^{-(t_j-t)}\;\mathrm{d}W_t \\
& = e^{-(t_j-t)}W_t\;\mathrm{d}t + e^{-(t_j-t)} \left(\mathrm{d}B_t + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\;\mathrm{d}t\right),
\end{align*}
```
so that
```math
\begin{align*}
\int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s & = e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i} + \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}\;\mathrm{d}B_s \\
& \qquad + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\int_{t_i}^{t_{i+1}} e^{-(t_j - s)}\;\mathrm{d}s.
\end{align*}
```

Taking the expectation, using that the expectation of an Itô integral with respect to a Brownian bridge with zero endpoints is zero, and using that the values at the mesh points are given, we find that
```math
\mathbb{E}\left[ \int_{t_i}^{t_{i+1}} e^{-(t_j - s)}W_s\;\mathrm{d}s\right] = e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i} + \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
```

Hence, given the realizations of a Wiener noise on the mesh points,
```math
\begin{align*}
\mathbb{E}[X_{t_j}] & = e^{-t_j}X_0 + \sum_{i=0}^{j-1} \left(e^{-(t_j-t_{i+1})}W_{t_{i+1}} - e^{-(t_j-t_i)}W_{t_i}\right) \\
& \qquad + \sum_{i=0}^{j-1} \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
\end{align*}
```

The first summation telescopes out and, since $W_0 = 0$, we are left with
```math
\mathbb{E}[X_{t_j}] = e^{-t_j}X_0 + W_{t_j} - \sum_{i=0}^{j-1} \frac{W_{t_{t+1}}-W_{t_i}}{t_{i+1}-t_i}\left( e^{-(t_j - t_{i+1})} - e^{-(t_j - t_i)}\right).
```

Thus, we estimate the error by calculating the difference from the numerical approximation to the above expectation.

## Numerical approximation

First we load the necessary packages

```@example linrode
using DiffEqNoiseProcess, StochasticDiffEq, Plots, DiffEqDevTools, Random
```

We set the number of simulations for the Monte Carlo estimate of the error:
```@example linrode
N = 1000
```

### Setting up the problem

Now we set up the RODE problem.

We define the right-hand-side $f=f(u, t, p, W)$ of the equation, in the form
```math
\frac{\mathrm{d}u}{\mathrm{d}t} = f(u, t, p, W)
```

so that
```math
f(u, t, p, W) = p * u + W,
```
with $p = -1.0$.

```@example linrode
f(u, p, t, W) = p * u + W
```

Next we define the function that yields the analytic solution for a given computed solution `sol`, that contains the noise `sol.W` and the info about the (still to be defined) RODE problem `prob`.

```@example linrode
function f_analytic!(sol)
    empty!(sol.u_analytic)

    u0 = sol.prob.u0
    p = sol.prob.p
    push!(sol.u_analytic, u0)

    t(i) = sol.W.t[i]
    W(i) = sol.W.W[i]

    ti1, Wi1 = sol.W.t[1], sol.W.W[1]
    expintegral1 = 1.0
    integral2 = 0.0
    for i in 2:length(sol)
        ti, Wi = sol.W.t[i], sol.W.W[i]
        expaux = exp(p * (ti - ti1))
        expintegral1 *= expaux
        integral2 = expaux * (integral2 + (Wi + Wi1) * (ti - ti1) / 2)
        push!(sol.u_analytic, u0 * expintegral1 + integral2)
        ti1, Wi1 = ti, Wi
    end
end
```

With the right-hand-side and the analytic solutions defined, we construct the `RODEFunction` to be passed on to the RODE problem builder.

```@example linrode
ff = RODEFunction(
    f,
    analytic = f_analytic!,
    analytic_full=true
)
```

Now we set up the RODE problem, with initial condition `X0 = 1.0`, parameter `p = -1.0` and time span `tspan = (0.0, 1.0)`:

```@example linrode
X0 = 1.0
p = - 1.0
tspan = (0.0, 1.0)

prob = RODEProblem(ff, X0, tspan, p)
```

### An illustrative sample path

Just for the sake of illustration, we solve for a solution path, using the Euler method, which in SciML is provided as `RandomEM()`, and with a fixed time step `dt = 1/100`:
```@example linrode
sol = solve(prob, RandomEM(), dt = 1/100)
```

and display the result
```@example linrode
plot(
    sol,
    title = "Sample path of \$X_t = -X_t + W_t\$",
    titlefont = 12,
    xaxis = "\$t\$",
    yaxis = "\$x\$",
    label="\$X_t\$"
)
```

### An illustrative ensemble of solutions

```@example linrode
ensprob = EnsembleProblem(prob)
```

```@example linrode
enssol = solve(ensprob, RandomEM(), dt = 1/100, trajectories=1000)
```

```@example linrode
enssumm = EnsembleSummary(enssol; quantiles=[0.05,0.95])
plot(enssumm,
    title = "Ensemble of solution paths of \$X_t = -X_t + W_t\$",
    titlefont = 12,
    xaxis = "\$t\$",
    yaxis = "\$x\$"
)
```