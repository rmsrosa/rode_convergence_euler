```@meta
EditURL = "../../literate/examples/09-risk.jl"
```

# An actuarial risk model

A classical model for the surplus $U_t$ at time $t$ of an insurance company is the Cramér–Lundberg model (see e.g. [Gerber & Shiu (1998)](https://doi.org/10.1080/10920277.1998.10595671)) given by
```math
  U_t = U_0 + \gamma t - \sum_{i=1}^{N_t} C_i
```
where $U_0$ is the initial capital, $\gamma$ is a constant premium rate received from the insurees, $C_i$ is a random variable representing the value of the $i$-th claim paid to a given insuree, and $N_t$ is the number of claims up to time $t$. The process $\{N_t\}_t$ is modeled as a Poisson counter, so that the accumulated claims form a compound Poisson process. It is also common to use inhomogeneous Poisson processes and Hawkes self-exciting process, or combinations of such processes for the incidence of the claim, but the classical model uses a homogeneous Poisson counter.

The model above, however, does not take into account the variability of the premium rate received by the company, nor the investiment of the accumulated reserves, among other things. Several diffusion type models have been proposed to account for these and other factors. We will consider a simple model, with a randomly perturbed premium and with variable rentability.

More precisely, we start by rewriting the above expression as the following jump (or impulse) differential equation
```math
  \mathrm{d}U_t = \gamma\;\mathrm{d}t - \mathrm{d}C_t,
```
where
```math
  C_t = \sum_{i=1}^{N_t} C_i.
```

The addition of an interest rate $r$ leads to
```math
  \mathrm{d}U_t = r U_t \mathrm{d}t + \gamma\;\mathrm{d}t - \mathrm{d}C_t.
```

Assuming a premium rate perturbed by a white noise and assuming the interest rate as a process $\{R_t\}_t$, we find
```math
  \mathrm{d}U_t = R_t U_t\;\mathrm{d}t + \gamma\;\mathrm{d}t + \varepsilon\;\mathrm{d}W_t - \mathrm{d}C_t,
```
so the equation becomes
```math
  \mathrm{d}U_t = (\gamma + R_t U_t)\;\mathrm{d}t + \varepsilon\;\mathrm{d}W_t - \mathrm{d}C_t.
```

Since we can compute exactly the accumulated claims $C_t$, we subtract it from $U_t$ to get rid of the jump term. We also subtract an Ornstein-Uhlenbeck process, in the classical way to transform an SDE into a RODE. So, defining
```math
  X_t = U_t - C_t - O_t
```
where $\{O_t\}_t$ is given by
```math
  \mathrm{d}O_t = -\nu O_t\;\mathrm{d}t + \varepsilon\;\mathrm{d}W_t,
```
we obtain
```math
  \mathrm{d}X_t = (\gamma + R_t U_t)\;\mathrm{d}t + \nu O_t\;\mathrm{d}t = (\gamma + R_t (X_t + C_t + O_t))\;\mathrm{d}t + \nu O_t\;\mathrm{d}t.
```

This leads us to the linear random ordinary differential equation
```math
  \frac{\mathrm{d}X_t}{\mathrm{d}t} = R_t X_t + R_t (C_t + O_t) + \nu O_t + \gamma.
```

This equation has the explicit solution
```math
  X_t = X_0 e^{\int_0^t R_s\;\mathrm{d}s} + \int_0^t e^{\int_s^t R_\tau\;\mathrm{d}\tau} (R_s (C_s + O_s) + \nu O_s + \gamma)\;\mathrm{d}s.
```

As for the interest rate process $\{R_t\}_t$, there is a vast literature with models for it, see e.g. Chapter 3 of [Brigo & Mercurio (2006)](https://doi.org/10.1007/978-3-540-34604-3), in particular Table 3.1. Here, we consider the Dothan model (Section 3.2.2 of the aforementioned reference), which consists simply of a geometric Brownian motion process

```math
  \mathrm{d}R_t = \mu R_t \;\mathrm{d}t + \sigma R_t\;\mathrm{d}t,
```
with $R_t = r_0$, where $\mu, \sigma, r_0$ are positive constants. This has an explicit solution
```math
  R_t = r_0 e^{(\mu - \sigma^2/2)t + \sigma W_t},
```
so that the above equation for $\{X_t\}_t$ is a genuine random ODE.

Once the solution of $\{X_t\}_t$ is obtained, we find an explicit formula for the surplus $X_t = U_t - C_t - O_t$, namely
```math
  U_t = C_t + O_t + X_0 e^{\int_0^t R_s\;\mathrm{d}s} + \int_0^t e^{\int_s^t R_\tau\;\mathrm{d}\tau} (R_s (C_s + O_s) + \nu O_s + \gamma)\;\mathrm{d}s,
```
with $\{R_t\}_t$ as above.

## Numerical simulations

### Setting up the problem

First we load the necessary packages:

````@example 09-risk
using JLD2
using Plots
using Measures
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
````

Then we define the random seed:

````@example 09-risk
rng = Xoshiro(123)
nothing # hide
````

The evolution law:

````@example 09-risk
ν = 5.0
γ = 1.0

params = (ν, γ)

function f(t, x, y, p)
    ν = p[1]
    γ = p[2]
    o = y[1]
    r = y[2]
    c = y[3]
    dx = r * (x + c + o) + ν * o + γ
    return dx
end
nothing # hide
````

The time interval:

````@example 09-risk
t0, tf = 0.0, 3.0
nothing # hide
````

The law for the initial condition:

````@example 09-risk
x0 = 1.0
x0law = Dirac(x0)
````

The Ornstein-Uhlenbeck, geometric Brownian motion, and compound Poisson processes for the noise term:

````@example 09-risk
O0 = 0.0
Oν = 5.0
Oε = 0.8
R0 = 0.2
Rμ = 0.02
Rσ = 0.4
Cmax = 0.2
Cλ = 8.0
Claw = Uniform(0.0, Cmax)
noise = ProductProcess(
    OrnsteinUhlenbeckProcess(t0, tf, O0, Oν, Oε),
    GeometricBrownianMotionProcess(t0, tf, R0, Rμ, Rσ),
    CompoundPoissonProcess(t0, tf, Cλ, Claw)
)
nothing # hide
````

The resolutions for the target and approximating solutions, as well as the number of simulations for the Monte-Carlo estimate of the strong error:

````@example 09-risk
ntgt = 2^18
ns = 2 .^ (6:9)
````

````@example 09-risk
nsample = ns[[1, 2, 3, 4]]
````

The number of simulations for the Monte Carlo estimate is set to

````@example 09-risk
m = 400
nothing # hide
````

And add some information about the simulation, for the caption of the convergence figure.

````@example 09-risk
info = (
    equation = "a risk model",
    noise = "coupled Ornstein-Uhlenbeck, geometric Brownian motion, and compound Poisson processes",
    ic = "\$X_0 = $x0\$"
)
nothing # hide
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 09-risk
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the [`ConvergenceSuite`](@ref):

````@example 09-risk
suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors via [`solve`](@ref):

````@example 09-risk
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 09-risk
table = generate_error_table(result, suite, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 09-risk
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide
````

### Plots

We plot the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 09-risk
plt_result = plot(result)
````

````@example 09-risk
savefig(plt_result, joinpath(@__DIR__() * "../../../../latex/img/", "order_riskmodel.pdf")) # hide
nothing # hide
````

For the sake of illustration of the behavior of the system, we visualize a sample solution

````@example 09-risk
plt_sols = plot(suite, ns=nothing, label="\$X_t\$", linecolor=1)
````

````@example 09-risk
savefig(plt_sols, joinpath(@__DIR__() * "../../../../latex/img/", "evolution_riskmodel.pdf")) # hide
nothing # hide
````

We also illustrate the convergence to a sample solution

````@example 09-risk
plt_suite = plot(suite)
````

````@example 09-risk
savefig(plt_suite, joinpath(@__DIR__() * "../../../../latex/img/", "approximation_riskmodel.pdf")) # hide
nothing # hide
````

We can also visualize the noises associated with this sample solution:

````@example 09-risk
plt_noises = plot(suite, xshow=false, yshow=true, label=["\$O_t\$" "\$R_t\$" "\$C_t\$"], linecolor=[1 2 3])
````

````@example 09-risk
savefig(plt_noises, joinpath(@__DIR__() * "../../../../latex/img/", "riskmodel_noises.pdf")) # hide
nothing # hide
````

The actual surplus is $U_t = X_t - O_t - C_t$, so we may visualize a sample solution of the surplus by subtracting these two noises from the solution of the above RODE.

````@example 09-risk
plt_surplus = plot(range(t0, tf, length=ntgt+1), suite.xt .- suite.yt[:, 1] .- suite.yt[:, 3], xaxis="\$t\$", yaxis="\$u\$", label="\$U_t\$", linecolor=1)
````

````@example 09-risk
savefig(plt_surplus, joinpath(@__DIR__() * "../../../../latex/img/", "riskmodel_surplus.pdf")) # hide
nothing # hide
````

Combining the plots

````@example 09-risk
tt = range(t0, tf, length=8ns[end]+1)

ds = div(ntgt, 8ns[end])

plt_surplus_and_noises = plot(tt, suite.xt[begin:ds:end] .- suite.yt[begin:ds:end, 1] .- suite.yt[begin:ds:end, 3], xaxis="\$t\$", yaxis="\$u\$", label="\$U_t\$", linecolor=1)

plt_surplus_and_noises_twin = twinx(plt_surplus_and_noises)

plot!(plt_surplus_and_noises_twin, tt, suite.yt[begin:ds:end, 1], yaxis="\$\\textrm{noise}\$", label="\$O_t\$", legend=:top, linecolor=2)
plot!(plt_surplus_and_noises_twin, tt, suite.yt[begin:ds:end, 2], label="\$R_t\$", linecolor=3)
plot!(plt_surplus_and_noises_twin, tt, suite.yt[begin:ds:end, 3], label="\$C_t\$", linecolor=4)

plt_combined = plot(plt_result, plt_surplus_and_noises, legendfont=6, size=(800, 240), title=["(a) risk model" "(b) sample paths" ""], titlefont=10, bottom_margin=5mm, left_margin=5mm)
````

````@example 09-risk
savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "riskmodel_combined.pdf")) # hide
nothing # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

