# # A toggle-switch model for gene expression
#
# Here, we consider the toggle-switch model in Section 7.8 of [Asai (2016)](https://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/40146), originated from [Verd, Crombach & Jaeger (2014)](https://doi.org/10.1186/1752-0509-8-43). See also [Strasser, Theis & Marr (2012)](https://doi.org/10.1016/j.bpj.2011.11.4000).

# Toogle switches in gene expression consist of genes that mutually repress each other and exhibit two stable steady states of ON and OFF gene expression. It is a regulatory mechanism which is active during cell differentiation and is believed to act as a memory device, able to choose and maintain cell fate decisions.

# ## The equation

# We consider the following simple model as discussed in [Asai (2016)](https://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/40146), of two interacting genes, $X$ and $Y$, with the concentration of their corresponding protein products at each time $t$ denoted by $X_t$ and $Y_t$. These are stochastic processes defined by the system of equations
# ```math
#   \begin{cases}
#   \frac{\displaystyle \mathrm{d}X_t}{\displaystyle \mathrm{d} t} = \left( A_t + \frac{\displaystyle X_t^4}{\displaystyle a^4 + X_t^4}\right)\left(\frac{\displaystyle b^4}{\displaystyle b^4 + Y_t^4}\right) - \mu X_t, \\ \\ 
#   \frac{\displaystyle \mathrm{d}Y_t}{\displaystyle \mathrm{d} t} = \left( B_t + \frac{\displaystyle Y_t^4}{\displaystyle c^4 + Y_t^4}\right)\left(\frac{\displaystyle d^4}{\displaystyle d^4 + X_t^4}\right) - \nu Y_t, \\ \\
#   \left. X_t \right|_{t = 0} = X_0, \\ \\
#   \left. Y_t \right|_{t = 0} = Y_0,
#   \end{cases}
# ```
# 
# where $\{A_t\}_{t\geq 0}$ and $\{B_t\}_{t\geq 0}$ are given stochastic process representing the external activation on each gene; $a$ and $c$ determine the auto-activation thresholds; $b$ and $d$ determine the thresholds for mutual repression; and $\mu$ and $\nu$ are protein decay rates. In this model, the external activation $A_t$ is a compound Poisson processes (cP), while $B_t$ is a time-dependent version of the geometric Brownian motion process (tgBm).
#
# In the simulations below, we use the following parameters: We fix $a = c = 0.25$; $b = d = 0.4$; and $\mu = \nu = 0.75.$ The initial conditions are set to $X_0 = Y_0 = 4.0.$ The external activation $\{A_t\}_t$ is a compound Poisson process with Poisson rate $\lambda = 5.0$ and jumps uniformly distributed on $[0.0, 0.5]$. The external activation $\{B_t\}_t$ is a non-autonomous version of a geometric Brownian motion process given by
# ```math
#   \mathrm{d}B_t = (\mu_1 + \mu_2\sin(\omega t))B_t\;\mathrm{d}t + \sigma\sin(\omega t)B_t\;\mathrm{d}W_t,
# ```
# and we choose $\mu_1 = 0.7,$ $\mu_2 = 0.3,$ $\sigma = 0.3,$ and $\omega=3\pi,$ with initial condition $A_0 = 0.2.$
#
# We don't have an explicit solution for the equation so we just use as target for the convergence an approximate solution via Euler method at a much higher resolution.
#
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages:

using Plots
using Measures
using Random
using LinearAlgebra
using Distributions
using RODEConvergence

# Then we define the random seed:

rng = Xoshiro(123)
nothing # hide

# The evolution law:

a⁴ = c⁴ = 0.25 ^ 4
b⁴ = d⁴ = 0.4 ^ 4
μ = ν = 0.75

params = (a⁴, c⁴, b⁴, d⁴, μ,  ν)

function f!(dx, t, x, y, p)
    a⁴, c⁴, b⁴, d⁴, μ, ν = p
    α = y[1]
    β = y[2]
    x₁⁴ = x[1]^4
    x₂⁴ = x[2]^4
    dx[1] = ( α + x₁⁴ / (a⁴  + x₁⁴) ) * ( b⁴ / ( b⁴ + x₂⁴)) - μ * x[1]
    dx[2] = ( β + x₂⁴ / (c⁴  + x₂⁴) ) * ( d⁴ / ( d⁴ + x₁⁴)) - ν * x[1]
    return dx
end
nothing # hide

# The time interval:

t0, tf = 0.0, 5.0
nothing # hide

# The law for the initial condition:

x0 = 4.0
y0 = 4.0
x0law = product_distribution(Dirac(x0), Dirac(y0))

# The compound Poisson and the geometric Brownian motion processes, for the noisy source terms:

BM = 0.5
Bλ = 5.0
Bylaw = Uniform(0.0, BM)
Aμ1 = 0.7
Aμ2 = 0.3
Aσ = 0.3
Aω = 3π
A0 = 0.2
Aprimitive_a = t -> Aμ1 * t - Aμ2 * cos(Aω * t) / Aω
Aprimitive_b2 = t -> Aσ^2 * ( t/2 - sin(Aω * t) * cos(Aω * t) / 2Aω )
noise = ProductProcess(
    CompoundPoissonProcess(t0, tf, Bλ, Bylaw),
    HomogeneousLinearItoProcess(t0, tf, A0, Aprimitive_a, Aprimitive_b2)
)

# The resolutions for the target and approximating solutions, as well as the number of simulations for the Monte-Carlo estimate of the strong error

ntgt = 2^18
ns = 2 .^ (5:9)

#

nsample = ns[[1, 2, 3, 4]]

# The number of simulations for the Monte Carlo estimate is set to

m = 100
nothing # hide

# And add some information about the simulation, for the caption of the convergence figure.

info = (
    equation = "a toggle-switch model of gene regulation",
    noise = "coupled compound Poisson process and geometric Brownian motion noises",
    ic = "\$X_0 = $x0, Y_0 = $y0\$"
)
nothing # hide

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))

# ### Order of convergence

# With all the parameters set up, we build the [`ConvergenceSuite`](@ref):       

suite = ConvergenceSuite(t0, tf, x0law, f!, noise, params, target, method, ntgt, ns, m)

# Then we are ready to compute the errors via [`solve`](@ref):

@time result = solve(rng, suite)
nothing # hide

# The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:
# 

table = generate_error_table(result, suite, info)

println(table) # hide
nothing # hide

# 
# The calculated order of convergence is given by `result.p`:

println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide

# 
# ### Plots
# 
# We plot the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt_result = plot(result)

#

savefig(plt_result, joinpath(@__DIR__() * "../../../../latex/img/", "order_toggleswitch.pdf")) # hide
nothing # hide

# For the sake of illustration of the behavior of the system, we rebuild the problem with a longer time step and do a single run with it, for a single sample solution.

t0, tf = 0.0, 10.0
m = 1
ns = 2 .^ (6:9)
suite = ConvergenceSuite(t0, tf, x0law, f!, noise, params, target, method, ntgt, ns, m)
@time result = solve(rng, suite)
nothing # hide

# We visualize the pair of solutions

plt_sols = plot(suite, xshow=1, ns=nothing, label="\$X_t\$", linecolor=1)
plot!(plt_sols, suite, xshow=2, ns=nothing, label="\$Y_t\$", linecolor=2, ylabel="\$\\textrm{concentration}\$")

#

savefig(plt_sols, joinpath(@__DIR__() * "../../../../latex/img/", "evolution_toggleswitch.pdf")) # hide
nothing # hide

# We also illustrate the convergence to say the expression of the X gene:

plt_suite = plot(suite, legend=:top)

#

savefig(plt_suite, joinpath(@__DIR__() * "../../../../latex/img/", "approximation_toggleswitch.pdf")) # hide
nothing # hide

# We can also visualize the noises associated with this sample solution:

plt_noises = plot(suite, xshow=false, yshow=true, label=["\$A_t\$" "\$B_t\$"], linecolor=[1 2], linestyle=:auto)

#

savefig(plt_noises, joinpath(@__DIR__() * "../../../../latex/img/", "noises_toggleswitch.pdf")) # hide
nothing # hide

# We finally combine all plots into a single one, for a visual summary.

plot(plt_result, plt_sols, plt_suite, plt_noises, size=(800, 600), title=["(a) toggle-switch model" "(b) sample solution" "(c) Euler approximations" "(d) sample noises"], titlefont=10, legendfont=7)

# We also combine just some of them, for the article

plt_combined = plot(plt_result, plt_suite, size=(800, 240), title=["(a) toggle-switch model" "(b) Euler approximations"], titlefont=10, layout = (1, 2), legendfont=7, bottom_margin=5mm, left_margin=5mm)

#

savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "toggleswitch_combined.pdf")) # hide
nothing # hide
