# # Population dynamics with harvest
#
# This time we consider a population dynamics model with two types of noise, a geometric Brownian motion process affecting the growth rate and a point Poisson step process affecting the harvest.

# ## The equation

# More precisely, we consider the RODE
# ```math
#   \begin{cases}
#     \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \Lambda_t X_t (r - X_t) - \alpha H_t\frac{2X_t}{r + X_t}, \qquad 0 \leq t \leq T, \\
#   \left. X_t \right|_{t = 0} = X_0,
#   \end{cases}
# ```
# with 
# ```math
#   \Lambda_t = \gamma(1 + \epsilon\sin(G_t))
# ```
# where $\{G_t\}_{t\geq 0}$ is a geometric Brownian motion process and $\{H_t\}_{t \geq 0}$ is a point Poisson step process with Beta-distributed steps.
#
# We fix $\gamma = 1.0$, $\epsilon = 0.3$, and $r = 1.0$. Notice the critical value for the bifurcation oscilates between $\gamma (1 - \epsilon) / 4$ and $\gamma (1 + \epsilon) / 4$, while the harvest term oscillates between 0 and $\alpha$, and we choose $\alpha = \gamma / 2$ so it oscillates below and above the critical value.
#
# We choose a Beta distribution as the step law, with mean a little below $1/2$, so it stays mostly below the critical value, but often above it.
#
# The geometric Brownian motion process is chosen with drift $\mu = 1.0$, diffusion $\sigma = 0.8$ and initial value $y_0 = 1.0$.
#
# The Poisson counter for the point Poisson step process is chosen with rate $\lambda = 15.0,$ while the time interval is chosen with unit time span.
#
# As for the initial condition, we also choose a Beta distribution, so it stays within the growth region.

# We do not have an explicit solution for the equation so we use as target for the convergence an approximate solution via Euler method at a much higher resolution.
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages

using JLD2
using Plots
using Random
using Distributions
using RODEConvergence

# Then we set up the problem parameters.
#
# We set the seed:

rng = Xoshiro(123)
nothing # hide

# The right hand side of the evolution equation:

γ = 0.8
ϵ = 0.3
r = 1.0
α = γ * r^2
params = (γ, ϵ, r, α)

function f(t, x, y, p)
    γ = p[1]
    ϵ = p[2]
    r = p[3]
    α = p[4]
    dx = x > zero(x) ? γ * (1 + ϵ * sin(y[1])) * x * (1 - x / r) - α * y[2] * x / (r + x) : zero(x)
    return dx
end
nothing # hide

# The time interval:

t0, tf = 0.0, 1.0
nothing # hide

# The law for the initial condition:

x0law = Beta(7.0, 5.0)

# The noise parameters:

μ = 1.0
σ = 0.8
y0 = 1.0
noise1 = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)

λ = 15.0
steplaw = Beta(5.0, 7.0)
noise2 = PoissonStepProcess(t0, tf, λ, steplaw)

noise = ProductProcess(noise1, noise2)

# The mesh resolutions:

ntgt = 2^18
ns = 2 .^ (4:9)

# and

nsample = ns[[1, 2, 3, 4]]

# The number of simulations for the Monte Carlo estimate is set to

m = 100
nothing # hide

# And add some information about the simulation, for the caption of the convergence figure.

info = (
    equation = "population dynamics",
    noise = "gBm and step process noises",
    ic = "\$X_0 \\sim \\mathrm{Beta}($(x0law.α), $(x0law.β))\$"
)
nothing # hide

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler()
method = RandomEuler()

# ### Order of convergence

# With all the parameters set up, we build the [`ConvergenceSuite`](@ref):       

suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)

# Then we are ready to compute the errors via [`solve`](@ref):

@time result = solve(rng, suite)
nothing # hide

# The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:
# 

table = generate_error_table(result, suite, info)

println(table) # hide
nothing # hide

# 
# 
# The calculated order of convergence is given by `result.p`:

println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide

#

## save to build/
save(joinpath(@__DIR__(), "results/06-popdyn_result.jld2"), Dict("result" => result)) # hide

# 
# ### Plots
# 
# We illustrate the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt = plot(result)

#

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "order_popdyn_gBmPoisson.pdf")) # hide
nothing # hide

# For the sake of illustration, we plot some approximations of a sample target solution:

plt = plot(suite, ns=nsample)

#

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "sample_popdyn_gBmPoisson.pdf")) # hide
nothing # hide

# We can also visualize the noises associated with this sample solution:

plot(suite, xshow=false, yshow=true, label=["Z_t" "H_t"], linecolor=:auto)

# The gBm noises enters the equation via $\Lambda_t = \gamma(1 + \epsilon\sin(G_t))$. Using the chosen parameters, this noise can be visualized below

plot(suite, xshow=false, yshow= y -> γ * ( 1  + ϵ * sin(y[1]) ), label="\$G_t\$")
