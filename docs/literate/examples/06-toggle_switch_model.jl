# # A toggle-switch model for gene-expression with compound Poisson external activation process

# Here, we consider the toggle-switch model in Section 7.8 of [Asai (2016)](https://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/40146), originated from [Verd, Crombach & Jaeger (2014)](https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-8-43).

# ## The equation

# The equation takes the form
# ```math
#   \begin{cases}
#     \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \lambda(1 + \epsilon\sin(G_t)) X_t (r - X_t) - \alpha H_t, \qquad 0 \leq t \leq T, \\
#   \left. X_t \right|_{t = 0} = X_0,
#   \end{cases}
# ```
# where $\{G_t\}_{t\geq 0}$ is a geometric Brownian motion process and $\{H_t\}_{t \geq 0}$ is a point Poisson step process.
#
# For the sake of simplicity, we fix $\lambda = 10.0$, $\epsilon = 0.3$, $r = 1.0$, and $\alpha = 0.05$. Notice the critical value for the bifurcation oscilates between $\lambda (1 - \epsilon) / 4$ and $\lambda (1 + \epsilon) / 4$, while the harvest term oscillates between 0 and $\alpha$, so we choose $\alpha = \lambda / 2$ so it oscillates below and above the critical value.
# More precisely, we choose a Beta distribution as the step law, with mean a little below $1/2$, so it stays mostly below the critical value, but often above it.
#
# The geometric Brownian motion process is chosen with drift $\mu = 1$, diffusion $\sigma = 0.8$ and initial value $y_0 = 0.1$.
#
# The point Poisson step process
#
# As for the initial condition, we choose a Beta distribution with shape parameter 

# We don't have an explicit solution for the equation so we just use as target for the convergence an approximate solution via Euler method at a much higher resolution.
#
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages

using Plots
using Random
using LinearAlgebra
using Distributions
using RODEConvergence

# Then we define the random seed

rng = Xoshiro(123)

# The evolution law

function f!(dx, t, x, y)
    a⁴ = c⁴ = 0.25 ^ 4
    b⁴ = d⁴ = 0.4 ^ 4
    λ₁ = λ₂ = 1.25
    η₁, η₂ = y
    x₁⁴ = x[1]^4
    x₂⁴ = x[2]^4
    dx[1] = ( η₁ + x₁⁴ / (a⁴  + x₁⁴) ) * ( b⁴ / ( b⁴ + x₂⁴)) - λ₁ * x[1]
    dx[2] = ( η₂ + x₂⁴ / (c⁴  + x₂⁴) ) * ( d⁴ / ( d⁴ + x₁⁴)) - λ₂ * x[1]
    return dx
end

# The time interval

t0, tf = 0.0, 4.0

# The law for the initial condition

α = 5.0
β = 5.0
x0law = product_distribution(Beta(α, β), Beta(α, β))

# The compound Poisson processes for the source terms

λ = 12.0
ylaw = Uniform(0.0, 0.5)
noise = ProductProcess(CompoundPoissonProcess(t0, tf, λ, ylaw), CompoundPoissonProcess(t0, tf, λ, ylaw))

# The resolutions for the target and approximating solutions, as well as the number of simulations for the Monte-Carlo estimate of the strong error

ntgt = 2^18
ns = 2 .^ (4:9)
nsample = ns[[1, 2, 3, 4]]
m = 1_000

# And add some information about the simulation:

info = (
    equation = "toggle-swith model of gene regulation",
    noise = "compound Poisson process noises",
    ic = "\$X_0 \\sim \\mathcal{B}($(round(α, sigdigits=1)), $(round(β, sigdigits=1)))^2\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))

# ### Order of convergence

# With all the parameters set up, we build the [`ConvergenceSuite`](@ref):       

suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)

# Then we are ready to compute the errors:

@time result = solve(rng, suite)

nothing # hide

# The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:
# 

table = generate_error_table(result, info)

println(table) # hide
nothing # hide

# 
# The calculated order of convergence is given by `result.p`:

println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")

# 
# 
# ### Plots
# 
# We plot the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plot(result)

# 

# savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
# nothing # hide

# For the sake of illustration, we plot a sample of an approximation of a target solution:

plot(suite, ns=nsample)

# We can also visualize the noises associated with this sample solution:

plot(suite, xshow=false, yshow=true, label=["Compound Poisson noise 1" "Compound Poisson noise 2"])
