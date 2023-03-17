# # Non-homogenous linear system of RODEs with all implemented noises
#
# This time we consider a linear system of equations with a noise that combines all the implemented noises.

# ## The equation

# More precisely, we consider the RODE
# ```math
#   \begin{cases}
#     \displaystyle \frac{\mathrm{d}\mathbf{X}_t}{\mathrm{d} t} = - \|\mathbf{Y}_t\|^2 \mathbf{X}_t + \mathbf{Y}_t, \qquad 0 \leq t \leq T, \\
#   \left. \mathbf{X}_t \right|_{t = 0} = \mathbf{X}_0,
#   \end{cases}
# ```
# where $\{\mathbf{Y}_t\}_{t\geq 0}$ is a vector-valued noise with each component being each of the implemented noises.
#
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages

using Plots
using LinearAlgebra
using Random
using Distributions
using RODEConvergence

# Then we set up some variables, as in the first example

rng = Xoshiro(123)

f!(dx, t, x, y) = (dx .= .- sum(abs2, y) .* x .+ y)

t0 = 0.0
tf = 1.0

y0 = 0.2
μ = 0.3
σ = 0.2
ν = 0.3
λ = 10.0
α = 2.0
β = 15.0
λ₀ = 2.0
a = 0.8
δ = 0.9
β̃ = 1.8
dylaw2 = Exponential(1/β̃)
dylaw = Normal(μ, σ)
steplaw = Beta(α, β)
nr = 20
transport(t, r) = mapreduce(ri -> cbrt(sin(t/ri)), +, r) / length(r)
ylaw = Beta(α, β)
hurst = 0.6

ntgt = 2^18
ns = 2 .^ (5:9)
nsample = ns[[1, 2, 3]]
m = 1_000

noise = ProductProcess(
    WienerProcess(t0, tf, y0),
    OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ),
    GeometricBrownianMotionProcess(t0, tf, y0, μ, σ),
    CompoundPoissonProcess(t0, tf, λ, dylaw),
    PoissonStepProcess(t0, tf, λ, steplaw),
    ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw2),
    TransportProcess(t0, tf, ylaw, transport, nr),
    FractionalBrownianMotionProcess(t0, tf, y0, hurst, ntgt)
)

x0law = MvNormal(zeros(length(noise)), I)

# And add some information about the simulation:

info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + W_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)


# The method for which want to estimate the rate of convergence is, naturally, the Euler method:

target = RandomEuler(length(noise))
method = RandomEuler(length(noise))

# ### Order of convergence

# With all the parameters set up, we build the [`ConvergenceSuite`](@ref):       

suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)

# Then we are ready to compute the errors:

@time result = solve(rng, suite)

# The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:
# 

table = generate_error_table(result, info)

println(table) # hide
nothing # hide

# 
# 
# The calculated order of convergence is given by `result.p`:

println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")

# 
# 
# ### Plots
# 
# We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plot(result)

# 

## savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
## nothing # hide

# For the sake of illustration, we plot a sample of an approximation of a target solution:

plot(suite, ns=nsample)

# We can also visualize the noise associated with this sample solution, both individually

plot(suite, xshow=false, yshow=true, linecolor=:auto)

# and combined into a sum

plot(suite, xshow=false, yshow=:sum)
