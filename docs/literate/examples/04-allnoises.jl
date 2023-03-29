# # Non-homogenous linear system of RODEs with all implemented noises
#
# ```@meta
# Draft = false
# ```
#
# This time we consider a linear *system* of equations with a vector-valued noise made of all the implemented noises.

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
# Again, the *target* solution is construced by solving the system via Euler method at a much higher resolution.
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

# Then we set up some variables, as in the first example. First, the RNG

rng = Xoshiro(123)

# Next the right hand side of the system of equations, in the *in-place* version, for the sake of performance. Here, the vector variable `dx` is updated with the right hand side of the system of equations. The norm squared of the noise `y` at a given time `t` is computed via `sum(abs2, y)`.

f!(dx, t, x, y) = (dx .= .- sum(abs2, y) .* x .+ y)

# The time interval is given by the end points

t0, tf = 0.0, 1.0

# and the mesh parameters are set to 

ntgt = 2^18
ns = 2 .^ (6:9)
nsample = ns[[1, 2, 3]]

# The number of simulations for the Monte Carlo estimate is set to

m = 200

# Now we define all the noise parameters

y0 = 0.2
μ = 0.3
σ = 0.5
ν = 0.3
λ = 5.0
α = 2.0
β = 15.0
λ₀ = 3.0
a = 2.0
δ = 3.0
dylaw = Exponential(0.5)
steplaw = Uniform(0.0, 1.0)
nr = 6
transport(t, r) = mapreduce(ri -> cbrt(sin(ri * t)), +, r) / length(r)
ylaw = Gamma(7.5, 2.0)
hurst = 0.6

# The noise is, then, defined as a (vectorial) [`ProductProcess`](@ref), where each coordinate is one of the implemented noise types:

noise = ProductProcess(
    WienerProcess(t0, tf, 0.0),
    OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ),
    GeometricBrownianMotionProcess(t0, tf, y0, μ, σ),
    CompoundPoissonProcess(t0, tf, λ, dylaw),
    PoissonStepProcess(t0, tf, λ, steplaw),
    ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw),
    TransportProcess(t0, tf, ylaw, transport, nr),
    FractionalBrownianMotionProcess(t0, tf, y0, hurst, ntgt)
)

# Notice we chose the hurst parameter of the fractional Brownian motion process to be between 1/2 and 1, so that the strong convergence is also expected to be of order 1, just like for the other types of noises in `noise`. Both the Wiener and the Orsntein-Uhlenbeck processes are additive noises so the strong order 1 convergence is expected. All the others would be thought to have a lower order of convergence but our results prove they are still of order 1. Hence, their combination is also expected to be of order 1, as illustrated here.
#

# Now we set up the initial condition, taking into account the number of equations in the system, which is determined by the dimension of the vector-valued noise.

x0law = MvNormal(zeros(length(noise)), I)

# We finally add some information about the simulation:

info = (
    equation = "\$\\mathrm{d}\\mathbf{X}_t/\\mathrm{d}t = - \\| \\mathbf{Y}_t\\|^2 \\mathbf{X}_t + \\mathbf{Y}_t\$",
    noise = "vector-valued noise \$\\{Y_t\\}_t\$ with all the implemented noises",
    ic = "\$X_0 \\sim \\mathcal{N}(\\mathbf{0}, \\mathrm{I})\$"
)

# The method for which want to estimate the rate of convergence is, naturally, the Euler method, which is also used to compute the *target* solution, at the finest mesh:

target = RandomEuler(length(noise))
method = RandomEuler(length(noise))

# ### Order of convergence

# With all the parameters set up, we build the [`ConvergenceSuite`](@ref):       

suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)

# Then we are ready to compute the errors via [`solve`](@ref):

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
nothing # hide

# 
# ### Plots
# 
# We illustrate the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt = plot(result)

# and we save the plot for inclusion in the article

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "order_allnoises.png")) # hide
nothing # hide

# For the sake of illustration, we plot a sample of the norms of a sequence of approximations of a target solution, along with the norm of the target:

plts = [plot(suite, ns=nsample, xshow=i, resolution=2^4, title="Coordinate $i", titlefont=8) for i in axes(suite.xt, 2)]

plot(plts..., legend=false)

# We can also visualize the noises associated with this sample solution, both individually, as they enter the non-homogenous term,

plot(suite, xshow=false, yshow=true, linecolor=:auto)

# and combined, with their sum squared, as it enters the homogenous term,

plot(suite, xshow=false, yshow= y -> sum(abs2, y), label="sum of squares of noises")
