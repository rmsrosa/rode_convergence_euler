# # Kanai-Tajimi Earthquake model

# Now we consider a single-storey Kanai-Tajimi Earthquake model.

# ## The equation
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

# Then we set up some variables

rng = Xoshiro(123)

function f(dx, t, x, y)
    ζ = 0.64
    ω = 15.56
    dx[1] = -x[2] + y[1]
    dx[2] = -2 * ζ * ω * (x[2] + y[1]) + ω ^ 2 * x[1] + y[1] + y[2]
    return dx
end

t0 = 0.0
tf = 1.0

x0law = MvNormal(zeros(2), I(2))

μ = 1.0
σ = 0.8
y0 = 0.1
noise1 = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ) # It should actually be an Orstein-Uhlenbeck, but apparently I haven't implemented it yet

α = 2.0
β = 15.0
ylaw = Beta(α, β)
nr = 25
g(t, r) = mapreduce(ri -> cbrt(sin(t/ri)), +, r) / length(r)
noise2 = TransportProcess(t0, tf, ylaw, g, nr)

noise = ProductProcess(noise1, noise2)

ntgt = 2^18
ns = 2 .^ (5:9)
nsample = ns[[1, 2, 3, 4]]
m = 1_000

# And add some information about the simulation:

info = (
    equation = "Kanai-Tajimi model",
    noise = "Orstein-Uhlenbeck and Transport Process",
    ic = "\$X_0 \\sim \\mathcal{N}(\\mathbf{0}, I\\_2)\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))

include(@__DIR__() * "/common_end.jl")
