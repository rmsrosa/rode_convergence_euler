# # Modified Kanai-Tajimi earthquake model

# Now we consider an example based on the single-storey Kanai-Tajimi Earthquake model. The model uses a white noise source of groundshake excitations driving a mechanical structure. This can be turned into a Random ODE by means of an Orstein-Uhlenbeck process. Similar models use a transport process exciting a few specific frequencies. The white noise enters the stochastic equation as an additive noise and in this case the order of convergence is known to be of first order. On the other hand, this particular transport process also leads to a first order convergence since the sample paths are smooth. Since the main point is not the model itself but rather the convergence even for rough noises, we modify the problem a bit and use a geometric Brownian motion process, which is a multiplicative noise, together with a transport process with Hölder continuous sample paths. Our results show that we still get the order 1 convergence, which is illustrated in the simulations performed here.

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

μ = -1.0
σ = 0.8
y0 = 1.0
noise1 = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ) # It should actually be an Orstein-Uhlenbeck, but apparently I haven't implemented it yet

α = 2.0
β = 15.0
ylaw = Beta(α, β)
nr = 12
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

# ### Order of convergence

# With all the parameters set up, we build the convergence suite:     

suite = ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)

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

# savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
# nothing # hide

# For the sake of illustration, we plot a sample of an approximation of a target solution:

plot(suite, ns=nsample)

# We can also visualize the noise separate or combined:

plot(suite, shownoise=true, showapprox=false, showtarget=false)

plot(suite, shownoise=true, showapprox=false, showtarget=false, noiseidxs = :sum)
