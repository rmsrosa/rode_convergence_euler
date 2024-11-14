# # Linear system with all implemented noises
# 
# This time we consider linear equations in two different ways. Either a series of scalar equations each with a different noise or a system of equations with a vector-valued noise composed of all the implemented noises.

# ## The equation

# More precisely, we consider the RODE
# ```math
#   \begin{cases}
#     \displaystyle \frac{\mathrm{d}{X}_t}{\mathrm{d} t} = - \|{Y}_t\|^2 {X}_t + {Y}_t, \qquad 0 \leq t \leq T, \\
#   \left. {X}_t \right|_{t = 0} = {X}_0,
#   \end{cases}
# ```
# where $\{{Y}_t\}_{t\geq 0}$ is either a scalar noise with one of the many implemented noises or a vector-valued noise with all the noises, each as an independent component of the vector process.
#
# Again, the *target* solution is construced by solving the system via Euler method at a much higher resolution.
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages

using Plots
using Measures
using LinearAlgebra
using Random
using Distributions
using RODEConvergence

# Then we set up some variables, as in the first example. First, the RNG:

rng = Xoshiro(123)
nothing # hide

# The time interval is given by the end points

t0, tf = 0.0, 1.0
nothing # hide

# and the mesh parameters are set to 

ntgt = 2^18
ns = 2 .^ (6:9)

# and 

nsample = ns[[1, 2, 3]]

# The number of simulations for the Monte Carlo estimate is set to

m = 92
nothing # hide

# Now we define all the noise parameters

y0 = 0.2

ν = 0.3
σ = 0.5

μ = 0.3

μ1 = 0.5
μ2 = 0.3
ω = 3π
primitive_a = t -> μ1 * t - μ2 * cos(ω * t) / ω
primitive_b2 = t -> σ^2 * ( t/2 - sin(ω * t) * cos(ω * t) / 2ω )

λ = 5.0
dylaw = Exponential(0.5)

steplaw = Uniform(0.0, 1.0)

λ₀ = 3.0
a = 2.0
δ = 3.0

nr = 6
transport(t, r) = mapreduce(ri -> cbrt(sin(ri * t)), +, r) / length(r)
ylaw = Gamma(7.5, 2.0)

hurst = 0.6
nothing # hide

# The noise is, then, defined as a (vectorial) [`ProductProcess`](@ref), where each coordinate is one of the implemented noise types:

noise = ProductProcess(
    WienerProcess(t0, tf, 0.0),
    OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ),
    GeometricBrownianMotionProcess(t0, tf, y0, μ, σ),
    HomogeneousLinearItoProcess(t0, tf, y0, primitive_a, primitive_b2),
    CompoundPoissonProcess(t0, tf, λ, dylaw),
    PoissonStepProcess(t0, tf, λ, steplaw),
    ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw),
    TransportProcess(t0, tf, ylaw, transport, nr),
    FractionalBrownianMotionProcess(t0, tf, y0, hurst, ntgt)
)
nothing # hide

# Both the Wiener and the Orsntein-Uhlenbeck processes are additive noises so the strong order 1 convergence for them is not a surprise based on previous work since the Euler method coincides with the Euler-Maruyama method which is known to be of order 1 for additive noises.

# The geometric Brownian motion noise and the time-dependent linear Itô diffusion noise are multiplicative noises and were also known to yield strong order 1 convergence since the Euler method for the RODE coincides also with the Milstein method for SDEs, which is known to be of order 1.
#
# Another way of addressing the geometric Brownian motion process as a noise for a RODE is to observe that it, say $\{G_t\}_t,$ is given explicitly in the form $G_t = g(t, W_t)$ for a Wiener process $\{W_t\}_{t}$, so the Euler method for the associated RODE coincides with the Euler method for an associated RODE with additive noise $\{W_t\}_t.$ However, the corresponding nonlinear term does not have global Lipschitz bound, so one needs to be careful with this interpretation. Our results, however, apply without further assumptions.
#
# All the other noises, however, would be thought to have a lower order of convergence but our results prove they are also of order 1. Hence, their combination is also expected to be of order 1, as illustrated here.
#
# Notice we chose the hurst parameter of the fractional Brownian motion process to be between 1/2 and 1, so that the strong convergence is also of order 1, just like for the other types of noises in `noise`. Previous knowledge would expect a strong convergence of order $H$, with $1/2 < H < 1,$ instead.
#
# ### The system with all noises combined
#
# Now we define the right hand side of the system of equations, in the *in-place* version, for the sake of performance. Here, the vector variable `dx` is updated with the right hand side of the system of equations. The norm squared of the noise `y` at a given time `t` is computed via `sum(abs2, y)`.

f!(dx, t, x, y, p) = (dx .= .- sum(abs2, y) .* x .+ y)

params = nothing

# The initial condition for the system takes into account the number of equations in the system, which is determined by the dimension of the vector-valued noise.

x0law = MvNormal(zeros(length(noise)), I)

# We now add some information about the simulation, for the caption of the convergence figure.

info = (
    equation = "\$\\mathrm{d}{X}_t/\\mathrm{d}t = - \\| {Y}_t\\|^2 {X}_t + {Y}_t\$",
    noise = "vector-valued noise \$\\{{Y}_t\\}_t\$ with all the implemented noises",
    ic = "\${X}_0 \\sim \\mathcal{N}({0}, \\mathrm{I})\$"
)
nothing # hide

# The method for which want to estimate the rate of convergence is, naturally, the Euler method, which is also used to compute the *target* solution, at the finest mesh:

target = RandomEuler(length(noise))
method = RandomEuler(length(noise))

# #### Order of convergence for the system with all the noises

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
# #### Plots
# 
# We illustrate the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt_result = plot(result)

#

savefig(plt_result, joinpath(@__DIR__() * "../../../../latex/img/", "order_allnoises.pdf")) # hide
nothing # hide

# For the sake of illustration, we plot a sample of the norms of a sequence of approximations of a target solution, along with the norm of the target:

plts = [plot(suite, ns=nsample, xshow=i, resolution=2^4, title="Coordinate $i", titlefont=8) for i in axes(suite.xt, 2)]

plot(plts..., legend=false)

# We can also visualize the noises associated with this sample solution, both individually, as they enter the non-homogenous term,

plt_noises = plot(suite, xshow=false, yshow=true, linecolor=:auto, label=["W" "OU" "gBm" "hlp" "cP" "sP" "H" "T" "fBm"], legend=:topright)

#

savefig(plt_noises, joinpath(@__DIR__() * "../../../../latex/img/", "noisepath_allnoises.pdf")) # hide
nothing # hide

# and combined, with their sum squared, as it enters the homogenous term,

plot(suite, xshow=false, yshow= y -> sum(abs2, y), label="\$\\left\\|\\left\\|{Y}_t\\right\\|\\right\\|^2\$")

# We finally combine all the convergence plot and the noises into a single plot, for the article:

plt_noises = plot(suite, xshow=false, yshow=true, linecolor=:auto, legend=nothing)

plt_combined = plot(plt_result, plt_noises, legendfont=6, size=(800, 240), title=["(a) non-homogeneous linear system" "(b) sample paths of all noises"], titlefont=10, bottom_margin=5mm, left_margin=5mm)

#

savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "allnoises_combined.pdf")) # hide
nothing # hide

# ### Scalar equations with the individual noises

# Now we simulate a series of Random ODEs, each with one of the noises above, instead of the system with all combined noises.

# In the univariate case, the right hand side of the equation becomes

f(t, x, y, p) = - y^2 * x + y

# The initial condition is also univariate

eachx0law = Normal()

# and so is the Euler method

eachtarget = RandomEuler()
eachmethod = RandomEuler()

# Now we compute the error for each noise and gather the order of convergence in a vector.

ps = Float64[result.p]
pmins = Float64[result.pmin]
pmaxs = Float64[result.pmax]
noises = String["all noises combined"]

for eachnoise in noise.processes
    eachsuite = ConvergenceSuite(t0, tf, eachx0law, f, eachnoise, params, eachtarget, eachmethod, ntgt, ns, m)

    @time eachresult = solve(rng, eachsuite)
    
    @info "noise = $(typeof(eachnoise)) => p = $(eachresult.p) ($(eachresult.pmin), $(eachresult.pmax))"

    push!(noises, string(nameof(typeof(eachnoise)))[begin:end-7])
    push!(ps, eachresult.p)
    push!(pmins, eachresult.pmin)
    push!(pmaxs, eachresult.pmax) 
end

# We print them out for inclusing in the paper:

noises_short = ["all"; "W"; "OU"; "gBm"; "hlp"; "cP"; "sP"; "H"; "T"; "fBm"]

for (noisej, noiseshortj, pj, pminj, pmaxj) in zip(noises, noises_short, ps, pmins, pmaxs)
    println("$noisej ($noiseshortj) & $(round(pj, sigdigits=6)) & $(round(pminj, sigdigits=6)) & $(round(pmaxj, sigdigits=6)) \\\\")
end

# The following plot helps in visualizing the result.

plt_eachnoise = plot(ylims=(-0.1, 1.5), ylabel="\$p\$", guidefont=10, legend=:bottomright)
scatter!(plt_eachnoise, noises_short, ps, yerror=(ps .- pmins, pmaxs .- ps), xrotation = 30, label="computed")
hline!(plt_eachnoise, [1.0], linestyle=:dash, label="theory",bottom_margin=5mm, left_margin=5mm)

# Strong order $p$ of convergence of the Euler method for $\mathrm{d}X_t/\mathrm{d}t = - Y_t^2 X_t + Y_t$ for a series of different noise $\{Y_t\}_t$ (scattered dots: computed values; dashed line: expected $p = 1;$ with 95% confidence interval).

savefig(plt_eachnoise, joinpath(@__DIR__() * "../../../../latex/img/", "order_dep_on_noise_allnoises.pdf")) # hide
nothing # hide

# Combined with the noise sample paths:

plt_combined = plot(plt_eachnoise, plt_noises, legendfont=6, size=(800, 240), title=["(a) non-homogeneous linear system" "(b) sample paths of all noises"], titlefont=10, bottom_margin=5mm, left_margin=5mm)

#

savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "allnoises_combined.pdf")) # hide
nothing # hide
