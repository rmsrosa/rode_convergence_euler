# # Random Fisher-KPP partial differential equation

# Here we simulate a Fisher-KPP equation with random coefficients as considered by [Salako & Shen (2020)](https://link.springer.com/article/10.1007/s10884-020-09847-2). We use the method of lines with finite differences to approximate the partial differential equation by a system of random ODEs.
#
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

function f!(dx, t, x, y)
    axes(x, 1) isa Base.OneTo || error("indexing of `x` should be Base.OneTo")

    dx² = (length(x) - 1)^2
    dx[begin+1:end-1] .= ( x[begin:end-2] - 2x[begin+1:end-1] + x[begin+2:end] ) / dx² .+ y .* x[begin+1:end-1] .* ( 1 .- x[begin+1:end-1] )

    dx[end] = dx[begin] = ( x[end - 1] - 2x[begin] + x[begin+1]) / dx² .+ y .* x[begin] * ( 1 - x[begin] )
    return dx
end

t0 = 0.0
tf = 2.0

# Initial condition

l = 20
k = 5.0
x0law = product_distribution(Tuple(Dirac(sin(2π * k * li / l)) for li in 0:l)...)

# The noise is a Wiener process modulated by a transport process

y0 = 0.0
θ = 200.0 # = 1 / 0.005 => time-scale = 0.005
σ = 20.0
noise = OrnsteinUhlenbeckProcess(t0, tf, y0, θ, σ)

#

ntgt = 2^14
ns = 2 .^ (4:7)
nsample = ns[[1, 2, 3, 4]]
m = 1_000

# And add some information about the simulation:

info = (
    equation = "Kanai-Tajimi model",
    noise = "Orstein-Uhlenbeck modulated by a transport process",
    ic = "\$X_0 = \\mathbf{0}\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))

# ### Order of convergence

# With all the parameters set up, we build the convergence suite:     

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

# savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
# nothing # hide

# For the sake of illustration, we plot a sample of an approximation of a target solution:
