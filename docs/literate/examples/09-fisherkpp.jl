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

    l = length(x)
    ds² = 1.0 / (l - 1)^2
    for j in 2:l-1
        dx[j] = (x[j-1] - 2x[j] + x[j+1]) / ds² + max(0.0, y) * x[j] * (1.0 - x[j])
    end

    dx[1] = (4x[2] - x[3] ) / 3 # Neumann no-flux
    dx[l] = (4x[l-1] - x[l-2]) / 3 # Neumann no-flux
    #dx[1] = ( x[l] - 2x[1] + x[2]) / ds² .+ y .* x[1] * ( 1 - x[1] )
    #dx[l] = ( x[l-1] - 2x[l] + x[1]) / ds² .+ y .* x[l] * ( 1 - x[l] )
    return nothing
end

t0 = 0.0
tf = 2.0

# Initial condition

l = 16
x0law = product_distribution(Tuple(Dirac(sin(2π * li / l)^2) for li in 0:l)...)

# Teste 

function evolve(x0law, t0, tf)
    x = rand(rng, x0law)
    l = length(x)
    ds² = 1 / (l - 1)^2
    dt = ds² / 2
    dx = similar(x)
    y = 1.0
    t = t0
    while t < tf
        f!(dx, t, x, y)
        x .+= dt .* dx
        t += dt
        display(plot(range(0.0, 1.0, length=l), x))
    end
    return x
end

# x = evolve(x0law, t0, tf/100)

# plot(range(0.0, 1.0, length=l), x)

# The noise is a Wiener process modulated by a transport process

y0 = 0.0
ν = 200.0 # = 1 / 0.005 => time-scale = 0.005
σ = 10.0 # variance σ^2 / 2ν = 0.25
noise = OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ)

#

ntgt = 2^22
ns = 2 .^ (9:11)

ntgt = 2^15 * 3^3 * 5
ns = [2^10, 2^7 * 3^2, 2^8 * 5, 2^9 * 3, 2^7 * 3 * 5, 2^11]
all(mod(ntgt, n) == 0 for n in ns)
ntgt ≥ last(ns)^2
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
nothing

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
