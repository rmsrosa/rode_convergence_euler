# # Random Fisher-KPP partial differential equation

# Here we simulate a Fisher-KPP equation with random boundary conditions, as inspired by the works of [Salako & Shen (2020)](https://doi.org/10.1007/s10884-020-09847-2) and [Freidlin & Wentzell (1992)](https://doi.org/10.1214/aop/1176989813)). The first work addresses the Fisher-KPP equation with a random reaction coefficient, while the second work considers more general reaction-diffusion equations but driven by random boundary conditions. Here, we consider the Fisher-KPP equation driven by random Neumann-type boundary conditions. The intent here is to illustrate the strong order 1 convergence rate on a nonlinear partial differential equation.
#
# We use the method of lines (MOL), with finite differences in space, to approximate the random partial differential equation (PDE) by a system of random ODEs.
#
# The equation is a nonlinear parabolic equation of reaction-diffusion type, modeling inhomogeneous population growth and displaying wave propagation. We force the system with random incoming/outcoming population fluctuations on the frontiers of the spatial domain.
#
# ## The equation
#
# The equation takes the form
#
# ```math
#   \frac{\partial u}{\displaystyle \partial t} = D\frac{\partial^2 u}{\partial x^2} + \lambda u(1 - \frac{u}{c_m}), \quad (t, x) \in (0, \infty) \times (0, 1),
# ```
# endowed with the boundary conditions
#
# ```math
#   \frac{\partial u}{\partial x}(t, 0) = A_t, \quad \frac{\partial u}{\partial x}(t, 1) = - B_t
# ```
#
# and a given a initial condition
# ```math
#   u(0, x) = u_0(x).
# ```
# 
# The unknown $u(t, x)$ represents the density of a given quantity at time $t$ and point $x$; $D$ is a diffusivity coefficient; $\lambda$ is a reaction, or proliferation, coefficient; and $c_m$ is a carrying capacity density coefficient.
#
# The random processes $\{A_t\}_t$ and $\{B_t\}_t$ driven the boundary fluxes are taken to be bounded processes of the form
#
# ```math
#   A_t = \delta \cos(G_t^{(1)}); \quad B_t = \delta \cos(G_t^{(2)}),
# ```
# where $\{G_t^{(i)}\}_t$, $i = 1, 2$, are independent geometric Brownian motion processes, reflecting a short time random nature and a long time oscillatory behavior.
#
# This equations allows initial conditions to evolve towards traveling wave solutions with a minimum wave speed of $2 \sqrt{\lambda D}$. We choose $\lambda = 10$ and $D = 0.009$, so the limit traveling speed is about $0.6$. The carrying capacity is set to $c_m = 1.0$.
#
# The coefficient $\delta$ for the noises are taken to be $\delta = 0.1$.
#
# The initial condition is taken to be of the form
#
# ```math
#   u_0(x) = a (x^2 - 1)^{2k},
# ```
# where $0 < a < 1$ and $k \in \mathbb{N}$, so that
#
# ```math
#   u_0(0) = a, \quad u_0(1) = 0, \quad u_0'(0) = u_0'(1) = 0.
# ```
#

using Plots # hide
plot(0.0:0.01:1.0, x -> 2x^3 - 3x^2 + 1, title="Initial condition", titlefont=8, legend=nothing) # hide

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
using BenchmarkTools

# Then we set up some variables as usual, starting with the random seed.

rng = Xoshiro(123)

# The time interval

t0, tf = 0.0, 2.0

# The discretization in space is made with `l` mesh points $x_j = (j-1) / (l-1)$, for $j = 1, \ldots, l$, corresponding to `l-1` mesh intervals of length $\Delta x = 1 / (l-1)$. The points $x_1 = 0$ and $x_l = 1$ are the boundary points. We set `l` to

l = 16

# Notice that for the target solution we need a very fine *time* mesh, on top of having to repeat the simulation a number of times for the Monte-Carlo estimate. This is computationally demanding for large `l`, so we choose a moderate number just for illustration purpose.

# The initial condition is chosen with $a = 0.8$ and $k = 6$:

u₀(x) = 0.8 * (x^2 - 1) ^ 12

# The discretized initial condition is then

u0law = product_distribution(Tuple(Dirac(u₀((j-1) / (l-1))) for j in 1:l)...)

plot(title="Discretized initial condition", titlefont=8, ylims=(0.0, 1.0), xlabel="\$x\$", ylabel="\$u\$")
plot!(0.0:0.01:1.0, u₀, label="initial condition")
scatter!((0:l-1) ./ (l-1), u₀, label="discretization")

#
# For the discretization of the equation we use finite differences with the classic second-order discretization of the second derivative:
#
# ```math
#   \frac{\partial^2 u}{\partial x^2}(t, x_j) \approx \frac{u(t, x_{j+1}) - 2u(t, x_j) + u(t, x_{j-1})}{\Delta x^2}, \quad j = 1, \ldots, l
# ```
#
# Notice this goes up to the boundary points $j=1$ and $j=l$, corresponding to $x=0$ and $x=1$, and depends on the "ghost" points $x_0 = -\Delta x$ and $x_{l+1} = 1 + \Delta x$. These points are solved for by using the Neumann boundary conditions and a centered second-order finite difference approximation of the first derivative, namely
#
# ```math
#   \frac{\partial u}{\partial x}(t, x_j) \approx \frac{u(t, x_{j+1} - u(t, x_{j-1}))}{2\Delta x},
# ```
#
# on the boundary points $j=1$ and $j=l$, so that
#
# ```math
#   u(t, x_0) = u(t, x_2) - A_t \Delta x, \qquad u(t, x_{l}) = u(t, x_{l-2}) + B_t \Delta x.
# ```
# 
# These points are plugged into the second-order approximation of the second derivatives at the boundary points.
#
# This yields the following in-place formulation for the right-hand side of the MOL Random ODE approximation of the Random PDE.

function f!(du, t, u, y)
    axes(u, 1) isa Base.OneTo || error("indexing of `x` should be Base.OneTo")

    D = 0.009
    λ = 10.0
    cₘ = 1.0
    δ = 1.0

    l = length(u)
    dx = 1.0 / (l - 1)
    dx² = dx ^ 2

    ## interior points
    for j in 2:l-1
        du[j] = D * (u[j-1] - 2u[j] + u[j+1]) / dx² + λ * u[j] * (1.0 - u[j] / cₘ)
    end

    ## ghost points
    gh1 = u[2] - 2dx * δ * cos(y[1])
    gh2 = u[l-1] - 2dx * δ * cos(y[2])

    ## boundary points
    du[1] = D * ( u[2] - 2u[1] + gh1 ) / dx² + λ * u[1] * ( 1.0 - u[1] / cₘ )
    du[l] = D * ( gh2 - 2u[l] + u[l-1] ) / dx² + λ * u[l] * ( 1.0 - u[l] / cₘ )
    return nothing
end

# Alternatively, we may use a second-order forward difference scheme for the first derivative at the left end point and a backward one at the right end point:
#
# ```math
#   \frac{\partial u}{\partial x}(t, x_j) \approx \frac{-u(t, x_{j+2} + 4u(t, x_{j+1}) - 3u(t, x_{j}))}{2\Delta x}, \quad \frac{\partial u}{\partial x}(t, x_j) \approx \frac{3u(t, x_{j} - 4u(t, x_{j-1}) + u(t, x_{j-2}))}{2\Delta x}.
# ```
#

function f_alt!(du, t, u, y)
    axes(u, 1) isa Base.OneTo || error("indexing of `x` should be Base.OneTo")

    D = 0.1
    λ = 1.0

    l = length(u)
    dx = 1.0 / (l - 1)
    dx² = dx ^ 2

    ## interior points
    for j in 2:l-1
        du[j] = D * (u[j-1] - 2u[j] + u[j+1]) / dx² + λ * u[j] * (1.0 - u[j])
    end

    ## ghost points
    gh1 = ( 4 * u[2] - u[1] ) / 3 + 2dx * y[1]
    gh2 = ( 4 * u[l] - u[l-1]) / 3 - 2dx * y[2]

    ## boundary points
    du[1] = D * ( u[2] - 2u[1] + gh1 ) / dx² + λ * u[1] * ( 1.0 - u[1] )
    du[l] = D * ( gh2 - 2u[l] + u[l-1] ) / dx² + λ * u[l] * ( 1.0 - u[l] )
    return nothing
end

# Now we make sure this is non-allocating. We use a finer spatial mesh for testing.

xx = 0.0:0.01:1.0
u = sin.(π * xx) .^ 2
du = similar(u)
du_alt = similar(u)
y = [0.0, 0.0]
t = 0.0
f!(du, t, u, y)
f_alt!(du_alt, t, u, y)

plot(xx, u, label="u")
plot!(xx, du, label="du/dt")
plot!(xx, du_alt, label="du/dt alt")

#

@btime f!($du, $t, $u, $y)
nothing # hide

# The noise terms $\{G_t^{(i)}\}_t$, $i=1, 2$, are taken to be geometric Brownian motion processes with no drift:

y0 = 1.0
μ = 0.0 # 
σ = 1.0 # 
noise = ProductProcess(GeometricBrownianMotionProcess(t0, tf, y0, μ, σ), GeometricBrownianMotionProcess(t0, tf, y0, μ, σ))

# Here is a sample path of the noise

tt = range(t0, tf, length=2^9)
yt = Matrix{Float64}(undef, 2^9, 2)
rand!(rng, noise, yt)
plot(tt, yt, label=["gBm 1" "gBm 2"], xlabel="\$t\$", ylabel="\$y\$")

# and of the cosine of the noise, which drives the boundary conditions

plot(tt, cos.(yt), label=["cos(gBm 1)" "cos(gBm 2)"], xlabel="\$t\$", ylabel="\$y\$")

# Now we set up the mesh parameters

ntgt = 2^22
ns = 2 .^ (9:11)

ntgt = 2^15 * 3^3 * 5
ns = [2^10, 2^7 * 3^2, 2^8 * 5, 2^9 * 3, 2^7 * 3 * 5, 2^11]
all(mod(ntgt, n) == 0 for n in ns)
ntgt ≥ last(ns)^2

# The number of simulations for the Monte-Carlo estimate of the rate of strong convergence
m = 1_000
m = 1

# And add some information about the simulation:

info = (
    equation = "Fisher-KPP equation",
    noise = "geometric Brownian motion noise",
    ic = "\$X_0 = 0.8(x^2 - 1)^12\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(u0law))
method = RandomEuler(length(u0law))

# ### Order of convergence

# With all the parameters set up, we build the convergence suite:     

suite = ConvergenceSuite(t0, tf, u0law, f!, noise, target, method, ntgt, ns, m)

# Then we are ready to compute the errors:

@time result = solve(rng, suite)
nothing # hide

# The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:
# 

table = generate_error_table(result, info)

println(table) # hide
nothing # hide

# 
# 
# The calculated order of convergence is given by `result.p`:

println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")
nothing # hide

# 
# 
# ### Plots
# 
# We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt = plot(result)

# and save it for inclusion in the article.

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/",  "order_fisherkpp.png")) # hide
nothing # hide

# For the sake of illustration, we plot the evolution of a sample target solution:

dt = (tf - t0) / (ntgt - 1)

anim = @animate for i in 1:div(ntgt, 2^10):div(ntgt, 2^2)
    plot(range(0.0, 1.0, length=l), view(suite.xt, i, :), ylim=(0.0, 1.0), xlabel="\$x\$", ylabel="\$u\$", fill=true, title="population density at time t = $(round((i * dt), digits=3))", legend=false)
end

gif(anim, joinpath(@__DIR__() * "../../../../latex/img/","fisherkpp.gif"), fps = 30) # hide
