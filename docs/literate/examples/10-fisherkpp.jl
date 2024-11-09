# # Random Fisher-KPP partial differential equation
# 
# Here we simulate a Fisher-KPP equation with random boundary conditions, as inspired by the works of [Salako & Shen (2020)](https://doi.org/10.1007/s10884-020-09847-2) and [Freidlin & Wentzell (1992)](https://doi.org/10.1214/aop/1176989813). The first work addresses the Fisher-KPP equation with a random reaction coefficient, while the second work considers more general reaction-diffusion equations but driven by random boundary conditions. The deterministic Fisher-KPP equations has its origins in [Fisher (1937)](https://doi.org/10.1111/j.1469-1809.1937.tb02153.x) and [Kolmogorov, Petrovskii & Piscunov (1937)](https://www.bibsonomy.org/bibtex/23cfaf2cd2a49db658463fc5b115b3aa4/peter.ralph)

# We consider the Fisher-KPP equation driven by Neumann boundary conditions, with a random influx on the left end point and no flux on the right end point. The intent here is to illustrate the strong order 1 convergence rate on a nonlinear partial differential equation.
#
# We use the method of lines (MOL), with finite differences in space, to approximate the random partial differential equation (PDE) by a system of random ODEs.
#
# The equation is a nonlinear parabolic equation of reaction-diffusion type, modeling inhomogeneous population growth displaying wave propagation, and many other phenomena such as combustion front wave propagation, physiollogy and crystallography pattern formation, and so on. We force the system with a random incoming population on one of the boundaries of the spatial domain.
#
# ## The equation
#
# The equation takes the form
#
# ```math
#   \frac{\partial u}{\displaystyle \partial t} = \mu\frac{\partial^2 u}{\partial x^2} + \lambda u\left(1 - \frac{u}{u_m}\right), \quad (t, x) \in (0, \infty) \times (0, 1),
# ```
# endowed with the boundary conditions
#
# ```math
#   \frac{\partial u}{\partial x}(t, 0) = - Y_t, \quad \frac{\partial u}{\partial x}(t, 1) = 0,
# ```
#
# and a given a initial condition
# ```math
#   u(0, x) = u_0(x).
# ```
# 
# The unknown $u(t, x)$ represents the density of a given quantity at time $t$ and point $x$; $D$ is a diffusivity coefficient; $\lambda$ is a reaction, or proliferation, coefficient; and $u_m$ is a carrying capacity density coefficient.
#
# The random process $\{Y_t\}_t,$ which drives the flux on the left boundary point, is taken to be a colored noise modulated by a exponentially decaying Hawkes process, representing random trains of incoming population.
#
# This equation displays traveling wave solutions with a minimum wave speed of $2 \sqrt{\lambda \mu}$. We choose $\lambda = 10$ and $\mu= 0.009$, so the limit traveling speed is about $0.6$. The carrying capacity is set to $u_m = 1.0$.
#
# The initial condition is taken to be zero, $u_0(x) = 0$, so all the population originates from the left boundary influx.
#
# The mass within the region $0\leq x \leq 1$ satisfies
#
# ```math
#   \frac{\mathrm{d}}{\mathrm{d} t} \int_0^1 u(t, x) \;\mathrm{d}x = \mu\int_0^1 u_{xx}(t, x) \;\mathrm{d}x + \lambda \int_0^1 u(t, x)\left(1 - \frac{u(t, x)}{u_m}\right)\;\mathrm{d}x.
# ```
#
# Using the boundary conditions, we find that
# ```math
#   \frac{\mathrm{d}}{\mathrm{d} t} \int_0^1 u(t, x) \;\mathrm{d}x = \mu Y_t  + \frac{\lambda}{u_m} \int_0^1 u(t, x)\left(u_m - u(t, x)\right)\;\mathrm{d}x,
# ```
# which is nonnegative, provided $0 \leq u \leq u_m$ and $Y_t \geq 0$.
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages:

using Plots
using Measures
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
using BenchmarkTools

# Then we set up some variables as usual, starting with the random seed:

rng = Xoshiro(123)
nothing # hide

# The time interval:

t0, tf = 0.0, 2.0
nothing # hide

# The discretization in space is made with `l+1` mesh points $x_j = j / l$, for $j = 0, \ldots, l$, corresponding to `l` mesh intervals of length $\Delta x = 1 / l$. The points $x_0 = 0$ and $x_l = 1$ are the boundary points. For illustration purposes, we start by setting `l` to

l = 64
nothing # hide

# Notice that for the target solution we need a very fine *time* mesh, on top of having to repeat the simulation a number of times for the Monte-Carlo estimate. This is computationally demanding for large `l`, so we choose a moderate number just for illustration purpose.

# The initial mass is zero:

u₀(x) = 0.0
nothing # hide

# The discretized initial condition is then

u0law = product_distribution(Tuple(Dirac(u₀(j / l)) for j in 0:l)...)

#
# For the discretization of the equation we use finite differences with the classic second-order discretization of the second derivative:
#
# ```math
#   \frac{\partial^2 u}{\partial x^2}(t, x_j) \approx \frac{u(t, x_{j+1}) - 2u(t, x_j) + u(t, x_{j-1})}{\Delta x^2}, \quad j = 1, \ldots, l
# ```
#
# Notice this goes up to the boundary points $j=0$ and $j=l$, corresponding to $x=0$ and $x=1$, and depends on the "ghost" points $x_{-1} = -\Delta x$ and $x_{l+1} = 1 + \Delta x$. These points are solved for by using the Neumann boundary conditions and a centered second-order finite difference approximation of the first derivative, namely
#
# ```math
#   \frac{\partial u}{\partial x}(t, x_j) \approx \frac{u(t, x_{j+1} - u(t, x_{j-1}))}{2\Delta x},
# ```
#
# on the boundary points $j=0$ and $j=l$, so that
#
# ```math
#   u(t, x_{-1}) = u(t, x_1) + 2Y_t \Delta x, \qquad u(t, x_{l+1}) = u(t, x_{l-1}).
# ```
# 
# These points are plugged into the second-order approximation of the second derivatives at the boundary points.
#
# This yields the following in-place formulation for the right-hand side of the MOL Random ODE approximation of the Random PDE, keeping in mind that julia is 1-based, so the `j` indices are shifted up by one.

μ = 0.009
λ = 10.0
uₘ = 1.0

params = (μ, λ, uₘ)

function f!(du, t, u, y, p) # ; μ=μ, λ=λ, uₘ=uₘ)
    axes(u, 1) isa Base.OneTo || error("indexing of `x` should be Base.OneTo")

    μ = p[1]
    λ = p[2]
    uₘ = p[3]

    l = length(u) - 1
    dx = 1.0 / l
    dx² = dx ^ 2

    ## interior points
    for j in 2:l
        du[j] = μ * (u[j-1] - 2u[j] + u[j+1]) / dx² + λ * u[j] * (1.0 - u[j] / uₘ)
    end

    ## ghost points
    gh01 = u[2] + 2dx * max(0.0, y[1] * y[2])
    ghl1 = u[l]

    ## boundary points
    du[1] = μ * ( u[2] - 2u[1] + gh01 ) / dx² + λ * u[1] * ( 1.0 - u[1] / uₘ )
    du[l+1] = μ * ( ghl1 - 2u[l+1] + u[l] ) / dx² + λ * u[l+1] * ( 1.0 - u[l+1] / uₘ )
    return nothing
end
nothing # hide

# We set the parameters for the equation

μ = 0.009
λ = 10.0
uₘ = 1.0

# Now we make sure this is non-allocating. We use a finer spatial mesh for testing.

xx = 0.0:0.01:1.0
u = sin.(π * xx) .^ 2
du = similar(u)
y = [0.0, 0.0]
t = 0.0
f!(du, t, u, y, params)
nothing # hide

# Visualize the results

plot(xx, u, label="u")
plot!(xx, du, label="du/dt")

# and check performace

@btime f!($du, $t, $u, $y, $params)
nothing # hide

# The noise is a colored Ornstein-Uhlenbeck noise truncated to non-negative values and modulated by a Hawkes process, which is implemented as two separate noises, which are combined in `f!`.

# The Ornstein-Uhlenbeck is defined as follows

y0 = 0.0
τ = 0.005 # time scale
ς = √τ # large-scale diffusion
ν = 1/τ # drift
σ = ς/τ # diffusion
colored_noise = OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ)

# And the exponentially-decaying Hawkes process is defined by

λ₀ = 5.0
a = 1.4
δ = 8.0
β = 0.4
θ = 1/β
dylaw = Exponential(θ)

hawkes_envelope_noise = ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw)

# The are combined into the following [`ProductProcess`](@ref)

noise = ProductProcess(colored_noise, hawkes_envelope_noise)

# Here is a sample path of the two noises:

tt = range(t0, tf, length=2^9+1)
yt = Matrix{Float64}(undef, 2^9+1, 2)
rand!(rng, noise, yt)

#

plt_OUandHawkes = plot(tt, yt, label=["OU" "Hawkes"], xlabel="\$t\$", ylabel="\$y\$")

# and the modulated and truncated colored noise:

plt_noise = plot(tt, map(y -> max(0.0, y[1] * y[2]), eachrow(yt)), label="noise", xlabel="\$t\$", ylabel="\$y\$")

# We also make sure drawing a noise sample path does not allocate. We use a different, and disposable, random seed, not to mess up with the reproducibility, since benchmarks can have different evaluations and samples, depending on the system itself and on the system environment:

@btime rand!($Xoshiro(), $noise, $yt)
nothing # hide

# Now that we are done with testing, we set up the mesh parameters for the convergence. For stability reasons, we let $\Delta t \sim \Delta x^2$ and make sure that $2\mu \Delta t/\Delta x^2 \leq 1.$ This condition follows from the Von Neumann stability analysis, by checking for discrete solution $E_{j,k} = A e^{\alpha k\tau  - i \beta j h}$ of the error, where $\tau = \Delta t$, $h = \Delta x$, and requiring that the amplification factor at each time step is bounded by $1 + \mathcal{O}(\tau).$

l = 512 # 2^9
u0law = product_distribution(Tuple(Dirac(u₀(j / l)) for j in 0:l)...)
ntgt = 2^18
ns = [2^5, 2^7, 2^9]

#

ks = [2^6, 2^5, 2^4]

#

@info  l ./ ks # 2^9 ./ ks = [2^3 2^4 2^5] = [8 16 32]
#nothing # hide

# and make sure they meet all the requirements:

all(mod(ntgt, n) == 0 for n in ns) && ntgt ≥ last(ns)^2

# The number of simulations for the Monte-Carlo estimate of the rate of strong convergence

m = 40
nothing # hide

# We then add some information about the simulation, for the caption of the convergence figure.

info = (
    equation = "the Fisher-KPP equation",
    noise = "Hawkes-modulated Ornstein-Uhlenbeck colored noise",
    ic = "\$X_0 = 0\$"
)
nothing # hide

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(u0law))
method = RandomEuler(length(u0law))

# ### Order of convergence

# With all the parameters set up, we build the convergence suite:     

suite = ConvergenceSuite(t0, tf, u0law, f!, noise, params, target, method, ntgt, ns, m, ks)

# Then we are ready to compute the errors:

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
# ### Plots
# 
# We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt_result = plot(result)

#

savefig(plt_result, joinpath(@__DIR__() * "../../../../latex/img/",  "order_fisherkpp.pdf")) # hide
nothing # hide

# We also combine some plots into a single figure, for the article:

plt_combined = plot(plt_result, plt_OUandHawkes, plt_noise, layout=@layout([ a [ b; c ] ]), size=(800, 280), title=["(a) Fisher-KPP model" "(b) noise parts" "(c) noise sample path"], legendfont=7, titlefont=10, bottom_margin=5mm, left_margin=5mm)

#

savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "fisherkpp_combined.pdf")) # hide
nothing # hide


# For the sake of illustration, we plot the evolution of a sample target solution:

dt = ( tf - t0 ) / ( ntgt - 1 )

@time anim = @animate for i in 1:div(ntgt, 2^7):div(ntgt, 1)
    plot(range(0.0, 1.0, length=l+1), view(suite.xt, i, :), ylim=(0.0, 1.1), xlabel="\$x\$", ylabel="\$u\$", fill=true, title="population density at time t = $(round((i * dt), digits=3))", titlefont=10, legend=false)
end

nothing # hide

#

gif(anim, fps = 30) # hide
