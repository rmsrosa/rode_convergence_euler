# # Earthquake model

# Now we consider a mechanical structure problem under ground-shaking excitations, based on Earthquake models, especially the Kanai-Tajimi model.
#
# The mechanical structure is forced by a stochastic noise modeling the effects of an Earthquake. Several types of noises have been considered in the literature. A typical one is a white noise. Further studies show that the noise is actually a colored noise. Here, we model the colored noise with a Orsntein-Uhlenbeck (OU) process $\{O_t\}_t$ with a relative small time scale $\tau$, i.e. satisfying the SDE
#
# ```math
#   \tau \mathrm{d}O_t = - \mathrm{d}t + D \mathrm{d}W_t,
# ```
# where $\{W_t\}_t$ is a standard Wiener process. This leads to an Orsntein-Uhlenbeck process with drift $\nu = 1/\tau$ and diffusion $\sigma = D/\tau$. This process has mean, variance, and covariance given by
#
# ```math
#   \mathbb{E}[O_t] = O_0 e^{-\frac{t}{\tau}}, \mathrm{Var}(O_t) = \frac{D^2}{2\tau}, \quad \mathrm{Cov}(O_t,O_s) = \frac{D^2}{2\tau} e^{-\frac{|t - s|}{\tau}}.
# ```
#
# Hence, $O_t$ and $O_s$ are significantly correlated only when $|t - s| \lessim \tau$. When $\tau \rightarrow 0$ with $D^2/2\tau \rightarrow 1$, this approximates a Gaussian white noise.
#
# Moreover, in order to simulate the start of the first shock-wave and the subsequent aftershocks, we modulate the OU process with either a transport process or a self-excited point process.
#
# The transport process is composed of a series of time-translations of a Hölder-continuous attack front, followed by an exponential decay: $\gamma (t - \delta)^\alpha e^{-\beta (t - \delta)}$, for $t \geq \delta$. The parameters $\alpha, \beta, \gamma, \delta$ are random parameters, with the Hölder exponent $\alpha$ being arbitrarily small.
#
# The self-excited point process is an exponentially decaying Hawkes process, commonly used in modeling Earthquake shots and aftershocks.
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
using FFTW
using Distributions
using RODEConvergence

# Then we set up some variables

rng = Xoshiro(123)

function f(dx, t, x, y)
    ζ = 0.64
    ω = 15.56
    dx[1] = -x[2] + y[1]
    dx[2] = -2 * ζ * ω * (x[2] + y[1]) + ω ^ 2 * x[1] + y[1] * y[2]
    return dx
end

t0 = 0.0
tf = 2.0

# The structure initially at rest

x0law = product_distribution(Dirac(0.0), Dirac(0.0))

# The noise is a Wiener process modulated by a transport process

y0 = 0.0
τ = 0.005 # time scale
D = 0.1 # large-scale diffusion
ν = 1/τ # drift
σ = D/τ # diffusion
colored_noise = OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ)

ylaw = product_distribution(Uniform(0.0, 2.0), Uniform(0.0, 0.5), Uniform(2.0, 8.0), Exponential())
nr = 5
g(t, r) = mapreduce(ri -> ri[1] * max(0.0, t - ri[4]) ^ ri[2] * exp(-ri[3] * max(0.0, t - ri[4])), +, eachcol(r))
transport_envelope_noise = TransportProcess(t0, tf, ylaw, g, nr)

λ₀ = 2.0
a = 0.8
δ = 0.9
β = 1.8
θ = 1/β
dylaw = Exponential(θ)

hawkes_envelope_noise = ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw)

transportmodulated_noise = ProductProcess(colored_noise, transport_envelope_noise)

hawkesmodulated_noise = ProductProcess(colored_noise, hawkes_envelope_noise)

#

ntgt = 2^12
yt1 = Vector{Float64}(undef, ntgt)
yt2 = Vector{Float64}(undef, ntgt)
yt3 = Vector{Float64}(undef, ntgt)
yt4 = Vector{Float64}(undef, ntgt)

rand!(rng, colored_noise, yt1)
rand!(rng, transport_envelope_noise, yt2)
rand!(rng, hawkes_envelope_noise, yt3)

wiener_noise = WienerProcess(t0, tf, y0)
rand!(rng, wiener_noise, yt4)
dt = (tf - t0) / (length(yt1) - 1)
wt = (yt4[2:end] .- yt4[1:end-1])/dt^0.5

begin
    plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(t0+dt:dt:tf, wt, label="white noise")
    plot!(t0:dt:tf, yt1, label="OU")
    plot!(t0:dt:tf, yt4, label="Wiener")
end

#

begin
    plot(abs2.(rfft(wt)), label="white noise spectrum")
    plot!(abs2.(rfft(yt1)), label="OU spectrum")
end

#

mean(wt)

#

mean(yt1)

#

std(wt)

#

std(yt1)

#

begin
    plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(t0:dt:tf, yt2 .* yt1, label="noise")
    plot!(t0:dt:tf, yt2, label="envelope")
end

#

begin
    plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(t0:dt:tf, yt3 .* yt1, label="noise")
    plot!(t0:dt:tf, yt3, label="envelope")
end

#

ntgt = 2^18
ns = 2 .^ (5:9)
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
