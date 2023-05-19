# # Mechanical structural under random Earthquake-like seismic disturbances
#
# ```@meta
# Draft = false
# ```
#
# Now we consider a mechanical structure problem under ground-shaking excitations, inspired by the model in [Bogdanoff, Goldberg & Bernard (1961)](https://doi.org/10.1785/BSSA0510020293).
#
# A single-storey building is considered, with its ground floor centered at position $M_t$ and its ceiling at position $M_t + X_t$. The random process $X_t$ refers to the motion relative to the ground. The ground motion $M_t$ affects the motion of the relative displacement $X_t$ as an excitation force proportional to the ground acceleration $\ddot M_t$. The damping and elastic forces are in effect within the structure. In this framework, the equation of motion for the relative displacement $X_t$ of the ceiling of the single storey building takes the form
#
# ```math
#   \ddot X_t + 2\zeta_0\omega_0\dot X_t + \omega_0^2 X_t = - \ddot M_t.
# ```
# where $\zeta_0$ and $\omega_0$ are damping and elastic model parameters depending on the structure.
#
# For the numerical simulations, the second-order equation is written as a system of first-order equations:
#
# ```math
#   \begin{cases}
#       \dot X_t = V_t, \\
#       \dot V_t = - \omega_0^2 X_t - 2\zeta_0\omega_0 X_t - Y_t,
#   \end{cases}
# ```
#
# where $\{V_t\}_t$ is the random velocity of the celing relative to the ground and where $\{Y_t\}_t$ is the stochastic noise excitation term given as the ground acceleration, $Y_t = \ddot M_t$, generated by an Earthquake and its aftershoks, or any other ground motion.
#
# The structure is originally at rest, so we have the conditions
#
# ```math
# X_0 = 0, \quad V_0 = \dot X_0 = 0.
# ```
#
# Here, we use $\zeta_0 = 0.6$ and $\omega_0 = 15 \texttt{rad}/\texttt{s}$.
#
# Several types of noises have been considered in the literature. A typical one is a white noise. In practice, the noise is actually a colored noise and have specific spectral signatures, as in the Kanai-Tajimi model and the Clough-Penzen model.
#
# Some specific oscillation frequencies depending on the type of ground motion and the composition of the rock layers, and is modulated, with amplitude decaying. In this regard, an important model is given by [Bogdanoff, Goldberg & Bernard (1961)](https://doi.org/10.1785/BSSA0510020293), which has a linear attack rate, an exponential decay, and a specific combination of frequencies as the background noise.
#
# Moreover, in order to simulate the start of the first shock-wave and the subsequent aftershocks, the actual motion can be simulated with a combination of modulated noises at different incidence times.
#
# With this framework in mind, we model the ground motion as a transport process composed of a series of time-translations of a square-power "attack" front, with an exponentially decaying kernel and an oscillating background: $\gamma (t - \tau)^2 e^{-\delta (t - \tau)}\cos(\omega t)$, for $t \geq \tau$. The parameters $\tau, \gamma, \delta,$ and $\omega$ are random parameters
#
# The aftershocks, however, tend to come in clusters, with the ocurrence of an event increasing the chances for subsequent events. As such, self-exciting intensity processes have been successful in modeling the arrival times of the aftershocks (see e.g. [Pratiwi, Slamet, Saputro & Respatiwulan (2017)](https://doi.org/10.1088/1742-6596/855/1/012033)). The decaying kernel is usually an inverse power law, starting with the celebrated Omori formula [T. Utsu, Y. Ogata & R. S. Matsu'ura, The centenary of the Omori formula for a decay law of aftershock activity, Journal of Physics of the Earth, Volume 43 (1995), no. 1, 1-33](https://doi.org/10.4294/jpe1952.43.1)). Here, we choose to consider an exponentially decaying self-excited Hawkes process, which is easier to implement and suffices for illustrating the rate of convergence. Moreover, the intensity, or rate, of this inhomogenous Poisson point process, for the interarrival times, is not directly related to the magnitude of the aftershocks, but, again, for the sake of simplicity, we use the intensity itself as an envelope for the noise.
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

# Then we set up some variables, starting with the random seed, for reproducibility of the pseudo-random number sequence generator

rng = Xoshiro(123)

# We define the evolution law for the displacement $X_t$ driven by a noise $Y_t$. Since it is a system of equations, we use the in-place formulation. Notice the noise is a product of the background colored noise `y[1]` and the envelope noise `y[2]`. The parameters are hard-coded for simplicity.

function f!(dx, t, x, y)
    ζ₀ = 0.6
    ω₀ = 15
    dx[1] = x[2]
    dx[2] = - 2 * ζ₀ * ω₀ * x[2] - ω₀ ^ 2 * x[1] - y
    return dx
end

# The time interval is defined by the following end points 

t0, tf = 0.0, 2.0

# The structure is initially at rest, so the probability law is a vectorial product of two Dirac delta distributions, in $\mathbb{R}^2$:

x0law = product_distribution(Dirac(0.0), Dirac(0.0))

# Two types of noise are considered, both with a colored Ornstein-Uhlenbeck (OU) base noise, but modulated by either a transport process or a self-excited Hawkes process.
#
#
# As described above, we assume the ground motion is an additive combination of translated exponentially decaying wavefronts of the form
# ```math
#   m_i(t) = \gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)),
# ```
# where $(t - \tau_i)_+ = \max\{0, t - \tau_i\}$, i.e. it vanishes for $t \leq \tau_i$ and is simply $(t - \tau_i)$ for $t\geq \tau_i$. The associated noise is a combination of the second derivatives $\ddot m_i(t)$, which has jump discontinuities. Indeed, we have the ground velocities
#
# ```math
#   \begin{align*}
#   \dot m_i(t) = & 2\gamma_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
#       & -\delta_i\gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
#       & -\omega_i\gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i))
#   \end{align*}
# ```
# and the ground accelerations
#
# ```math
#   \begin{align*}
#   \ddot m_i(t) = & 2\gamma_i H(t - \tau_i) e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
#       & + \delta_i^2\gamma_i (t - \tau_i)^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
#       & -\omega_i^2\gamma_i (t - \tau_i)^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
#       & -2\delta_i\gamma_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
#       & -2\omega_i\gamma_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i)) \\
#       & +\delta_i\omega_i\gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i))
#   \end{align*}
# ```
# where $H=H(s)$ is the Heaviside function, where, for definiteness, we set $H(s) = 1,$ for $s \geq 1,$ and $H(s) = 0$, for $s < 0$.
#
#
# We implement these functions as

function gm(t::T, t0::T, γ::T, α::T, ω::T) where {T}
    tshift = max(zero(T), t - t0)
    m = γ * tshift ^2 * exp( -α * tshift ) * cos( ω * tshift )
    return m
end

function dgm(t::T, τ::T, γ::T, δ::T, ω::T) where {T}
    t₊ = max(zero(T), t - τ)
    t₊² = t₊ ^ 2
    expδt₊ = exp( -δ * t₊ )
    sinωt₊, cosωt₊ = sincos(ω * t₊)
    ṁ = γ * ( ( 2t₊ + δ * τ² ) * cosωt₊ - ω * t₊²  * sinωt₊ ) * expδt₊
    return ṁ
end

function ddgm(t::T, τ::T, γ::T, δ::T, ω::T) where {T}
    h = convert(T, t ≥ τ)
    t₊ = ( t - τ ) * h
    t₊² = t₊ ^ 2
    expδt₊ = exp( -δ * t₊ )
    sinωt₊, cosωt₊ = sincos(ω * t₊)
    m̈ = γ * ( ( 2h + ( δ^2 - ω^2 ) * t₊² - 2δ * t₊) * cosωt₊ + ( -2ω * t₊ + δ * ω * t₊² ) * sinωt₊ ) * expδt₊
    return m̈
end

#

ylaw = product_distribution(Exponential(tf/8), Uniform(0.0, 4.0), Uniform(8.0, 12.0), Uniform(8π, 32π))
nr = 12
g(t, r) = mapreduce(ri -> ddgm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(r))
noise = TransportProcess(t0, tf, ylaw, g, nr)

#

n = 2^12
tt = range(t0, tf, length=n)
yt = Vector{Float64}(undef, n)
rand!(rng, noise, yt)

# Ground motion $m_t$:

mt = [mapreduce(ri -> gm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(noise.rv)) for t in range(t0, tf, length=length(yt))]
nothing # hide

# Envelope of ground excitation

et = [mapreduce(ri -> ddgm(t, ri[1], ri[2], ri[3], 0.0), +,  eachcol(noise.rv)) for t in range(t0, tf, length=length(yt))]
nothing # hide

# Visualization

plt1 = plot(tt, mt, label="ground motion")
plt2 = plot(tt, yt, label="ground acceleration")
plt3 = plot(tt, et, label="envelope of acceleration")
plot(plt1, plt2, plt3, layout = (3, 1))

# Now we are ready to check the order of convergence. We set the target resolution, the convergence test resolutions, the sample convergence resolutions, and the number of sample trajectories for the Monte-Carlo approximation of the strong error.

ntgt = 2^18
ns = 2 .^ (6:9)
m = 500 # 1_000
nothing # hide

# We add some information about the simulation:

info = (
    equation = "mechanical structure model under ground-shaking random excitations",
    noise = "transport process noise",
    ic = "\$X_0 = \\mathbf{0}\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))

# ### Order of convergence

# With all the parameters set up, we build the convergence suites for each noise:     

suite = ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)

# Then we are ready to compute the errors:

@time result = solve(rng, suite)

# The computed strong error for each resolution in `ns` is stored in field `errors`, and raw LaTeX tables can be displayed for inclusion in the article:
# 

println(generate_error_table(result, info)) # hide
nothing # hide

# 
# The calculated order of convergence is given by field `p`:

println("Order of convergence `C Δtᵖ` with the transport-modulated noise with p = $(round(result.p, sigdigits=2))")

# 
# ### Plots
# 
# We create plots with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt = plot(result)

# and we save them for the article:

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "convergence_earthquake.png")) # hide
nothing # hide

# For the sake of illustration, we plot a sample of an approximation of a target solution, in each case:

nsample = ns[[1, 2, 3]]
plot(suite, ns=nsample)

# We also draw an animation of the motion of the single-storey building in each case. First the model with the transport-modulated noise.

# And now with the Hawkes-modulated noise.

dt = (tf - t0) / (ntgt - 1)
mt = [mapreduce(ri -> gm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(noise.rv)) for t in range(t0, tf, length=ntgt)]

@time anim = @animate for i in 1:div(ntgt, 2^9):div(ntgt, 1)
    ceiling = mt[i] + suite.xt[i, 1]
    height = 3.0
    halfwidth = 2.0
    aspectfactor = (4/6) * 4halfwidth / height
    plot([mt[i] - halfwidth; ceiling - halfwidth; ceiling + halfwidth; mt[i] + halfwidth], [0.0; height; height; 0.0], xlim = (-2halfwidth, 2halfwidth), ylim=(0.0, aspectfactor * height), xlabel="\$\\mathrm{displacement}\$", ylabel="\$\\textrm{height}\$", fill=true, title="Building at time t = $(round((i * dt), digits=3))", legend=false)
end
nothing # hide

#

gif(anim, fps = 30) # hide
