# # Seismic excitations of mechanical structures with the Kanai-Takimi model

# Now we consider a mechanical structure problem under ground-shaking excitations, based on the Kanai-Tajimi model.
#
# We consider a single-storey structure, with a random horizontal displacement $X_t$ relative to the ground motion displacement $Y_t$, which follows the equation
# ```math
#   \ddot X_t + 2\zeta\omega\dot X_t + \omega^2 X_t = Y_t.
# ```
# where $\zeta$ and $\omega$ are damping and elastic model parameters depending on the structure.
#
# For the numerical simulations, the second-order equation is written as a system of first-order equations:
#
# ```math
#   \begin{cases}
#       \dot X_t = V_t, \\
#       \dot V_t = - \omega^2 X_t - 2\zeta\omega X_t + Y_t,
#   \end{cases}
# ```
#
# where $\{V_t\}_t$ is the random velocity relative to the ground.
#
# The term $Y_t$ is a stochastic noise representing the ground motion caused by the Earthquake and its aftershocks.
#
# The structure is originally at rest, before the first groundshaking excitations, so we have the conditions
#
# ```math
# X_t(0) = 0, \quad \dot X_t(0) = 0.
# ```
#
# When affected by the ground motion, the one-storey building has its ground at position $Y_t$ and its ceiling at position $Y_t + X_t$. The random process $X_t$ refers to the motion relative to the ground, and for which the damping and elastic forces are in effect.
#
# Here, as in [Neckel & Rupp (2013)](https://doi.org/10.2478/9788376560267), based on [Housner & Jenning (1964)](https://doi.org/10.1061/JMCEA3.0000448), we use $\zeta = 0.64$ and $\omega = 15.56 \texttt{rad}/\texttt{s}$.

# Several types of noises have been considered in the literature. A typical one is a white noise. In practice, the noise is actually a colored noise, whose amplitude decays in time, followed by smaller peaks in amplitude due to aftershocks.
#
# With this in mind, we model the background colored noise with an Orsntein-Uhlenbeck (OU) process $\{O_t\}_t$ with a relatively small time scale $\tau$. More precisely, we assume $\{O_t\}_t$ satisfies the stochastic differential equation
#
# ```math
#   \tau \mathrm{d}O_t = - \mathrm{d}t + \sigma \mathrm{d}W_t,
# ```
# where $\{W_t\}_t$ is a standard Wiener process. This leads to an Orsntein-Uhlenbeck process with drift $1/\tau$ and diffusion $\sigma/\tau$. This process has mean, variance, and covariance given by
#
# ```math
#   \mathbb{E}[O_t] = O_0 e^{-\frac{\displaystyle t}{\displaystyle\tau}}, \quad \mathrm{Var}(O_t) = \frac{\sigma^2}{2\tau}, \quad \mathrm{Cov}(O_t,O_s) = \frac{\sigma^2}{2\tau} e^{-\frac{\displaystyle |t - s|}{\displaystyle \tau}}.
# ```
#
# Hence, $O_t$ and $O_s$ are significantly correlated only when $|t - s| \lesssim \tau$. When $\tau \rightarrow 0$ with $\sigma^2/2\tau \rightarrow 1$, this process approximates a Gaussian white noise.
#
# Moreover, in order to simulate the start of the first shock-wave and the subsequent aftershocks, we modulate the Ornstein-Uhlenbeck process with either a transport process or a self-excited point process.
#
# The transport process is composed of a series of time-translations of a Hölder-continuous "attack" front, followed by an exponentially decaying kernel: $\gamma (t - \delta)^\alpha e^{-\beta (t - \delta)}$, for $t \geq \delta$. The parameters $\alpha, \beta, \gamma, \delta$ are random parameters, with the Hölder exponent $\alpha$ being arbitrarily small. This is based on the model by [Bogdanoff, Goldberg & Bernard (1961)](https://doi.org/10.1785/BSSA0510020293), which has a linear attack rate, an exponential decay, and a specific combination of frequencies as the background noise.
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
using FFTW
using Distributions
using RODEConvergence

# Then we set up some variables, starting with the random seed, for reproducibility of the pseudo-random number sequence generator

rng = Xoshiro(123)

# We define the evolution law for the displacement $X_t$ driven by a noise $Y_t$. Since it is a system of equations, we use the in-place formulation. Notice the noise is a product of the background colored noise `y[1]` and the envelope noise `y[2]`. The parameters are hard-coded for simplicity.

function f!(dx, t, x, y)
    ζ = 0.64
    ω = 15.56
    ζ = 0.064
    ω = 15.56
    dx[1] = x[2]
    dx[2] = - 2 * ζ * ω * x[2] - ω ^ 2 * x[1] + y[1] * y[2]
    return dx
end

# The time interval is defined by the following end points 

t0, tf = 0.0, 4.0

# The structure is initially at rest, so the probability law is a vectorial product of two Dirac delta distributions, in $\mathbb{R}^2$:

x0law = product_distribution(Dirac(0.0), Dirac(0.0))

# Two types of noise are considered, both with a colored Ornstein-Uhlenbeck (OU) base noise, but modulated by either a transport process or a self-excited Hawkes process.
#
# We define first the OU process

y0 = 0.0
τ = 0.005 # time scale
σ = 0.1 # diffusion
colored_noise = OrnsteinUhlenbeckProcess(t0, tf, y0, 1/τ, σ/τ)

# Just for the sake of comparison, we also define a Wiener process, in order to compare the colored noise with a white noise, obtained by finite differences on the Wiener process:

wiener_noise = WienerProcess(t0, tf, y0)

# The transport process is defined as follows

ylaw = product_distribution(Uniform(0.0, 2.0), Uniform(0.0, 0.5), Uniform(2.0, 8.0), Exponential())
nr = 8
g(t, r) = mapreduce(ri -> ri[1] * max(0.0, t - ri[4]) ^ ri[2] * exp(-ri[3] * max(0.0, t - ri[4])), +, eachcol(r))
transport_envelope_noise = TransportProcess(t0, tf, ylaw, g, nr)

# Finally, we define the exponentially decaying Hawkes process

λ₀ = 3.0
a = 0.3
δ = 5.0
β = 1.8
θ = 1/β
dylaw = Exponential(θ)

hawkes_envelope_noise = ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw)

# With those, we define the transport-modulated colored noise

transportmodulated_noise = ProductProcess(colored_noise, transport_envelope_noise)

# and the Hawkes-modulated colored noise

hawkesmodulated_noise = ProductProcess(colored_noise, hawkes_envelope_noise)

# Let us visualize a sample path of these process. We define the resolution, pre-allocate some vectors, and compute the sample paths.

ntgt = 2^12
yt1 = Vector{Float64}(undef, ntgt)
yt2 = Vector{Float64}(undef, ntgt)
yt3 = Vector{Float64}(undef, ntgt)
yt4 = Vector{Float64}(undef, ntgt)

rand!(rng, wiener_noise, yt4)
rand!(rng, colored_noise, yt1)
rand!(rng, transport_envelope_noise, yt2)
rand!(rng, hawkes_envelope_noise, yt3)

# The white noise is obtained, approximately, as a finite difference of the Wiener process, taking into account that $\mathrm{d}W_t \sim \sqrt{\mathrm{d}t}

dt = (tf - t0) / (length(yt1) - 1)
wt = (yt4[2:end] .- yt4[1:end-1])/dt^0.5
nothing

# Now we plot the Wiener process, the associated white noise, and the Ornstein-Uhlenbeck colored noise

begin
    plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(t0+dt:dt:tf, wt, label="white noise")
    plot!(t0:dt:tf, yt1, label="OU")
    plot!(t0:dt:tf, yt4, label="Wiener")
    savefig(joinpath(@__DIR__() * "../../../../latex/img/", "earthquake_ou_vs_whitenoise.png"))
end

# We can also check their spectrum, using [JuliaMath/FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/).

begin
    plot(abs2.(rfft(wt)), label="white noise spectrum")
    plot!(abs2.(rfft(yt1)), label="OU spectrum")
end

# For the sake of comparison, let us check their mean and variance

println("Mean of the white noise: $(mean(wt))")

println("Mean of the colored OU process: $(mean(yt1))")

println("Variance of the white noise: $(var(wt))")

println("Variance of the colored OU process: $(var(yt1))")

nothing # hide

# Finally, we plot the modulated noises.

begin
    plt1 = plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(plt1, t0:dt:tf, yt2 .* yt1, label="noise")
    plot!(plt1, t0:dt:tf, yt2, label="transport envelope")
    
    plt2 = plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(plt2, t0:dt:tf, yt3 .* yt1, label="noise")
    plot!(plt2, t0:dt:tf, yt3, label="Hawkes envelope")

    plot(plt1, plt2)
end

# Now we are ready to check the order of convergence. We set the target resolution, the convergence test resolutions, the sample convergence resolutions, and the number of sample trajectories for the Monte-Carlo approximation of the strong error.

ntgt = 2^18
ns = 2 .^ (6:9)
m = 1 # 1_000
nothing # hide

# We add some information about the simulation:

info_ou = (
    equation = "Kanai-Tajimi Earthquake model",
    noise = "Orstein-Uhlenbeck colored noise modulated by a transport process",
    ic = "\$X_0 = \\mathbf{0}\$"
)

info_hawkes = (
    equation = "Kanai-Tajimi Earthquake model",
    noise = "Orstein-Uhlenbeck colored noise modulated by an expontentially-decaying Hawkes process",
    ic = "\$X_0 = \\mathbf{0}\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))

# ### Order of convergence

# With all the parameters set up, we build the convergence suites for each noise:     

transportmodulated_suite = ConvergenceSuite(t0, tf, x0law, f!, transportmodulated_noise, target, method, ntgt, ns, m)

hawkesmodulated_suite = ConvergenceSuite(t0, tf, x0law, f!, hawkesmodulated_noise, target, method, ntgt, ns, m)

# Then we are ready to compute the errors:

@time transportmodulated_result = solve(rng, transportmodulated_suite)

#

@time hawkesmodulated_result = solve(rng, hawkesmodulated_suite)

# The computed strong error for each resolution in `ns` is stored in field `errors`, and raw LaTeX tables can be displayed for inclusion in the article:
# 

println(generate_error_table(transportmodulated_result, info_ou)) # hide
nothing # hide

#

println(generate_error_table(hawkesmodulated_result, info_hawkes)) # hide
nothing # hide

# 
# The calculated order of convergence is given by field `p`:

println("Order of convergence `C Δtᵖ` with the transport-modulated noise with p = $(round(transportmodulated_result.p, sigdigits=2))")

#

println("Order of convergence `C Δtᵖ` with the Hawkes-modulated nooise with p = $(round(hawkesmodulated_result.p, sigdigits=2))")

# 
# ### Plots
# 
# We create plots with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

plt_tr = plot(transportmodulated_result)
plt_hk = plot(hawkesmodulated_result)

plt = plot(plt_tr, plt_hk)

# and we save them for the article:

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "convergence_kanaitajimi.png")) # hide
nothing # hide

# For the sake of illustration, we plot a sample of an approximation of a target solution, in each case:

nsample = ns[[1, 2, 3]]
plot(transportmodulated_suite, ns=nsample)

#

plot(hawkesmodulated_suite, ns=nsample)

# The displacement of the building is tiny compared to the ground, in this simulation

plot(transportmodulated_suite, ns=nothing)
plot!(transportmodulated_suite, xshow = false, yshow = prod, ns=nothing)

# 

plot(hawkesmodulated_suite, ns=nothing)
plot!(hawkesmodulated_suite, xshow = false, yshow = prod, ns=nothing)

# We also draw an animation of the motion of the single-storey building in each case. First the model with the transport-modulated noise.

# And now with the Hawkes-modulated noise.

dt = (tf - t0) / (ntgt - 1)
ground = accumulate(+, accumulate(+, map(prod, eachrow(transportmodulated_suite.yt)))) * dt ^ 2

@time anim = @animate for i in 1:div(ntgt, 2^10):div(ntgt, 1)
    ceiling = ground[i] + transportmodulated_suite.xt[i, 1]
    heigth = 3.0
    halfwidth = 0.1
    plot([ground[i] - halfwidth; ceiling - halfwidth; ceiling + halfwidth; ground[i] + halfwidth], [0.0; heigth; heigth; 0.0], xlim = (-0.2, 0.2), ylim=(0.0, 4.0), xlabel="\$\\mathrm{displacement}\$", ylabel="\$\\textrm{height}\$", fill=true, title="Building at time t = $(round((i * dt), digits=3))", legend=false)
end
nothing # hide

#

gif(anim, fps = 30) # hide

# Now just the relative motion

@time anim = @animate for i in 1:div(ntgt, 2^12):div(ntgt, 2^4)
    suite = transportmodulated_suite
    ground = 0.0
    ceiling = ground + suite.xt[i, 1]
    heigth = 0.02
    halfwidth = 0.001
    plot([ground - halfwidth; ceiling - halfwidth; ceiling + halfwidth; ground + halfwidth], [0.0; heigth; heigth; 0.0], xlim = (-0.004, 0.004), ylim=(0.0, 0.1), xlabel="\$\\mathrm{displacement}\$", ylabel="\$\\textrm{height}\$", fill=true, title="Building at time t = $(round((i * dt), digits=3))", legend=false)
end

nothing # hide

#

gif(anim, fps = 30) # hide

#

tt = range(t0, tf, length=length(yt))
rand!(rng, noise, yt)
at = yt[:, 2] .* sin.(4π * tt)
plot(at)
bt = map(prod, eachrow(yt))
nt = at .+ 0.1 .* yt[:, 1]
plot(nt)

dgrd = accumulate(+, nt) * dt
grd = accumulate(+, dgrd) * dt

plot(dgrd)