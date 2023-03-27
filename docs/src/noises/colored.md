# Ornstein-Uhlenbeck colored noise approximation of white noise

White noise, as modeled by the "derivative" of the Wiener process, in a suitable distributional sense, is an ubiquotous noise in modelling stochastic behavior in evolutionary systems.

In many situations, however, the actual noise is a colored noise, sometimes with a characteristic amplitude decay, giving a certain hue to the noise, or some other amplitude form.

Here, we explore the approximation of a white noise by an Orstein-Uhlenbeck (OU) process ``\{O_t\}_t``. This is achieved by controlling a *time-scale* ``\tau`` for the process. More precisely, we assume ``\{O_t\}_t`` satisfies the stochastic differential equation (SDE)

```math
    \tau \mathrm{d}O_t = - \mathrm{d}t + \varsigma \mathrm{d}W_t,
```
where ``\{W_t\}_t`` is a standard Wiener process. This leads to an Orsntein-Uhlenbeck process with drift ``\nu = 1/\tau`` and diffusion ``\sigma = \varsigma/\tau``. This process has mean, variance, and covariance given by

```math
   \mathbb{E}[O_t] = O_0 e^{-\frac{\displaystyle t}{\displaystyle\tau}}, \quad \mathrm{Var}(O_t) = \frac{\varsigma^2}{2\tau}, \quad \mathrm{Cov}(O_t,O_s) = \frac{\varsigma^2}{2\tau} e^{-\frac{\displaystyle |t - s|}{\displaystyle \tau}}.
```

Hence, ``O_t`` and ``O_s`` are significantly correlated only within the time scale ``|t - s| \lesssim \tau``.

Moreover, when
```math
\tau \rightarrow 0, \quad \textrm{with} \quad \frac{\varsigma^2}{2\tau} \rightarrow 1,
```
this process approximates a Gaussian white noise. This is equivalent to 
```math
\nu \rightarrow \infty, \quad \frac{\sigma^2}{2\nu} \rightarrow 1,
```
in the usual drift and diffusion Ornstein-Uhlenbeck parameters.

Below, we illustrate this approximation with a few numerical simulations.

## Loading the packages

We start by loading the necessary packages:

```@example colored
using Random
using Distributions
using Statistics
using BenchmarkTools
using FFTW
using LinearAlgebra
using Plots
using RODEConvergence
```

## Setup

We fix the initial condition `y0` and the initial and final times `t0` and `tf` of the time interval of the desired simulations. We also fix the size `n` of the sample paths. The sample paths will be generated for a set of `n` times uniformly distributed within the time interval from `t0` to `tf`, which yields a time mesh which we define as `tt`:

```@example colored
y0 = 0.0
t0 = 0.0
tf = 2.0
n = 2^10
tt = range(t0, tf, length=n)
dt = (tf - t0) / (n - 1)
nothing # hide
```

We also choose a few set of parameters for the Ornstein-Uhlenbeck process, to illustrate the convergence process.

```@example colored
τs = (1.0, 0.1, 0.01, 0.001)
nothing # hide
```

With this setup, we create the corresponding OU processes:

```@example colored
ou_noises = (OrnsteinUhlenbeckProcess(t0, tf, y0, 1/τ, √(2/τ)) for τ in τs)
nothing # hide
```

This `noises` is a `Tuple` where each element is an OU sampler. With each sampler `noises[i]`, we draw a sample path with `rand!(rng, noises[i], yt)`, with a given random number generator `rng` and a vector of floats `yt` of size `n`, so that this sampling fills up the pre-allocated vector `yt` with a sample path over the interval `t0` to `tf`, with the corresponding resolution `dt = (tf - t0) / (n - 1)`. For that, we set up the `rng`, used for reproducibility.

```@example colored
rng = Xoshiro(123)
```

Let us visualize a sample path of these process. We define the resolution, pre-allocate some vectors, and compute the sample paths.

```@example colored
w_t = Vector{Float64}(undef, n) # for the white noise
ous_t = [Vector{Float64}(undef, n) for τ in τs] # for the OU noises
nothing # hide
```

Now we draw sample paths of the OU noise

```@example colored
for (i, ou_n) in enumerate(ou_noises)
    rand!(rng, ou_n, ous_t[i])
end
```

as well as a white noise sample path
```@example colored
rand!(rng, Normal(), w_t)
nothing # hide
```

For the sake of comparison, let us check their mean and variance

```@example colored
for (τ, ou_t) in zip(τs, ous_t)
    println("OU process with τ=$τ:")
    println("  Mean $(mean(ou_t))")
    println("  Variance $(var(ou_t))")
end
println("White noise:")
println("  Mean $(mean(w_t))")
println("  Variance $(var(w_t))")
nothing # hide
```

Now we plot the obtained sample paths:

```@example colored
plts = [plot(tt, ou_t, xlabel="\$t\$", ylabel="\$y\$", guidefont=10, label="OU τ = $τ", legend=:topright) for (τ, ou_t) in zip(τs, ous_t)]
pltw = plot(tt, w_t, xlabel="\$t\$", ylabel="\$y\$", guidefont=10, legend=:topright)
plot(plts..., pltw, layout=(length(plts) + 1, 1), size=(600, 900))
```

We can also check the spectrum of each sample path signal, using [JuliaMath/FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/).

```@example colored
plts = [plot(abs2.(rfft(ou_t)), xlabel="\$k\$", ylabel="\$\\hat y\$", guidefont=10, label="OU τ = $τ spectrum", legend=:topright) for (τ, ou_t) in zip(τs, ous_t)]
pltw = plot(abs2.(rfft(w_t)), xlabel="\$k\$", ylabel="\$\\hat y\$", guidefont=10, label="white noise spectrum", legend=:topright)
plot(plts..., pltw, layout=(length(plts) + 1, 1), size=(600, 900))
```

For a proper evaluation of the statistics of the processes above, we should draw many samples and average them, but the above suffices for illustrative purposes.
