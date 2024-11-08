# Homogeneous linear Itô process noise

```@meta
    Draft = false
```

A classical type of noise is the geometric Brownian motion (gBm) process $\{Y_t\}_t$ satisfying
```math
\mathrm{d}Y_t = \mu Y_t \;\mathrm{d}t + \sigma Y_t \;\mathrm{d}W_t,
```
where $\mu$ is a drift parameter and $\sigma$ is a diffusion parameter, both assumed constant. It appears naturally when the *specific growth* parameter $\mu$ in the classical growth model
```math
\frac{\mathrm{d}x}{\mathrm{d}t} = \mu x,
```
is substituted by a randomly varying parameter given as a stochastic process $\{M_t\}$ with a base growth rate $\mu$ "perturbed" by a white noise, i.e.
```math
M_t = \mu + \sigma\frac{\mathrm{d}W_t}{\mathrm{d}t},
```
for some disturbance *amplitude* $\sigma > 0.$ This means rigorously that
```math
\mathrm{d}M_t = \mu\;\mathrm{d}t + \sigma\mathrm{d}W_t.
```
Thus, the natural-growth ordinary differential equation becomes the stochastic differential equation known as the geometric Brownian motion $\{Y_t\}_t$.

More generally, the basal coefficient $\mu$ and the diffusion coefficient $\sigma$ in the equation for $M_t$ may also vary deterministically for seasonal reasons, on top of the random disturbances, so it is natural to consider the non-autonomous version
```math
\mathrm{d}M_t = a(t)\;\mathrm{d}t + b(t)\mathrm{d}W_t.
```

In this case, the gBm equation is substituted by a more general homogeneous linear equation with time-dependent coefficients,
```math
\mathrm{d}Y_t = a(t) Y_t \;\mathrm{d}t + b(t) Y_t \;\mathrm{d}W_t.
```

This equation has the explicit solution
```math
Y_t = y_0 e^{\int_0^t (a(s) - \frac{b(s)^2}{2}) \;\mathrm{d}s + \int_0^t b(s) \;\mathrm{d}W_s}.
```

This can also be written in a recursive way via the expression
```math
Y_{t+\tau} = Y_t e^{\int_t^{t+\tau} (a(s) - \frac{b(s)^2}{2}) \;\mathrm{d}s + \int_t^\tau b(s) \;\mathrm{d}W_s}.
```

When primitives of $a=a(t)$ and $b^2=b(t)^2$ are given, a distributionally exact discretization $\{Y_{t_j}\}_j$ can be computed on the mesh points $(t_j)_j$ in a recursive manner by
```math
Y_{t_j} = Y_{t_{j-1}} e^{(p_a(t_j) - p_a(t_{j-1})) - (p_{b^2}(t_j) - p_{b^2}(t_{j-1})/2 + Z_j)}, \qquad j = 1, \ldots,
```
with $Y_0 = y_0$, and where $p_a = p_a(t)$ is the given primitive of $a=a(t),$ $p_{b^2} = p_{b^2}(t)$ is the given primitive of $b^2 = b(t)^2,$ and $\{Z_j\}_j$ is a random vector with independent, normally distributed random variables
```math
Z_j \sim \mathcal{N}(0, p_{b^2}(t_j) - p_{b^2}(t_{j-1})).
```

The basic statistics (mean and variance) for this process can be computed by first computing the statistics for its logarithm, 
```math
\ln Y_t = \ln y_0 + \int_0^t (a(s) - \frac{b(s)^2}{2}) \;\mathrm{d}s + \int_0^t b(s) \;\mathrm{d}W_s,
```
which is a Gaussian process satisfying
```math
\mathbb{E}\left[ \ln Y_t \right] = \ln y_0 + \int_0^t (a(s) - \frac{b(s)^2}{2}) \;\mathrm{d}s,
```
and
```math
\mathrm{Var}\left( \ln Y_t \right) = \int_0^t b(s)^2 \;\mathrm{d}s.
```

Since each ``\ln Y_t`` is Gaussian, ``Y_t`` is log-normal with
```math
\mathbb{E}\left[ Y_t \right] = y_0 e^{\int_0^t a(s) \;\mathrm{d}s},
```
and
```math
\mathrm{Var}\left( Y_t \right) = y_0^2 e^{\int_0^t 2a(s) \;\mathrm{d}s}\left( e^{\int_0^t b(s)^2 \;\mathrm{d}s} - 1 \right).
```

We simulate, below, some examples of homogeneous linear processes for illustration purposes and for exhibiting the correctness of the implementation.

## Loading the packages

We start by loading the necessary packages:

```@example homlin
using Random
using Statistics
using Plots
using Test
using RODEConvergence
```

## Setup

For the setup, we first set the time interval, the mesh parameters, and the number of sample paths, common to all simulations,
```@example homlin
t0 = 0.0
tf = 2.0
n = 2^10
tt = range(t0, tf, length=n+1)
dt = (tf - t0) / n
m = 200
nothing # hide
```

### Geometric Brownian Motion

For the gBm process, we have $a(t) = \mu$ and $b(t) = \sigma$ constant. This can be implemented with the gBm constructor
```@example homlin
y0 = 0.4
μ = 0.3
σ = 0.2
noise_gbm = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)
nothing # hide
```
or via the homogeneous linear Itô process constructor, by given the primitive of the constants $\mu$ and $\sigma^2$, which in this case are just linear functions:
```@example homlin
primitive_a = t -> μ * t
primitive_b2 = t -> σ^2 * t
noise = HomogeneousLinearItoProcess(t0, tf, y0, primitive_a, primitive_b2)
nothing # hide
```

For comparison, we generate a bunch of sample paths with both constructors and check their statistics.

```@example homlin
rng = Xoshiro(123)

gBmt = Matrix{Float64}(undef, n+1, m)
Yt = Matrix{Float64}(undef, n+1, m)

for j in 1:m
    rand!(rng, noise, view(gBmt, :, j))
    rand!(rng, noise, view(Yt, :, j))
end
```

We check their expectation and variance against the theoretical values.
```@example homlin
@testset "Statistics of gBm and linear" begin
    @test mean(view(Yt, lastindex(Yt, 1), :)) ≈ y0 * exp(μ * tf) (atol = 0.1)
    @test mean(view(gBmt, lastindex(Yt, 1), :)) ≈ y0 * exp(μ * tf) (atol = 0.1)
    @test var(view(Yt, lastindex(Yt, 1), :)) ≈ y0^2 * exp( 2μ * tf ) * ( exp( ( σ^2 ) * tf ) - 1 ) (atol = 0.1)
    @test var(view(gBmt, lastindex(Yt, 1), :)) ≈ y0^2 * exp( 2μ * tf ) * ( exp( ( σ^2 ) * tf ) - 1 ) (atol = 0.1)
end
nothing # hide
```

We plot the ensemble of paths just for the sake of illustration, along with the theoretical expected mean value.
```@example homlin
plt = plot(title="Sample paths for the gBm process", titlefont=10)
plot!(plt, tt, t -> y0 * exp( μ * t), label="expectation", color = 2)
plot!(plt, tt, Yt, xlabel="\$t\$", ylabel="\$y\$", guidefont=10, label=nothing, color=1, alpha=0.2)
```

### Exponential decay with oscilatory random perturbation

Now we consider the homogeneous linear Itô process noise defined by a decaying drift factor and an oscillatory diffusion,
```math
\mathrm{d}Y_t = -2t Y_t \;\mathrm{d}t + \sin(3\pi t) Y_t \;\mathrm{d}W_t.
```

With the following primitives of $a(t) = -2t$ and $b(t)^2 = \sin(3\pi t)^2:$
```math
p_a(t) = -t^2, \qquad p_{b^2}(t) = \frac{1}{2}\left(t - \frac{1}{3\pi}\sin(3\pi t)\cos(3\pi t)\right),
```
we define the process as
```@example homlin
primitive_a = t -> -t^2
primitive_b2 = t -> t/2 - sin(3π * t) * cos(3π * t) / 6π 
noise = HomogeneousLinearItoProcess(t0, tf, y0, primitive_a, primitive_b2)
```

Now we simulate a number of sample paths:

```@example homlin
rng = Xoshiro(123)

for j in 1:m
    rand!(rng, noise, view(Yt, :, j))
end
```

We check their expectation and their variance against the theoretical values in this case, as well.
```@example homlin
@testset "Statistics" begin
    @test mean(view(Yt, lastindex(Yt, 1), :)) ≈ y0 * exp(primitive_a(tf)) (atol = 0.1)
    @test var(view(Yt, lastindex(Yt, 1), :)) ≈ y0^2 * exp( 2 * primitive_a(tf) ) * ( exp( primitive_b2(tf) ) - 1 ) (atol = 0.1)
end
nothing # hide
```

For the sake of illustration, we plot the computed sample paths, along with the theoretical expected mean value.
```@example homlin
plt = plot(title = "Sample paths for the homogeneous linear RODE with exponential decay", titlefont=10)
plot!(plt, tt, t -> y0 * exp(primitive_a(t)), label="expectation", color = 2)
plot!(plt, tt, Yt, xlabel="\$t\$", ylabel="\$y\$", guidefont=10, label=nothing, color=1, alpha=0.2)
```

### Exponential growth with oscilatory random perturbation

Now we consider the homogeneous linear Itô process noise defined by a decaying drift factor and an oscillatory diffusion factor,
```math
\mathrm{d}Y_t = 0.2t Y_t \;\mathrm{d}t + 0.1\sin(3\pi t) Y_t \;\mathrm{d}W_t.
```

We consider the following primitives of $a(t) = t/5$ and $b(t)^2 = \sin(3\pi t)^2/10:$
```math
p_a(t) = \frac{t^2}{10}, \qquad p_{b^2}(t) = \frac{1}{20}\left(t - \frac{1}{3\pi}\sin(3\pi t)\cos(3\pi t)\right).
```

Hence, we define the process as

```@example homlin
y0 = 0.1
primitive_a = t -> t^2/10
primitive_b2 = t -> t/20 - sin(3π * t) * cos(3π * t) / 60π 
noise = HomogeneousLinearItoProcess(t0, tf, y0, primitive_a, primitive_b2)
nothing # hide
```

```@example homlin
rng = Xoshiro(123)
nothing # hide
```

```@example homlin
for j in 1:m
    rand!(rng, noise, view(Yt, :, j))
end
```

We check their expectation and their variance against the theoretical values.
```@example homlin
@testset "Statistics" begin
    @test mean(view(Yt, lastindex(Yt, 1), :)) ≈ y0 * exp(primitive_a(tf)) (atol = 0.1)
    @test var(view(Yt, lastindex(Yt, 1), :)) ≈ y0^2 * exp( 2 * primitive_a(tf) ) * ( exp( primitive_b2(tf) ) - 1 ) (atol = 0.1)
end
nothing # hide
```

For the sake of illustration, we plot the computed sample paths, along with the theoretical expected mean value.
```@example homlin
plt = plot(title = "Sample paths for the homogeneous linear RODE with exponential growth", titlefont=10)
plot!(plt, tt, t -> y0 * exp(primitive_a(t) - 2sqrt(primitive_b2(t))), fillrange = t -> y0 * exp(primitive_a(t) + 2sqrt(primitive_b2(t))), alpha=0.5, color=2, label="95% region")
plot!(plt, tt, t -> y0 * exp(primitive_a(t)), label="expectation", color = 2)
plot!(plt, tt, Yt, xlabel="\$t\$", ylabel="\$y\$", guidefont=10, label=nothing, color=1, alpha=0.2)
```
