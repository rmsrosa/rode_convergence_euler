# Simulating fractional Brownian motion

We've implemented the Davies-Harte method for simulating exact sample paths of a fractional Brownian motion $\{B_H(t)\}_{0\leq t \leq T}$ with Hurst parameter $0 < H < 1$. There are a number of methods to simulate exact sample paths, such as the Cholesky method, which is of order $O(N^3)$, and the Hosking method, which is $O(N^2)$. These are too expensive for our needs, since we plan to build lots of sample paths with a million points. So we chose to implement the Davies-Harte method, which uses fast fourier transform (FFTs) to achieve $O(N\log N)$. For that, we rely on [FFTW.jl](https://github.com/JuliaMath/FFTW.jl), which has the julia bindings to the [FFTW](http://www.fftw.org) library.

The point in this section is to illustrate the use of the implemented method and to do some simple sanity checks to make sure our implementation is correct.

## Loading the packages

We start by loading the necessary packages:

```@example fBm
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

We fix the initial condition `y0` and the initial and final times `t0` and `tf` of the time interval of the desired fractional Brownian motion process. We also fix the size $N$ of the sample paths. The sample paths will be generated for a set of `n` times uniformly distributed within the time interval from `t0` to `tf`, which yields a time mesh which we define as `tt`:

```@example fBm
y0 = 0.0
t0 = 0.0
tf = 2.0
n = 2^9
tt = range(t0, tf, length=n+1)
nothing # hide
```

We also choose a few values of the Hurst parameter for the tests, one within $0 < H < 1/2$, one exactly $H = 1/2$ corresponding to the standard Brownian motion process, and one within $1/2 < H < 1.$

```@example fBm
Hs = (0.2, 0.5, 0.8)
nothing # hide
```

With this setup, we create the corresponding fractional Brownian motion processes:

```@example fBm
noise = Dict(H => FractionalBrownianMotionProcess(t0, tf, y0, H, n) for H in Hs)
nothing # hide
```

This `noise` is a `Dict` with the keys being the chosen Hurst parameters and with each `noise[H]` being a fractional Brownian noise sampler with the corresponding `H` in `Hs`. With each sampler, we draw a sample path with `rand!(rng, noise[H], Yt)`, with a random number generator `rng` and a vector of floats `Yt` of size `N`, so that this sampling fills up the pre-allocated vector `Yt` with a sample path. For that, we set up the `rng`, used for reproducibility, and create the vector `Yt`.

```@example fBm
rng = Xoshiro(123)
yt = Vector{Float64}(undef, n+1)
nothing # hide
```

## Plotting some sample paths

Here we generate and plot a few sample paths with the different Hurst parameters.

First with a Hurst parameter within the range $1/2 < H < 1$.

```@example fBm
H = Hs[3]

plt = plot(title="Sample paths of fractional Brownian motion of length $n with Hurst parameter H=$H", titlefont=8, xlabel="t", ylabel="W", legend=nothing, size=(800, 400))
for _ in 1:3
    rand!(rng, noise[H], yt)
    plot!(plt, tt, yt)
end
plt
```

Now with $H=1/2$, which yields a standard Brownian motion.

```@example fBm
H = Hs[2]

plt = plot(title="Sample paths of fractional Brownian motion of length $n with Hurst parameter H=$H", titlefont=8, xlabel="t", ylabel="W", legend=nothing, size=(800, 400))
for _ in 1:3
    rand!(rng, noise[H], yt)
    plot!(plt, tt, yt)
end
plt
```

Finally a rougher path with $0 < H < 1/2$.

```@example fBm
H = Hs[1]

plt = plot(title="Sample paths of fractional Brownian motion of length $n with Hurst parameter H=$H", titlefont=8, xlabel="t", ylabel="W", legend=nothing, size=(800, 400))
for _ in 1:3
    rand!(rng, noise[H], yt)
    plot!(plt, tt, yt)
end
plt
```

## Checking some basic statistics

Now we simulate a bunch of sample paths and check their statistics. We start by defining how much is "a bunch":

```@example fBm
m = 200
nothing # hide
```

Now we generate the sets of sample paths for each Hurst parameter.

```@example fBm
W = Dict(H => Matrix{Float64}(undef, n+1, m) for H in Hs)
for H in Hs
    for i in 1:m
        rand!(rng, noise[H], view(W[H], :, i))
    end
end

means = Dict(H => mean(W[H], dims=2) for H in Hs)
stds = Dict(H => std(W[H], dims=2) for H in Hs)
nothing # hide
```

```@example fBm
H = Hs[3]

plt = plot(title="Sample paths, mean, and std")
plot!(plt, tt, view(W[H], :, 1:100), alpha=0.05, color=1, label=nothing)
plot!(plt, tt, means[H], label="Mean")
plot!(plt, tt, means[H] .+ stds[H], label="Stds", color=7)
plot!(plt, tt, means[H] .- stds[H], label=nothing, color=7)
plot!(plt, tt, tt.^H, label="theoretical", color=:black, style=:dash)
plot!(plt, tt, -tt.^H, label=nothing, color=:black, style=:dash)
```

Now with $H=1/2$.

```@example fBm
H = Hs[2]

plt = plot(title="Sample paths, mean, and std")
plot!(plt, tt, view(W[H], :, 1:100), alpha=0.05, color=1, label=nothing)
plot!(plt, tt, means[H], label="Mean")
plot!(plt, tt, means[H] .+ stds[H], label="Stds", color=7)
plot!(plt, tt, means[H] .- stds[H], label=nothing, color=7)
plot!(plt, tt, tt.^H, label="theoretical", color=:black, style=:dash)
plot!(plt, tt, -tt.^H, label=nothing, color=:black, style=:dash)
```

Finally with $0 < H < 1/2$.

```@example fBm
H = Hs[1]

plt = plot(title="Sample paths, mean, and std")
plot!(plt, tt, view(W[H], :, 1:100), alpha=0.05, color=1, label=nothing)
plot!(plt, tt, means[H], label="Mean")
plot!(plt, tt, means[H] .+ stds[H], label="Stds", color=7)
plot!(plt, tt, means[H] .- stds[H], label=nothing, color=7)
plot!(plt, tt, tt.^H, label="theoretical", color=:black, style=:dash)
plot!(plt, tt, -tt.^H, label=nothing, color=:black, style=:dash)
```

## Checking the probability distribution function

At each time $t$, the distribution of $B_H(t)$ is a Normal distribution with mean 0 and standard deviation $t^H$. So we check this by plotting the normalized histogram of the simulated paths, along with the theoretical distribution, for a few instants of time.

```@example fBm
H = Hs[1]
xx = -3*last(tt)^H:0.01:3*last(tt)^H
plts = []
for ni in div.(n, (100, 10, 2, 1))
    plt = plot(title="Histogram and PDF for H=$H at t=$(round(tt[ni], sigdigits=3))", xlims=(first(xx), last(xx)), titlefont=8, legend=nothing)
    histogram!(plt, view(W[H], ni, :), bins=40, normalize=true)
    plot!(plt, xx, x -> pdf(Normal(0.0, tt[ni]^H), x))
    push!(plts, plt)
end
plot(plts...)
```

```@example fBm
H = Hs[2]
xx = -3*last(tt)^H:0.01:3*last(tt)^H
plts = []
for ni in div.(n, (100, 10, 2, 1))
    plt = plot(title="Histogram and PDF for H=$H at t=$(round(tt[ni], sigdigits=3))", xlims=(first(xx), last(xx)), titlefont=8, legend=nothing)
    histogram!(plt, view(W[H], ni, :), bins=40, normalize=true)
    plot!(plt, xx, x -> pdf(Normal(0.0, tt[ni]^H), x))
    push!(plts, plt)
end
plot(plts...)
```

```@example fBm
H = Hs[3]
xx = -3*last(tt)^H:0.01:3*last(tt)^H
plts = []
for ni in div.(n, (100, 10, 2, 1))
    plt = plot(title="Histogram and PDF for H=$H at t=$(round(tt[ni], sigdigits=3))", xlims=(first(xx), last(xx)), titlefont=8, legend=nothing)
    histogram!(plt, view(W[H], ni, :), bins=40, normalize=true)
    plot!(plt, xx, x -> pdf(Normal(0.0, tt[ni]^H), x))
    push!(plts, plt)
end
plot(plts...)
```

## Checking the covariance

In theory, the covariance of a fractional Brownian motion $\{B_H(t)\}_{t}$ with Hurst parameter $H$ should be given by

```math
    \rho(t, s) = \mathbb{E}[B_H(t)B_H(s)] = \frac{1}{2}\left\{ t^{2H} + s^{2H} - |t - s|^{2H}\right\}.
```

So, we compute the covariance from the generated sample paths and check the relative difference from the expected exact covariance:

```@example fBm
H = Hs[1]
covmatW = cov(W[H], dims=2, corrected=false)
covmat = [0.5*(t^(2H) + s^(2H) - abs(t - s)^(2H)) for t in tt, s in tt]
extrema(covmatW .- covmat)
```

Let us visualize the covariance of the sample paths:

```@example fBm
heatmap(tt, tt, covmatW, title="Covariance of the generated samples", titlefont=8, xlabel="t", ylabel="s")
```

And compare it with the theoretical covariance:

```@example fBm
heatmap(tt, tt, covmat, title="Theoretical covariance", titlefont=8, xlabel="t", ylabel="s")
```

Apparently there are just numerical errors. Here are the heatmap and the surface plots of the difference:

```@example fBm
heatmap(tt, tt, covmatW .- covmat, title="Difference", titlefont=8, xlabel="t", ylabel="s")
```

```@example fBm
surface(tt, tt, covmatW .- covmat, title="Difference", titlefont=8, xlabel="t", ylabel="s")
```

Looks good!

## Benchmark

The function `fBm_noise(t0, T, y0, H, N)` returns a fractional Brownian motion *sampler.* That means when we set `noise = FractionalBrownianMotionProcess(t0, T, y0, H, N)`, then `noise` is a sampler, from which we draw sample paths with `rand!(rng, noise, Yt)`, filling up the preallocated vector `Yt` with a sample path.

When calling `FractionalBrownianMotionProcess(t0, T, y0, H, N)`, a composite type is created containing cache vectors to hold the intermediate results needed when generating each sample path, including going through inverse and direct fast Fourier transforms. Besised creating the auxiliary cache vectors, it also builds the FFT plans used in the FFT transforms via [FFTW.jl](https://github.com/JuliaMath/FFTW.jl). In this way, the resulting sampler `noise` contains everthing pre-allocated so generating a sample is non-allocating. This can be benchmarked as follows.

```@example fBm
H = Hs[1]
@btime rand!($rng, $(noise[H]), $yt)
nothing # hide
```
