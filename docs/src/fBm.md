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

## Plotting some sample paths

Here we generate and plot a few sample paths with different Hurst parameters.

First with a Hurst parameter within the range $1/2 < H < 1$.

```@example fBm
N = 2^9
t0 = 0.0
T = 2.0
tt = range(t0, T, length=N)
y0 = 0.0
H = 0.8

noise! = Dict(H => fBm_noise(t0, T, y0, H, N))

Yt = Vector{Float64}(undef, N)

rng = Xoshiro(123)

plt = plot(title="Sample paths of fractional Brownian motion of length $N with Hurst parameter H=$H", titlefont=8, xlabel="t", ylabel="W", legend=nothing, size=(800, 400))
for _ in 1:3
    noise![H](rng, Yt)
    plot!(plt, tt, Yt)
end
plt
```

Now with $H=1/2$, which yields a standard Brownian motion.

```@example fBm
H = 0.5

noise![H] = fBm_noise(t0, T, y0, H, N)

plt = plot(title="Sample paths of fractional Brownian motion of length $N with Hurst parameter H=$H", titlefont=8, xlabel="t", ylabel="W", legend=nothing, size=(800, 400))
for _ in 1:3
    noise![H](rng, Yt)
    plot!(plt, tt, Yt)
end
plt
```

Finally a rougher path with $0 < H < 1/2$.

```@example fBm
H = 0.2

noise![H] = fBm_noise(t0, T, y0, H, N)

plt = plot(title="Sample paths of fractional Brownian motion of length $N with Hurst parameter H=$H", titlefont=8, xlabel="t", ylabel="W", legend=nothing, size=(800, 400))
for _ in 1:3
    noise![H](rng, Yt)
    plot!(plt, tt, Yt)
end
plt
```

## Checking some basic statistics

Now we simulate a bunch of sample paths and check their statistics.

First with $1/2 < H < 1$.

```@example fBm
H = 0.8

W = Dict(H => reduce(hcat, RODEConvergence.fBm_daviesharte(rng, T, N, H) for _ in 1:1_000))

means = mean(W[H], dims=2)
stds = std(W[H], dims=2)
nothing
```

```@example fBm
plt = plot(title="Sample paths, mean, and std")
plot!(plt, tt, view(W[H], :, 1:100), alpha=0.05, color=1, label=nothing)
plot!(plt, tt, means, label="Mean")
plot!(plt, tt, means .+ stds, label="Stds", color=7)
plot!(plt, tt, means .- stds, label=nothing, color=7)
plot!(plt, tt, tt.^H, label="theoretical", color=:black, style=:dash)
plot!(plt, tt, -tt.^H, label=nothing, color=:black, style=:dash)
```

Now with $H=1/2$.

```@example fBm
H = 0.5

W[H] = reduce(hcat, RODEConvergence.fBm_daviesharte(rng, T, N, H) for _ in 1:1_000)

means = mean(W[H], dims=2)
stds = std(W[H], dims=2)
nothing
```

```@example fBm
plt = plot(title="Sample paths, mean, and std")
plot!(plt, tt, view(W[H], :, 1:100), alpha=0.05, color=1, label=nothing)
plot!(plt, tt, means, label="Mean")
plot!(plt, tt, means .+ stds, label="Stds", color=7)
plot!(plt, tt, means .- stds, label=nothing, color=7)
plot!(plt, tt, tt.^H, label="theoretical", color=:black, style=:dash)
plot!(plt, tt, -tt.^H, label=nothing, color=:black, style=:dash)
```

Finally with $0 < H < 1/2$.

```@example fBm
H = 0.2

W[H] = reduce(hcat, RODEConvergence.fBm_daviesharte(rng, T, N, H) for _ in 1:1_000)

means = mean(W[H], dims=2)
stds = std(W[H], dims=2)
nothing
```

```@example fBm
plt = plot(title="Sample paths, mean, and std")
plot!(plt, tt, view(W[H], :, 1:100), alpha=0.05, color=1, label=nothing)
plot!(plt, tt, means, label="Mean")
plot!(plt, tt, means .+ stds, label="Stds", color=7)
plot!(plt, tt, means .- stds, label=nothing, color=7)
plot!(plt, tt, tt.^H, label="theoretical", color=:black, style=:dash)
plot!(plt, tt, -tt.^H, label=nothing, color=:black, style=:dash)
```

## Checking the probability distribution function

At each time $t$, the distribution of $B_H(t)$ is a Normal distribution with mean 0 and standard deviation $t^H$. So we check this by plotting the normalized histogram of the simulated paths, along with the theoretical distribution, for a few instants of time.

```@example fBm
H = 0.2
xx = -3*last(tt)^H:0.01:3*last(tt)^H
plts = []
for n in div.(N, (100, 10, 2, 1))
    plt = plot(title="Histogram and PDF for H=$H at t=$(round(tt[n], sigdigits=3))", xlims=(first(xx), last(xx)), titlefont=8, legend=nothing)
    histogram!(plt, view(W[H], n, :), bins=40, normalize=true)
    plot!(plt, xx, x -> pdf(Normal(0.0, tt[n]^H), x))
    push!(plts, plt)
end
plot(plts...)
```

```@example fBm
H = 0.5
xx = -3*last(tt)^H:0.01:3*last(tt)^H
plts = []
for n in div.(N, (100, 10, 2, 1))
    plt = plot(title="Histogram and PDF for H=$H at t=$(round(tt[n], sigdigits=3))", xlims=(first(xx), last(xx)), titlefont=8, legend=nothing)
    histogram!(plt, view(W[H], n, :), bins=40, normalize=true)
    plot!(plt, xx, x -> pdf(Normal(0.0, tt[n]^H), x))
    push!(plts, plt)
end
plot(plts...)
```

```@example fBm
H = 0.8
xx = -3*last(tt)^H:0.01:3*last(tt)^H
plts = []
for n in div.(N, (100, 10, 2, 1))
    plt = plot(title="Histogram and PDF for H=$H at t=$(round(tt[n], sigdigits=3))", xlims=(first(xx), last(xx)), titlefont=8, legend=nothing)
    histogram!(plt, view(W[H], n, :), bins=40, normalize=true)
    plot!(plt, xx, x -> pdf(Normal(0.0, tt[n]^H), x))
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
H = 0.2
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

```@example fBm
@btime noise![H](rng, Yt);
```
