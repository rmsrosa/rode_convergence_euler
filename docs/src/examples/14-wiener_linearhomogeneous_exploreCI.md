```@meta
EditURL = "../../literate/examples/14-wiener_linearhomogeneous_exploreCI.jl"
```

# Testing the confidence regions and intervals 3


We consider a simple and quick-to-solve Random ODE to test the confidence regions and intervals. With a simple model, we can easily run a million simulations to test the statistics.

The Random ODE is a simple homogeneous linear equation in which the coefficient is a Wiener process and for which we know the distribution of the exact solution.

Now we consider an arbitrary number of mesh resolutions.

## The equation

We consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = W_t X_t, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a standard Wiener process.
The explicit solution is
```math
  X_t = e^{\int_0^t W_s \;\mathrm{d}s} X_0.
```

As seen in the first example of this documentation, once an Euler approximation is computed, along with realizations $\{W_{t_i}\}_{i=0}^n$ of a sample path of the noise, we consider an exact sample solution given by
```math
    X_{t_j} = X_0 e^{\sum_{i = 0}^{j-1}\left(\frac{1}{2}\left(W_{t_i} + W_{t_{i+1}}\right)(t_{i+1} - t_i) + Z_i\right)},
```
for realizations $Z_i$ drawn from a normal distribution and scaled by the standard deviation $\sqrt{(t_{i+1} - t_i)^3/12}$. This is implemented by computing the integral recursively, via
```math
    \begin{cases}
        I_j = I_{j-1} + \frac{1}{2}\left(W_{t_{j-1}} + W_{t_j}\right)(t_{j} - t_{j-1}) + Z_j, \\
        Z_j = \sqrt{\frac{(t_{j} - t_{j-1})^3}{12}} R_j, \\
        R_j \sim \mathcal{N}(0, 1), \\
    \end{cases}
```
with $I_0 = 0$, and setting
```math
  X_{t_j} = X_0 e^{I_j}.
```

## Setting up the problem

First we load the necessary packages

````@example 14-wiener_linearhomogeneous_exploreCI
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables, starting by choosing the `Xoshiro256++` pseudo-random number generator, and setting its seed for the sake of reproducibility:

````@example 14-wiener_linearhomogeneous_exploreCI
rng = Xoshiro(123)
nothing # hide
````

Next we set up the time interval and the initial distribution law for the initial value problem, which we take it to be a standard [Distributions.Normal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Normal) random variable:

````@example 14-wiener_linearhomogeneous_exploreCI
t0, tf = 0.0, 1.0
x0law = Normal()
````

The noise is a [`WienerProcess`](@ref) starting at ``y_0 = 0``:

````@example 14-wiener_linearhomogeneous_exploreCI
y0 = 0.0
noise = WienerProcess(t0, tf, y0)
````

There is no parameter in the equation, so we just set `params` to `nothing`.

````@example 14-wiener_linearhomogeneous_exploreCI
params = nothing
````

The number of mesh points for the approximations

````@example 14-wiener_linearhomogeneous_exploreCI
ns = 2 .^ (5:8)
````

The method for which we want to estimate the rate of convergence is, naturally, the Euler method, denoted [`RandomEuler`](@ref):

````@example 14-wiener_linearhomogeneous_exploreCI
method = RandomEuler()
````

We now have some choises of equations to test out.

````@example 14-wiener_linearhomogeneous_exploreCI
begin ## linear homogenous with Wiener noise
    f(t, x, y, p) = y * x

    ns = 2 .^ (4:10)

    target_solver! = function (xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, params::Q, rng::AbstractRNG) where {T, F, Q}
        axes(xt) == axes(yt) || throw(
            DimensionMismatch("The vectors `xt` and `yt` must match indices")
        )

        n = size(xt, 1)
        dt = (tf - t0) / (n - 1)
        i1 = firstindex(xt)
        xt[i1] = x0
        integral = zero(T)
        zscale = sqrt(dt^3 / 12)
        for i in Iterators.drop(eachindex(xt, yt), 1)
            integral += (yt[i] + yt[i1]) * dt / 2 + zscale * randn(rng)
            xt[i] = x0 * exp(integral)
            i1 = i
        end
    end

    target = CustomUnivariateMethod(target_solver!, rng)

    ntgt = 2^12
end

begin ## linear non-homogeneous with Wiener noise
    f(t, x, y, p) = - x + y

    # The *target* solution as described above is implemented as
    target_solver! = function (xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, params::Q, rng::AbstractRNG) where {T, F, Q}
        axes(xt) == axes(yt) || throw(
            DimensionMismatch("The vectors `xt` and `yt` must match indices")
        )

        n = size(xt, 1)
        dt = (tf - t0) / (n - 1)
        i1 = firstindex(xt)
        xt[i1] = x0
        integral = zero(T)
        ti1 = zero(T)
        zscale = sqrt(dt^3 / 12)
        for i in Iterators.drop(eachindex(xt, yt), 1)
            ti = ti1 + dt
            integral += (yt[i] - yt[i1]) * (exp(ti) - exp(ti1)) / dt +  zscale * randn(rng)
            xt[i] = exp(-ti) * (x0 - integral) + yt[i]
            ti1 = ti
            i1 = i
        end
    end

    target = CustomUnivariateMethod(target_solver!, rng)

    ntgt = 2^10
end

begin ## linear homogenous with sine of Wiener noise
    f(t, x, y, p) = sin(y) * x

    noise = GeometricBrownianMotionProcess(t0, tf, 1.0, 1.0, 0.2)

    target = RandomEuler()
    ntgt = 2^16
end

begin
    f(t, x, y, p) = - sum(abs, y) * x + prod(y)

    noise = ProductProcess(
        OrnsteinUhlenbeckProcess(t0, tf, 0.2, 0.3, 0.5),
        GeometricBrownianMotionProcess(t0, tf, 0.2, 0.3, 0.5),
        CompoundPoissonProcess(t0, tf, 5.0, Exponential(0.5)),
        ExponentialHawkesProcess(t0, tf, 0.5, 0.5, 0.5, Exponential(0.5))
    )

    target = RandomEuler()
    ntgt = 2^16
end

nothing # hide
````

## Statistics

Now, with the helper functions, we run a loop varying the number $m$ of samples in each run and the number $nk$ of test runs, showing some relevant statistics.

````@example 14-wiener_linearhomogeneous_exploreCI
m = 1
nk = 10000

nktenths = div(nk, 10)
nkfortieth = div(nk, 40)
nkhundredths = div(nk, 100)

suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)

deltas = (suite.tf - suite.t0) ./ suite.ns

@time allerrors = mapreduce(i -> solve(rng, suite).errors', vcat, 1:nk)
````

````@example 14-wiener_linearhomogeneous_exploreCI
logallerrors = log.(allerrors)

allmeans = mean(allerrors, dims=1)
````

````@example 14-wiener_linearhomogeneous_exploreCI
means_num = 100
@assert mod(nk, means_num) == 0
means_size = div(nk, means_num)
@assert means_num * means_size == nk
partialmeans = mapreduce(n -> mean(allerrors[n+1:n+1+means_size, :], dims=1), vcat, 0:means_num-1)

meansofmeans = mean(partialmeans, dims=1)
stdmeans = std(partialmeans, dims=1)
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plot(title="Pathwise and mean errors", titlefont=12, xscale=:log10, yscale=:log10, xlabel="\$\\Delta t\$", ylabel="\$\\textrm{error}\$", legend=:topleft, xlims=(last(deltas)/2, 2first(deltas)))
    scatter!(deltas, allerrors[1, :], color=:gray, alpha=0.1, label="pathwise errors ($nktenths samples)")
    scatter!(deltas, allerrors[2:nktenths, :]', label=false, color=:gray, alpha=0.01)
    scatter!(deltas, mean(allerrors[1:nkhundredths, :], dims=1)', label="mean errors ($nkhundredths samples)", color=1, alpha=0.8)
    scatter!(deltas, mean(allerrors[1:nkfortieth, :], dims=1)', label="mean errors ($nkfortieth samples)", color=2, alpha=0.8)
    scatter!(deltas, allmeans', label="mean errors ($nk samples)", color=3, alpha=0.8)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plts = Any[]
    for ncol in axes(allerrors, 2)
        pltcol = plot(title="Histogram of pathwise errors for \$\\delta= 2^{$(ns[ncol])}\$", titlefont=8)
        histogram!(pltcol, allerrors[1:nktenths, ncol], label="$nktenths samples")
        histogram!(pltcol, allerrors[1:nkfortieth, ncol], label="$nkfortieth samples")
        histogram!(pltcol, allerrors[1:nkhundredths, ncol], label="$nkhundredths samples")
        push!(plts, pltcol)
    end
    plot(size=(800, 300*div(length(ns), 2)), plts...)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plts = Any[]
    for ncol in axes(allerrors, 2)
        pltcol = plot(title="Histogram of log of pathwise errors for \$\\delta= 2^{$(ns[ncol])}\$", titlefont=8, legend=:topleft)
        histogram!(pltcol, logallerrors[1:nktenths, ncol], label="$nktenths samples")
        histogram!(pltcol, logallerrors[1:nkfortieth, ncol], label="$nkfortieth samples")
        histogram!(pltcol, logallerrors[1:nkhundredths, ncol], label="$nkhundredths samples")
        push!(plts, pltcol)
    end
    plot(size=(800, 300*div(length(ns), 2)), plts...)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plts = Any[]
    for ncol in axes(allerrors, 2)
        fitteddists = [
            (dist, fit(dist, logallerrors[:, ncol]) ) for dist in (Normal, Cauchy)
        ]
        pltcol = plot(title="Histogram and fit of log of pathwise errors for \$\\delta= 2^{$(ns[ncol])}\$", titlefont=8, legend=:topleft)
        histogram!(pltcol, logallerrors[1:nktenths, ncol], normalize=:pdf, label="$nktenths samples")
        for fitteddist in fitteddists
            plot!(pltcol, x -> pdf(fitteddist[2], x), label="fitted $(fitteddist[1])")
        end
        plot!(pltcol, x -> pdf(Logistic(mean(logallerrors[:, ncol]), √3*var(logallerrors[:, ncol])/π), x), label="adjusted Logistic")
        push!(plts, pltcol)
    end
    plot(size=(800, 300*div(length(ns), 2)), plts...)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plts = Any[]
    for ncol in axes(allerrors, 2)
        fittedlognormal = fit(LogNormal, allerrors[:, ncol])
        fittedexponential = fit(Exponential, allerrors[:, ncol])
        pltcol = plot(title="Histogram and fit of log of pathwise errors for \$\\delta= 2^{$(ns[ncol])}\$", titlefont=8, legend=:topleft, xlims=(0.0, mean(allerrors[:, ncol]) + 2std(allerrors[:, ncol])))
        histogram!(pltcol, allerrors[1:nktenths, ncol], normalize=:pdf, label="$nktenths samples")
        plot!(pltcol, x -> pdf(fittedlognormal, x), label="fitted logNormal")
        plot!(pltcol, x -> pdf(fittedexponential, x), label="fitted Exponential")
        push!(plts, pltcol)
    end
    plot(size=(800, 300*div(length(ns), 2)), plts..., legend=:topright)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plts = Any[]
    for ncol in axes(allerrors, 2)
        pltcol = plot(title="Histogram of  error means for \$\\delta= 2^{$(ns[ncol])}\$", titlefont=8)
        histogram!(pltcol, partialmeans[:, ncol], nbins = 40, label="$means_num means from $means_size samples")
        push!(plts, pltcol)
    end
    plot(size=(800, 300*div(length(ns), 2)), plts...)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
plt = begin
    plot(log.(deltas), log.(allerrors[1:100, :]'), alpha=0.5, legend=false)
    plot!(log.(deltas), l -> l + 1, color=:black, linewidth=2)
    plot!(log.(deltas), l -> l - 1.5, color=:black, linewidth=2)
    plot!(log.(deltas), l -> l - 4, color=:black, linewidth=2)
    plot!(log.(deltas), l -> l/2 - 10, color=:red, alpha=0.5, linewidth=2)
    plot!(log.(deltas), l -> 2l - 1, color=:blue, alpha=0.5, linewidth=2)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
A = [one.(deltas) log.(deltas)]
L = inv(A' * A) * A'
lCps = reduce(hcat, [L * log.(allerrors[n, :]) for n in axes(allerrors, 1)])

extrema(lCps[2, :])
pmeans = mean(lCps[2, :])
Nps = fit(Normal, lCps[2, :])

plt = begin
    plot(title="Order of convergence of individual path samples")
    histogram(lCps[2, :], label="\$p(\\omega)\$", normalize=:pdf)
    vline!([pmeans], linewidth=2, label="\$\\mathbb{E}[p] = $(round(pmeans, digits=2))\$")
    plot!(p -> pdf(Nps, p), label="fitted \$\\mathcal{N}($(round(Nps.μ, digits=2)), $(round(Nps.σ, digits=2))^2)\$")
end

pbounds = [minimum( ( log.(allerrors[n, :]) .- maximum(lCps[1, :]) ) ./ log.(deltas) ) for n in axes(allerrors, 1)]

plt = begin
    plot(title="Not right quantity to look at")
    histogram(pbounds)
end

nothing # hide

pboundsev = [minimum( ( log.(allerrors[n, 1:k]) .- maximum(lCps[1, 1:k]) ) ./ log.(deltas[1:k]) ) for n in axes(allerrors, 1), k in 2:7]

begin
    histogram(pboundsev[:, 1], alpha=0.1)
    histogram!(pboundsev[:, 2], alpha=0.1)
    histogram!(pboundsev[:, 3], alpha=0.1)
    histogram!(pboundsev[:, 4], alpha=0.1)
    histogram!(pboundsev[:, 5], alpha=0.1)
    histogram!(pboundsev[:, 5], alpha=0.1)
end
````

````@example 14-wiener_linearhomogeneous_exploreCI
nothing
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

