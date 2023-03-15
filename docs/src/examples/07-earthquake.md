```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/examples/07-earthquake.jl"
```

# Earthquake model

Now we consider a mechanical structure problem under ground-shaking excitations, based on Earthquake models, especially the Kanai-Tajimi model.

The mechanical structure is forced by a stochastic noise modeling the effects of an Earthquake. Several types of noises have been considered in the literature. A typical one is a white noise. Further studies show that the noise is actually a colored noise. So, we model the noise with a Orsntein-Uhlenbeck (OU) process with a relative small type scale $\tau$ with drift $\nu = 1/\tau$, and relatively large dissipation $\sigma$. When $\sigma/\nu \rightarrow \infty$, the colored-noise OU process approaches a white noise.

Moreover, in order to simulate the start of the first shock-wave and the subsequent aftershocks, we module the OU process with a transport process composed of a series of time-translations of a initially Hölder-continuous front with exponential decay, $\gamma (t - \delta)^\alpha e^{-\beta (t - \delta)}$, for $t \geq \delta$, with random parameters $\alpha, \beta, \gamma, \delta$, with arbitrarly small Hölder exponents $\alpha$.


## The equation

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 07-earthquake
using Plots
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
````

Then we set up some variables

````@example 07-earthquake
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
````

The structure initially at rest

````@example 07-earthquake
x0law = product_distribution(Dirac(0.0), Dirac(0.0))
````

The noise is a Wiener process modulated by a transport process

````@example 07-earthquake
y0 = 0.0
θ = 200.0 # = 1 / 0.005 => time-scale = 0.005
σ = 20.0
noise1 = OrnsteinUhlenbeckProcess(t0, tf, y0, θ, σ)

ylaw = product_distribution(Uniform(0.0, 2.0), Uniform(0.0, 0.5), Uniform(2.0, 8.0), Exponential())
nr = 5
g(t, r) = mapreduce(ri -> ri[1] * max(0.0, t - ri[4]) ^ ri[2] * exp(-ri[3] * max(0.0, t - ri[4])), +, eachcol(r))
noise2 = TransportProcess(t0, tf, ylaw, g, nr)

noise = ProductProcess(noise1, noise2)
````

````@example 07-earthquake
ntgt = 2^12
yt1 = Vector{Float64}(undef, ntgt)
yt2 = similar(yt1)

rand!(rng, noise1, yt1)
rand!(rng, noise2, yt2)

noise3 = WienerProcess(t0, tf, y0)
yt3 = similar(yt)
rand!(rng, noise3, yt3)
dt = (tf - t0) / (length(yt) - 1)

begin
    plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(t0+dt:dt:tf, (yt3[2:end] .- yt3[1:end-1])/dt^0.5, label="white noise")
    plot!(t0:dt:tf, yt1, label="OU")
    plot!(t0:dt:tf, yt3, label="Wiener")
end
````

````@example 07-earthquake
mean((yt3[2:end] .- yt3[1:end-1])/dt^0.5)
````

````@example 07-earthquake
mean(yt1)
````

````@example 07-earthquake
std((yt3[2:end] .- yt3[1:end-1])/dt^0.5)
````

````@example 07-earthquake
std(yt1)
````

````@example 07-earthquake
begin
    plot(xlabel="\$t\$", ylabel="\$\\mathrm{intensity}\$", guidefont=10)
    plot!(t0:dt:tf, yt2 .* yt1, label="noise")
    plot!(t0:dt:tf, yt2, label="envelope")
end
````

````@example 07-earthquake
ntgt = 2^18
ns = 2 .^ (5:9)
nsample = ns[[1, 2, 3, 4]]
m = 1_000
````

And add some information about the simulation:

````@example 07-earthquake
info = (
    equation = "Kanai-Tajimi model",
    noise = "Orstein-Uhlenbeck modulated by a transport process",
    ic = "\$X_0 = \\mathbf{0}\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 07-earthquake
target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))
````

### Order of convergence

With all the parameters set up, we build the convergence suite:

````@example 07-earthquake
suite = ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors:

````@example 07-earthquake
@time result = solve(rng, suite)
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 07-earthquake
table = generate_error_table(result, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 07-earthquake
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")
````

### Plots

We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 07-earthquake
plot(result)
````

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide

For the sake of illustration, we plot a sample of an approximation of a target solution:

````@example 07-earthquake
plot(suite, ns=nsample)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

