```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/examples/07-earthquake.jl"
```

# Earthquake model

Now we consider a mechanical structure problem under ground-shaking excitations, motivated by Earthquake models.


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
noise1 = WienerProcess(t0, tf, y0)

ylaw = product_distribution(Uniform(0.0, 1.0), Uniform(0.0, 1.0), Uniform(1.0, 5.0), Exponential())
nr = 5
g(t, r) = mapreduce(ri -> ri[1] * max(0.0, t - ri[4]) ^ ri[2] * exp(-ri[3] * max(0.0, t - ri[4])), +, eachcol(r))
noise2 = TransportProcess(t0, tf, ylaw, g, nr)

noise = ProductProcess(noise1, noise2)

yt = Vector{Float64}(undef, 2^8)

rand!(rng, noise2, yt)

plot(range(t0, tf, length=200), t -> g(t, noise2.rv))
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
    noise = "Orstein-Uhlenbeck and Transport Process",
    ic = "\$X_0 \\sim \\mathcal{N}(\\mathbf{0}, I\\_2)\$"
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

We can also visualize the noise separate

````@example 07-earthquake
plot(suite, xshow=false, yshow=2)
````

or combined

````@example 07-earthquake
plot!(suite, xshow=false, yshow=prod)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

