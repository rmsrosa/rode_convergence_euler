```@meta
EditURL = "../../literate/examples/05-fBm_linear.jl"
```

# Linear RODE with fractional Brownian motion

Here, we consider a linear equation driven by a fractional Brownian motion (fBm) noise. This noise has a parameter $H,$ in the range $0 < H < 1,$ which is called the Hurst parameter and controls the correlation between the increments of the process.

When $0 < H < 1/2,$ the increments are negatively correlated, meaning that there is a higher chance of an increment to be opposite to a previous increment, yielding "rougher" paths, while for $1/2 < H < 1,$ the correlation is positive, yielding "smoother" paths. For $H = 1/2,$ the increments are uncorrelated and the fBm is a Wiener process.

For $H \neq 1/2,$ this process is not a Markov process and, in particular, is not a semi-martingale. Thus, the result in the paper does not apply, and the strong convergence is not necessarily 1. A recent result (see [Wang, Cao, Han, & Kloeden (2021)](https://doi.org/10.1007/s11075-020-00967-w)) estimates the order of convergence to be the Hölder exponent of the sample paths, which is exactly $H.$ We, however, show, for this particular linear equation with the fBm in the non-homogeneous term, that the convergence is of order $\min\{H + 1/2, 1\},$ hence higher than the rate $H$ previously known. This rate coincides with the order 1 in the range $1/2 \leq H < 1,$ but is smaller than 1 in the range $0 < H < 1/2,$ which is fine since, as we said above, this is not a semi-martingale for $H$ in this range.

## The equation

More precisely, we consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = - X_t + B^H_t, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where the noise $\{B^H_t\}_t$ is assumed to be a fractional Brownian motion (fBm) with Hurst parameter $0 < H < 1$.

The explicit solution is
```math
  X_t = e^{-t}X_0 + \int_0^t e^{-(t-s)}B^H_s\;\mathrm{d}s,
```

We do not compute numerically this integral solution. Instead, we compare the Euler approximations with another Euler approximation in a much finer mesh.

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 05-fBm_linear
using Plots
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
````

Then we set up some variables:

````@example 05-fBm_linear
rng = Xoshiro(123)

f(t, x, y, p) = - x + y
params = nothing

t0 = 0.0
tf = 1.0

x0law = Normal()

ntgt = 2^18
ns = 2 .^ (6:9)
````

````@example 05-fBm_linear
nsample = ns[[1, 2, 3, 4]]
````

The number of simulations for the Monte Carlo estimate is set to

````@example 05-fBm_linear
m = 200
nothing # hide
````

````@example 05-fBm_linear
y0 = 0.0
hursts = Iterators.flatten((0.1:0.1:0.5, 0.7:0.2:0.9))
noise = FractionalBrownianMotionProcess(t0, tf, y0, first(hursts), ntgt)
````

And add some information about the simulation, for the caption of the convergence figure.

````@example 05-fBm_linear
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + B^H_t\$",
    noise = "fBm noise",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)
nothing # hide
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 05-fBm_linear
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the convergence suite:

````@example 05-fBm_linear
allctd = @allocated suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)
````

````@example 05-fBm_linear
pwr = Int(div(round(log10(allctd)), 3)) # approximate since Kb = 1024 bytes not 1000 and so on
@info "`suite` memory: $(round(allctd / 10^(3pwr), digits=2)) $(("bytes", "Kb", "Mb", "Gb", "Tb")[pwr+1])"
````

Then we are ready to compute the errors:

````@example 05-fBm_linear
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 05-fBm_linear
table = generate_error_table(result, suite, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 05-fBm_linear
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide
````

### Plots

We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 05-fBm_linear
plot(result)
````

For the sake of illustration, we plot a sample of an approximation of a target solution:

````@example 05-fBm_linear
plot(suite, ns=nsample)
````

We can also visualize the noise:

````@example 05-fBm_linear
plot(suite, xshow=false, yshow=true)
````

We save the order of convergence obtained

````@example 05-fBm_linear
ps = [result.p]
pmins = [result.pmin]
pmaxs = [result.pmax]
````

nothing

Now we vary the Hurst parameter and record the corresponding order of convergence.

````@example 05-fBm_linear
@info "h = $(first(hursts)); p = $(result.p)"

for h in Iterators.drop(hursts, 1)
    loc_noise = FractionalBrownianMotionProcess(t0, tf, y0, h, ntgt)
    loc_suite = ConvergenceSuite(t0, tf, x0law, f, loc_noise, params, target, method, ntgt, ns, m)
    @time loc_result = solve(rng, loc_suite)
    @info "h = $h => p = $(loc_result.p) ($(loc_result.pmin), $(loc_result.pmax))"
    push!(ps, loc_result.p)
    push!(pmins, loc_result.pmin)
    push!(pmaxs, loc_result.pmax)
end
````

We print them out for inclusing in the paper:

````@example 05-fBm_linear
[collect(hursts) ps pmins pmaxs]
````

The following plot helps visualizing the result.

````@example 05-fBm_linear
plt = plot(ylims=(-0.1, 1.2), xaxis="H", yaxis="p", guidefont=10, legend=:bottomright)
scatter!(plt, collect(hursts), ps, yerror=(ps .- pmins, pmaxs .- ps), label="computed")
plot!(plt, 0.0:0.5:1.0, p -> min(p + 0.5, 1.0), linestyle=:dash, label="current")
plot!(plt, 0.0:0.5:1.0, p -> p, linestyle=:dot, label="previous")
````

Strong order $p$ of convergence of the Euler method for $\mathrm{d}X_t/\mathrm{d}t = - Y_t^H X_t$ with a fBm process $\{Y_t^H\}_t$ for various values of the Hurst parameter $H$ (scattered dots: computed values; dashed line: expected $p = H + 1/2;$ dash-dot line: previous theory $p = H.$).

````@example 05-fBm_linear
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "order_dep_on_H_fBm.pdf")) # hide
nothing # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

