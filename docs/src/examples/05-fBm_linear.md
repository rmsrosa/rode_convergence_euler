```@meta
EditURL = "../../literate/examples/05-fBm_linear.jl"
```

# Linear RODE with fractional Brownian motion

## The equation

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

f(t, x, y) = - x + y

t0 = 0.0
tf = 1.0

x0law = Normal()

ntgt = 2^18
ns = 2 .^ (6:9)
nsample = ns[[1, 2, 3, 4]]
m = 200

y0 = 0.0
hursts = Iterators.flatten((0.1:0.1:0.5, 0.7:0.2:0.9))
noise = FractionalBrownianMotionProcess(t0, tf, y0, first(hursts), ntgt)
````

And add some information about the simulation:

````@example 05-fBm_linear
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = -X_t + B^H_t\$",
    noise = "fBm noise",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 05-fBm_linear
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the convergence suite:

````@example 05-fBm_linear
allctd = @allocated suite = ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)

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
table = generate_error_table(result, info)

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
    loc_suite = ConvergenceSuite(t0, tf, x0law, f, loc_noise, target, method, ntgt, ns, m)
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
plt = plot(ylims=(-0.1, 1.2), xaxis="H", yaxis="p", guidefont=10)
scatter!(plt, collect(hursts), ps, yerror=(ps .- pmins, pmaxs .- ps), label="computed")
plot!(plt, 0.0:0.01:1.0, p -> min(p + 0.5, 1.0), linestyle=:dash, label="expected")
````

Strong order $p$ of convergence of the Euler method for $\mathrm{d}X_t/\mathrm{d}t = - Y_t^H X_t$ with a fractional Brownian motion process $\{Y_t^H\}_t$ for various values of the Hurst parameter $H$ (scattered dots: computed values; dashed line: expected $p = H + 1/2$).

````@example 05-fBm_linear
savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", "order_dep_on_H_fBm.png")) # hide
nothing # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

