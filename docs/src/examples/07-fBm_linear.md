```@meta
EditURL = "https://github.com/rmsrosa/rode_conv_em/docs/literate/examples/07-fBm_linear.jl"
```

# Linear equation with fractional Brownian motion



## The equation

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 07-fBm_linear
using Plots
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
````

Then we set up some variables

````@example 07-fBm_linear
rng = Xoshiro(123)

f(t, x, y) = - x + y

t0 = 0.0
tf = 1.0

x0law = Normal()

ntgt = 2^16
ns = 2 .^ (4:8)
nsample = ns[[1, 2, 3, 4]]
m = 1_000

y0 = 0.0
hursts = 0.1:0.1:0.9
noise = FractionalBrownianMotionProcess(t0, tf, y0, first(hursts), ntgt)
````

And add some information about the simulation:

````@example 07-fBm_linear
info = (
    equation = "linear equation",
    noise = "fBm noise",
    ic = "\$X_0 \\sim \\mathcal{N}(\\mathbf{0}, I\\_2)\$"
)
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 07-fBm_linear
target = RandomEuler()
method = RandomEuler()
````

### Order of convergence

With all the parameters set up, we build the convergence suite:

````@example 07-fBm_linear
allctd = @allocated suite = ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)

pwr = Int(div(round(log10(allctd)), 3)) # approximate since Kb = 1024 bytes not 1000 and so on
@info "`suite` memory: $(round(allctd / 10^(3pwr), digits=2)) $(("bytes", "Kb", "Mb", "Gb", "Tb")[pwr+1])"
````

Then we are ready to compute the errors:

````@example 07-fBm_linear
@time result = solve(rng, suite)
````

The computed strong error for each resolution in `ns` is stored in `result.errors`, and a raw LaTeX table can be displayed for inclusion in the article:

````@example 07-fBm_linear
table = generate_error_table(result, info)

println(table) # hide
nothing # hide
````

The calculated order of convergence is given by `result.p`:

````@example 07-fBm_linear
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2))")
````

### Plots

We create a plot with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 07-fBm_linear
plot(result)
````

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide

For the sake of illustration, we plot a sample of an approximation of a target solution:

````@example 07-fBm_linear
plot(suite, ns=nsample)
````

We can also visualize the noise:

````@example 07-fBm_linear
plot(suite, xshow=false, yshow=true)
````

We save the order of convergence obtained

````@example 07-fBm_linear
ps = [result.p]
````

Now we vary the Hurst parameter and record the corresponding order of convergence.

````@example 07-fBm_linear
@info "h = $(first(hursts)); p = $(result.p)"

for h in Iterators.drop(hursts, 1)
    loc_noise = FractionalBrownianMotionProcess(t0, tf, y0, h, ntgt)
    loc_suite = ConvergenceSuite(t0, tf, x0law, f, loc_noise, target, method, ntgt, ns, m)
    @time loc_result = solve(rng, loc_suite)
    @info "h = $h => p = $(loc_result.p)"
    push!(ps, loc_result.p)
end
````

Strong order $p$ of convergence of the Euler method for $\mathrm{d}X_t/\mathrm{d}t = - X_t + Y_t^H$ with a fractional Brownian motion process $\{Y_t^H\}_t$ for various values of the Hurst parameter $H$ (scattered dots: computed values; dashed line: expected $p = H + 1/2$).

````@example 07-fBm_linear
plt = plot(ylims=(-0.1, 1.1), xaxis="H", yaxis="p", guidefont=10)
scatter!(plt, hursts, ps, label="computed")
plot!(plt, 0.0:0.01:1.0, p -> min(p + 0.5, 1.0), linestyle=:dash, label="expected")

scatter(hursts, ps)
plot!(0.0:0.01:1.0, p -> min(p + 0.5, 1.0), linestyle=:dash)
````

````@example 07-fBm_linear
f(t, x, y) = - y * x

ps = Vector{Float64}()
````

Now we vary the Hurst parameter and record the corresponding order of convergence.

````@example 07-fBm_linear
hursts = 0.05:0.05:0.45

for h in hursts
    loc_noise = FractionalBrownianMotionProcess(t0, tf, y0, h, ntgt)
    loc_suite = ConvergenceSuite(t0, tf, x0law, f, loc_noise, target, method, ntgt, ns, m)
    @time loc_result = solve(rng, loc_suite)
    @info "h = $h => p = $(loc_result.p)"
    push!(ps, loc_result.p)
end
````

Strong order $p$ of convergence of the Euler method for $\mathrm{d}X_t/\mathrm{d}t = - Y_t^H X_t$ with a fractional Brownian motion process $\{Y_t^H\}_t$ for various values of the Hurst parameter $H$ (scattered dots: computed values; dashed line: expected $p = H + 1/2$).

````@example 07-fBm_linear
plt = plot(ylims=(-0.1, 1.1), xaxis="H", yaxis="p", guidefont=10)
scatter!(plt, hursts, ps, label="computed")
plot!(plt, 0.0:0.01:0.5, p -> min(p + 0.5, 1.0), linestyle=:dash, label="expected")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

