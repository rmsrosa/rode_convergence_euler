# # Harvest model with a stepwise Poisson process

using Revise
using Plots
using Random
using Distributions
using RODEConvergence

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
α0 = 15.0
β0 = 2.0
X0 = Beta(α0, β0)
display(plot(0.0:0.01:1.0, pdf.(X0, 0.0:0.01:1.0)))
f(t, x, y) = x - x^2 - y
λ = 25.0 # rate of step changes per unit time
#R = Uniform(0.0, 0.1)
α = 2.0
β = 15.0
R = Beta(α, β)
display(plot(0.0:0.01:1.0, pdf.(R, 0.0:0.01:1.0)))
noise! = StepPoisson_noise(t0, tf, λ, R)
target! = solve_euler!

Ntgt = 2^18
Ns = 2 .^ (4:8)
M = 2_000

# plot(yy, linetype=:steppre)

@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)

table = generate_error_table(Ns, deltas, errors)

println(table)

#include("utils.jl")
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = X_t - X_t^2 - Y_t\$",
    noise = "a stepwise Poisson process noise \$\\{Y_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathrm{Beta}($α0, $β0)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)
filename = @__DIR__() * "/img/order_harvest.png"

plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)

filename = @__DIR__() * "/img/harvest_sample.png"
plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info, filename)

using BenchmarkTools

nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

@btime calculate_errors!($rng, $Yt, $Xt, $XNt, $X0, $f, $noise!, $target!, $trajerrors, $M, $t0, $tf, $Ns, $nsteps, $deltas)
