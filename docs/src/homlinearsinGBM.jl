# # Linear homogeneous RODE with sin of GBM noise

using Plots
using Random
using Distributions

include("utils.jl")

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
f(x, y) = sin(y) * x
μ = 1.0
σ = 0.2
y0 = 1.0
noise! = GBM_noise(t0, tf, y0, μ, σ)
target! = solve_euler!

Ntgt = 2^20
Ns = 2 .^ (4:10)
M = 1_000

info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(Y_t) X_t\$",
    noise = "a GBM noise with drift \$\\mu = $μ\$, diffusion \$\\sigma = $σ\$, and \$Y_0 = 1.0\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)

@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)

table = generate_error_table(Ns, deltas, errors)

println(table)

#include("utils.jl")

filename = "order_linearhomogenoussinGBM.png"
plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(deltas, trajerrors, t0, tf)

filename = "linearhomogenoussinGBM_sample.png"
plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info, filename)


using BenchmarkTools

nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

@btime calculate_errors!($rng, $Yt, $Xt, $XNt, $X0, $f, $noise!, $target!, $trajerrors, $M, $t0, $tf, $Ns, $nsteps, $deltas)
