# # Linear homogeneous RODE with sin of fBm noise

using Revise
using Plots
using Random
using Distributions
using RODEConvergence
using BenchmarkTools

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
f(t, x, y) = sin(y) * x
H = 0.1
y0 = 0.0

Ntgt = 2^18
Ns = 2 .^ (4:9)
M = 1_000

noise! = fBm_noise(t0, tf, y0, H, Ntgt)
target! = solve_euler!

info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(Y_t) X_t\$",
    noise = "an fBm noise with Hurst parameter \$H = $H\$ and \$Y_0 = 0.0\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)

@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)

table = generate_error_table(Ns, deltas, errors)

println(table)

#include("utils.jl")

filename = @__DIR__() * "/img/order_linearhomogenoussinfBm_H=$H.png"
plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)

filename = @__DIR__() * "/img/linearhomogenoussinfBm_H=$H_sample.png"
plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info, filename)

nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

@btime calculate_errors!($rng, $Yt, $Xt, $XNt, $X0, $f, $noise!, $target!, $trajerrors, $M, $t0, $tf, $Ns, $nsteps, $deltas)
