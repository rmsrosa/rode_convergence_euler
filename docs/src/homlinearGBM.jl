# # Linear homogeneous RODE with sin of GBM noise

using Plots
using Random

include("utils.jl")

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = randn
f(x, y) = sin(y) * x
μ = 1.0
σ = 0.2
Y0 = 1.0
noise! = GBM_noise(t0, tf, Y0, μ, σ)
solution! = solution_by_euler!

Nmax = 2^20
Ns = 2 .^ (4:10)
M = 1_000

info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(Y_t) X_t\$",
    noise = "a GBM noise with drift \$\\mu = $μ\$, diffusion \$\\sigma = $σ\$, and \$Y_0 = 1.0\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)

@time deltas, errors, trajerrors, lc, p = get_errors(rng, t0, tf, X0, f, noise!, solution_by_euler!, Nmax, Ns, M)

#= plot(range(t0, tf, length=Nmax), Yt, label="noise sample path")
plt = plot(range(t0, tf, length=Nmax), Xt, label="solution sample path")
plot!(plt, range(t0, tf, length=last(Ns)), XNt, label="approximate sample path")
display(plt) =#

table = table_errors(Ns, deltas, errors)

println(table)

#include("utils.jl")

filename = "order_linearhomogenousGBM.png"
plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(deltas, trajerrors, t0, tf)

using BenchmarkTools

nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Nmax, Ns)

@btime get_errors!($rng, $Yt, $Xt, $XNt, $X0, $f, $noise!, $solution!, $trajerrors, $M, $t0, $tf, $Ns, $nsteps, $deltas)
