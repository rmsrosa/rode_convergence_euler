# # Linear homogeneous RODE with Wiener noise

using Revise
using Plots
using Random
using Distributions
using RODEConvergence

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = Normal()
w0 = 0.0
f(t, x, y) = y * x
noise! = Wiener_noise(t0, tf, w0)
target! = function (rng, Xt, t0, tf, x0, f, Yt)
    Ntgt = length(Yt)
    dt = (tf - t0) / (Ntgt - 1)
    Xt[1] = x0
    It = 0.0
    for n in 2:Ntgt
        It += (Yt[n] + Yt[n-1]) * dt / 2 + randn(rng) * sqrt(dt^3) / 12
        Xt[n] = x0 * exp(It)
    end
end

Ntgt = 2^18
Ns = 2 .^ (4:10)
M = 1_000

@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)

table = generate_error_table(Ns, deltas, errors)

println(table)

#include("utils.jl")
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)
filename = @__DIR__() * "/img/order_linearhomogenous.png"

plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)

filename = @__DIR__() * "/img/linearhomogenous_sample.png"
plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info, filename)

using BenchmarkTools

nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

@btime calculate_errors!($rng, $Yt, $Xt, $XNt, $X0, $f, $noise!, $target!, $trajerrors, $M, $t0, $tf, $Ns, $nsteps, $deltas)
