# # Linear homogeneous RODE

# This checks the order of convergence of the Euler method for the linear ODE
# $$
# \begin{cases}
# dXₜ/dt = Wₜ Xₜ, \\
# X₀ ∼ 𝒩(0, 1)
# \end{cases}
# $$

using Plots
using Random

include("utils.jl")

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
X0 = randn
f(x, y) = y * x
noise! = Wiener_noise(t0, tf, 0.0)

solution! = function (rng, Xt, t0, tf, x0, f, Yt)
    Nmax = length(Yt)
    dt = (tf - t0) / (Nmax - 1)
    Xt[1] = x0
    It = 0.0
    for n in 2:Nmax
        It += (Yt[n] + Yt[n-1]) * dt / 2 + randn(rng) * sqrt(dt^3) / 12
        Xt[n] = x0 * exp(It)
    end
end

Nmax = 2^18
Ns = 2 .^ (4:10)
M = 1_000

@time deltas, errors, trajerrors, lc, p = get_errors(rng, t0, tf, X0, f, noise!, solution!, Nmax, Ns, M)

#= plot(range(t0, tf, length=Nmax), Yt, label="noise sample path")
plt = plot(range(t0, tf, length=Nmax), Xt, label="solution sample path")
plot!(plt, range(t0, tf, length=last(Ns)), XNt, label="approximate sample path")
display(plt) =#

table = table_errors(Ns, deltas, errors)

println(table)

#include("utils.jl")
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)
filename = "order_linearhomogenous.png"
plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(deltas, trajerrors, t0, tf)

#
using BenchmarkTools

nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Nmax, Ns)

@btime get_errors!($rng, $Yt, $Xt, $XNt, $X0, $f, $noise!, $solution!, $trajerrors, $M, $t0, $tf, $Ns, $nsteps, $deltas)

@profview get_errors!(rng, Yt, Xt, XNt, X0, f, noise!, solution!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

@profview_allocs get_errors!(rng, Yt, Xt, XNt, X0, f, noise!, solution!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

x0 = 1.0

@btime solution_by_euler!($rng, $XNt, $t0, $tf, $x0, $f, $(@view(Yt[1:first(nsteps):1+first(nsteps)*(first(Ns)-1)])))

@btime $noise!($rng, $Yt)

@btime $solution!($rng, $Xt, $t0, $tf, $x0, $f, $Yt)

@btime solution_by_euler!($rng, $XNt, $t0, $tf, $x0, $f, $(@view(Yt[1:first(nsteps):1+first(nsteps)*(first(Ns)-1)])))

function eee(Xt, XNt, M, i, N, nstep, trajerrors)
    for n in 2:N
        trajerrors[n, i] += abs(XNt[n] - Xt[1 + (n-1) * nstep])
    end
    trajerrors ./= M
    nothing
end

@btime $X0($rng)

@btime eee($Xt, $XNt, $M, $1, $(first(Ns)), $(first(nsteps)),$trajerrors)