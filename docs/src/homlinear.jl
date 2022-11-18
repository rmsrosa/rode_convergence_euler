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
Nmax = 2^18
Ns = 2 .^ (4:10)
M = 1_000

@time deltas, errors, trajerrors, lc, p = get_errors(rng, t0, tf, Nmax, Ns, M)

table = table_errors(Ns, deltas, errors)

println(table)

include("utils.jl")
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)
filename = "order_linearhomogenous.png"
plot_dt_vs_error(deltas, errors, lc, p, M; info, filename)

plot_t_vs_errors(deltas, trajerrors, t0, tf)
