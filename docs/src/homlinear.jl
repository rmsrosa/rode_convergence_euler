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
Nmax = 2^16
npowers = 12:-1:5
M = 1_000

deltas, Ns, errors, trajerrors, lc, p = get_errors(rng, t0, tf, Nmax, npowers, M)

table = table_errors(Ns, deltas, errors)

println(table)

include("utils.jl")
info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$",
    tspan="\$[0, T] = [$t0, $tf]\$"
)
filename = "order_linearhomogenous.png"
plot_error(deltas, errors, lc, p, M; info, filename)

error_evolution(deltas, trajerrors, t0, tf)
