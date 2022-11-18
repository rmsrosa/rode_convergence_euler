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

deltas, Ns, errors, lc, p = get_errors(rng, t0, tf, Nmax, npowers, M)

table = table_errors(Ns, deltas, errors)

println(table)

plot_error(deltas, errors, lc, p, t0, tf, M, filename = "order_linearhomogenous.png")

error_evolution(deltas, trajerrors, t0, tf)
