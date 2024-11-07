# # Combined plot
#
# Here we just read the results of some examples to build a combined plot for the companion article.

using JLD2
using Plots
using Measures
using Distributions
using RODEConvergence

# Read previously saved convergence data.

results_popdyn = load(joinpath(@__DIR__(), "results/06-popdyn_result.jld2"), "result")
results_earthquake = load(joinpath(@__DIR__(), "results/08-earthquake_result.jld2"), "result")
nothing

# Draw combined plot

plt_popdyn = plot(results_popdyn)
plt_earthquake = plot(results_earthquake)

plt_combined = plot(plt_popdyn, plt_earthquake, legendfont=6, size=(800, 240), title=["(a) population dynamics" "(b) seismic model"], titlefont=10, bottom_margin=5mm, left_margin=5mm)

# Save it

savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "combined_orders.pdf"))
nothing # hide
