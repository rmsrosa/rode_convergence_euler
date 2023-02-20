# ### An illustrative sample path

plt, plt_noise, = plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Nsample; info)
nothing # hide

# 

plt_noise

# 

plt

# ### An illustrative ensemble of solutions

# ### Order of convergence

# With everything set up, we compute the errors:

@time deltas, errors, trajerrors, lc, p = calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
nothing # hide

# 
# 
# The computed strong errors are stored in `errors`, and a raw LaTeX table can be displayed for inclusion in the article:
# 

table = generate_error_table(Ns, deltas, errors, info)

println(table) # hide
nothing # hide

# 
# 
# The calculated order of convergence is given by `p`:

println("Order of convergence `C Δtᵖ` with p = $(round(p, sigdigits=2))")

# 
# 
# ### Plots
# 
# We create a plot with the rate of convergence with the help of `plot_dt_vs_error`. This returns a handle for the plot and a title.

plt, title = plot_dt_vs_error(deltas, errors, lc, p, info)
nothing # hide

# 
# One can use that to plot the figure here:

plot(plt; title)

# While for the article, you plot a figure without the title and use `title` to create the caption for the latex source:

plot(plt)

println(title)

# 

savefig(plt, joinpath(@__DIR__() * "../../../../latex/img/", info.filename)) # hide
nothing # hide

# We can also plot the time-evolution of the strong errors along the time mesh, just for the sake of illustration:

plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
