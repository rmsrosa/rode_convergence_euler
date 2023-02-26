
"""
    plot_sample_approximations(rng, t0, tf, X0law, f, noise, target!, Ntgt, Ns; info = nothing, filename=nothing)

Plot a few sample paths in the interval `t0` to `tf`, with different time steps as given by a list/vector/range `Ns`, from the noise `noise!`, using the Euler method for the equation with right hand side `f` and initial condition with distribution law `X0law`, along with the `target!` solution solved on the mesh point with `Ntgt` points.
"""
function plot_sample_approximations(rng, t0, tf, X0law, f, noise, target!, Ntgt, Ns; info = nothing, filename=nothing)

    title = info === nothing ? "" : "Sample noise, sample target and numerical approximations for\n$(info.equation), with $(info.ic), on $(info.tspan)\nand $(info.noise)"

    # generate noise sample path
    Yt = Vector{Float64}(undef, Ntgt)
    rand!(rng, noise, Yt)
    plt_noise = plot(range(t0, tf, length=Ntgt), Yt, color=:black, title="noise sample path", titlefont = 10, legend=nothing)

    # generate target path
    x0 = rand(rng, X0law)
    Xt = Vector{Float64}(undef, Ntgt)
    target!(rng, Xt, t0, tf, x0, f, Yt)

    # solve approximate solutions at selected time steps
    nsteps = div.(Ntgt, Ns)
    deltas = Vector{Float64}()
    XNts = Vector{Vector{Float64}}()
    plt = plot(range(t0, tf, length=Ntgt), Xt, label="target", linewidth = 4, title=title, titlefont=10)

    for (nstep, N) in zip(nsteps, Ns)

        push!(deltas, (tf - t0) / (N - 1))

        XNt = Vector{Float64}(undef, N)
        solve_euler!(rng, XNt, t0, tf, x0, f, view(Yt, 1:nstep:1+nstep*(N-1)))

        push!(XNts, XNt)

        plot!(plt, range(t0, tf, length=N), XNt, linestyle=:auto, color=:black, label="\$N = $N\$")
    end

    filename === nothing || savefig(plt, filename)
    return plt, plt_noise, Yt, Xt, XNts
end

"""
    generate_error_table(Ns, deltas, errors, info)

Generate the markdown table with the data for the strong `errors` obtained with time steps `deltas` and length `Ns` and `info` for the problem, where `info` is given as a namedtuple with fields `info.equation`, `info.ic`, `info.tspan`, `info.noise`, `info.M`, `info.Ntgt`, and `info.Ns`.
"""
function generate_error_table(Ns, deltas, errors, info)
    table = "\\begin{tabular}[htb]{|l|l|l|}\n
    \\hline N & dt & error\\\\\n
    \\hline \\hline\n"
    # table = "N & dt & error \\\\\n"
    for (N, dt, error) in zip(Ns, round.(deltas, sigdigits=3), round.(errors, sigdigits=3))
        table *= "$N & $dt & $error \\\\\n"
    end
    table *= "\\hline \\\\\n
    \\end{tabular}\n
    \\caption{Mesh points (N), time steps (dt), and strong error (error) of the Euler method for $(info.equation), with initial condition $(info.ic) and $(info.noise), on the time interval $(info.tspan), based on \$M = $(info.M)\$ sample paths for each fixed time step, with the target solution calculated with \$2^{$(Int(log2(info.Ntgt)))}=$(info.Ntgt)\$ points.}"
    return table
end

"""
    plot_dt_vs_error(deltas, errors, lc, p; info = nothing, filename=nothing)

Plot the convergence estimate in a log-log scale (tme step vs strong error) based on the given values `deltas`, `errors`, `lc`, `p`, as computed by [`calculate_errors`](@ref), and the number `M` of sample points used in the Monte-Carlo method.

It draws a scatter plot from `deltas` and `errors` and a line plot from the fitted `errors ≈ C Δtᵖ`, where `C = exp(lc)`.

If `info` is given as a namedtuple with fields `info.equation`, `info.ic`, `info.tspan`, `info.noise`, `info.M`, `info.Ntgt`, and `info.Ns`, then a title is included in the plot, with this information.

If `filename` is given, then the picture is save to the given location.
"""
function plot_dt_vs_error(deltas, errors, lc, p, info = nothing)
    title = info === nothing ? "" : "Order of convergence \$p = $(round(p, sigdigits=2))\$ of the strong error of the Euler method for\n$(info.equation), on $(info.tspan), with $(info.ic),\nand $(info.noise),\n computed with $(info.M) sample paths,\nfor each \$\\Delta t = 1/N\$,  \$N = $(join(info.Ns, ", "))\$.\n"
    fit = exp(lc) * deltas .^ p
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "\$\\Delta t\$", xlims = [0.5, 2.0] .* extrema(deltas), ylims = [0.5, 2.0] .* extrema(errors), yaxis = "\$\\mathrm{error}\$", titlefont = 9, legend = :topleft)
    scatter!(plt, deltas, errors, marker=:circle, color=:black, label = "strong errors")
    plot!(plt, deltas, fit, linestyle = :dash, color=:black, label = "\$C\\Delta t^p\$ fit with p = $(round(p, digits=2))")
    return plt, title
end

"""
    plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)

Plot the evolution of the strong error in time `t` within the interval `(t0, tf)`.
"""
function plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf)
    plt = plot(title = "Evolution in time of the strong error of the Euler method\nfor each chosen fixed time step", xlabel="\$t\$", ylabel="error", titlefont=10, legend=:topleft)
    for (i, N) in enumerate(Ns)
        plot!(range(t0, tf, length=N), trajerrors[1:N, i], label="\$\\mathrm{d}t = $(round(deltas[i], sigdigits=2))\$")
    end
    return plt
end
