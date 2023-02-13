
"""
    plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info = nothing, filename=nothing)

Plot a few sample paths in the interval `t0` to `tf`, with different time steps as given by a list/vector/range `Ns`, from the noise `noise!`, using the Euler method for the equation with right hand side `f`.
"""
function plot_sample_approximations(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns; info = nothing, filename=nothing)

    title = info === nothing ? "" : "Sample noise, sample target and numerical approximations for\n$(info.equation), with $(info.ic), on $(info.tspan)\nand $(info.noise)"

    # generate noise sample path
    Yt = Vector{Float64}(undef, Ntgt)
    noise!(rng, Yt)
    plt_noise = plot(range(t0, tf, length=Ntgt), Yt, title="noise sample path", titlefont = 10, legend=nothing)

    # generate target path
    x0 = rand(rng, X0)
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

        plot!(plt, range(t0, tf, length=N), XNt, linestyle=:dash, label="\$N = $N\$")
    end

    filename === nothing || savefig(plt, filename)
    return plt, plt_noise, Yt, Xt, XNts
end

"""
    generate_error_table(Ns, deltas, errors)

Generate the markdown table with the data for the strong `errors` obtained with time steps `deltas` and length `Ns`.
"""
function generate_error_table(Ns, deltas, errors)
    table = "N & dt & error \\\\\n"
    for (N, dt, error) in zip(Ns, round.(deltas, sigdigits=3), round.(errors, sigdigits=3))
        table *= "$N & $dt & $error \\\\\n"
    end
    return table
end

function plot_dt_vs_error(deltas, errors, lc, p, M; info = nothing, filename=nothing)
    title = info === nothing ? "" : "Order of convergence of the strong error of the Euler method for\n$(info.equation), with $(info.ic), on $(info.tspan)\nand $(info.noise)"
    fit = exp(lc) * deltas .^ p
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "\$\\Delta t\$", xlims = [0.5, 2.0] .* extrema(deltas), ylims = [0.5, 2.0] .* extrema(errors), yaxis = "error", title = title, titlefont = 10, legend = :topleft)
    scatter!(plt, deltas, errors, marker = :star, label = "strong errors with $M samples")
    plot!(plt, deltas, fit, linestyle = :dash, label = "\$C\\Delta t^p\$ fit with p = $(round(p, digits=2))")
    filename === nothing || savefig(plt, filename)
    return plt
end

function plot_t_vs_errors(Ns, deltas, trajerrors, t0, tf, filename=nothing)
    plt = plot(title = "Evolution in time of the strong error of the Euler method\nfor each chosen fixed time step", xlabel="\$t\$", ylabel="error", titlefont=10, legend=:topleft)
    for (i, N) in enumerate(Ns)
        plot!(range(t0, tf, length=N), trajerrors[1:N, i], label="\$\\mathrm{d}t = $(round(deltas[i], sigdigits=2))\$")
    end
    filename === nothing || savefig(plt, filename)
    return plt
end
