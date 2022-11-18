using Plots
using Random

function prepare_error_computation(Nmax, Ns)

    nsteps = div.(Nmax, Ns)

    deltas = Vector{Float64}(undef, length(Ns))
    trajerrors = zeros(last(Ns), length(Ns))

    Wt = Vector{Float64}(undef, Nmax)
    Yt = Vector{Float64}(undef, Nmax)
    Xt = Vector{Float64}(undef, last(Ns))

    return nsteps, deltas, trajerrors, Wt, Yt, Xt
end

function get_errors!(rng, Wt, Yt, Xt, trajerrors, M, t0, tf, Ns, nsteps, deltas, Nmax)
    for _ in 1:M
        # draw initial condition
        x0 = randn(rng)

        # time step for noise and exact solutions
        dt = tf / (Nmax - 1)

        # get noise sample path
        Wt[1] = 0.0
        for n in 2:Nmax
            Wt[n] = Wt[n-1] + √dt * randn(rng)
        end

        # get exact pathwise solution
        Yt[1] = x0
        It = 0.0
        for n in 2:Nmax
            It += (Wt[n] + Wt[n-1]) * dt / 2 + randn(rng) * sqrt(dt^3) / 12
            Yt[n] = x0 * exp(It)
        end

        # solve approximate solutions at selected time steps
        for (i, (nstep, N)) in enumerate(zip(nsteps, Ns))

            dt = (tf - t0) / (N - 1)
            deltas[i] = dt

            Xt[1] = x0

            for n in 2:N
                Xt[n] = Xt[n-1] .* (
                    1 + Wt[1 + nstep * (n - 1)] * dt
                )
                trajerrors[n, i] += abs(Xt[n] - Yt[1 + (n-1) * nstep])
            end
        end
    end

    # normalize errors
    trajerrors ./= M
    nothing
end

function get_errors(rng, t0, tf, Nmax, Ns, M)
    nsteps, deltas, trajerrors, Wt, Yt, Xt = prepare_error_computation(Nmax, Ns)

    get_errors!(rng, Wt, Yt, Xt, trajerrors, M, t0, tf, Ns, nsteps, deltas, Nmax)

    errors = maximum(trajerrors, dims=1)[1,:]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, errors, trajerrors, lc, p
end

function table_errors(Ns, deltas, errors)
    table = "N & dt & error \\\\\n"
    for (N, dt, error) in zip(Ns, round.(deltas, sigdigits=3), round.(errors, sigdigits=3))
        table *= "$N & $dt & $error \\\\\n"
    end
    return table
end

function plot_dt_vs_error(deltas, errors, lc, p, M; info = nothing, filename=nothing)
    title = info === nothing ? "" : "Order of convergence of the strong error of the Euler method for\n$(info.equation), with $(info.ic), on $(info.tspan)"
    fit = exp(lc) * deltas .^ p
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "\$\\Delta t\$", ylims = [0.1, 10.0] .* extrema(errors), yaxis = "error", title = title, titlefont = 10, legend = :topleft)
    scatter!(plt, deltas, errors, marker = :star, label = "strong errors with $M samples")
    plot!(plt, deltas, fit, linestyle = :dash, label = "\$C\\Delta t^p\$ fit with p = $(round(p, digits=2))")
    display(plt)
    filename === nothing || savefig(plt, @__DIR__() * "/img/$filename")
end

function plot_t_vs_errors(deltas, trajerrors, t0, tf, filename=nothing)
    plt = plot(title = "Evolution of the strong error", titlefont=12, legend=:topleft)
    for (i, N) in enumerate(Ns)
        plot!(range(t0, tf, length=N), trajerrors[1:N, i], label="\$\\mathrm{d}t = $(round(deltas[i], sigdigits=2))\$")
    end
    display(plt)
    filename === nothing || savefig(plt, @__DIR__() * "/img/$filename")
end
