using Plots
using Random

function prepare_error_computation(Nmax, npowers)

    nsteps = collect(2^n for n in npowers)
    Ns = collect(div(Nmax, nstep) for nstep in nsteps)

    trajerrors = zeros(last(Ns), length(Ns))
    deltas = Vector{Float64}(undef, length(Ns))

    Wt = Vector{Float64}(undef, Nmax)
    Yt = Vector{Float64}(undef, Nmax)
    Xt = Vector{Float64}(undef, last(Ns))

    return nsteps, Ns, deltas, trajerrors, Wt, Yt, Xt
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

function get_errors(rng, t0, tf, Nmax, npowers, M)
    nsteps, Ns, deltas, trajerrors, Wt, Yt, Xt = prepare_error_computation(Nmax, npowers)

    get_errors!(rng, Wt, Yt, Xt, trajerrors, M, t0, tf, Ns, nsteps, deltas, Nmax)

    errors = maximum(trajerrors, dims=1)[1,:]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, Ns, errors, lc, p
end

function table_errors(Ns, deltas, errors)
    table = "N & dt & error \\\\\n"
    for (N, dt, error) in zip(Ns, round.(deltas, sigdigits=3), round.(errors, sigdigits=3))
        table *= "$N & $dt & $error \\\\\n"
    end
    return table
end

function plot_error(deltas, errors, lc, p, t0, tf, M; filename=nothing)
    fit = exp(lc) * deltas .^ p
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "Δt", ylims = [0.1, 10.0] .* extrema(errors), yaxis = "error", title = "Strong error p = $(round(p, digits=2)) with $M samples\nof the Euler method for \$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$\n\$X_0 \\sim \\mathcal{N}(0, 1)\$, on \$[0, T] = [$t0, $tf]\$", titlefont = 12, legend = :topleft)
    scatter!(plt, deltas, errors, marker = :star, label = "strong errors")
    plot!(plt, deltas, fit, linestyle = :dash, label = "\$C\\Delta t^p\$ fit p = $(round(p, digits=2))")
    display(plt)
    filename === nothing || savefig(plt, @__DIR__() * "/img/$filename")
end

function error_evolution(deltas, trajerrors, t0, tf, filename=nothing)
    plt = plot(title = "Evolution of the strong error", titlefont=12, legend=:topleft)
    for (i, N) in enumerate(Ns)
        plot!(range(t0, tf, length=N), trajerrors[1:N, i], label="\$\\mathrm{d}t = $(round(deltas[i], sigdigits=2))\$")
    end
    display(plt)
    filename === nothing || savefig(plt, @__DIR__() * "/img/$filename")
end
