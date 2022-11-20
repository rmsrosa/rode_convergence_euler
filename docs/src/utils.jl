using Plots
using Random

function Wiener_noise(t0, tf, Y0)
    fn = (rng, Yt) -> begin
        Yt[begin] = Y0
        dt = (tf - t0) / (length(Yt) - 1)
        sqrtdt = sqrt(dt)
        for n in firstindex(Yt)+1:lastindex(Yt)
            Yt[n] = Yt[n-1] + sqrtdt * randn(rng)
        end
    end
    return fn
end

function GBM_noise(t0, tf, μ, σ,  Y0)
    fn = (rng, Yt) -> begin
        Yt[begin] = Y0
        dt = (tf - t0) / (length(Yt) - 1)
        sqrtdt = sqrt(dt)
        a = (μ + σ^2/2)
        for n in firstindex(Yt)+1:lastindex(Yt)
            Yt[n] = Yt[n-1] * exp(a * dt + σ * sqrtdt * randn(rng))
        end
    end
    return fn
end

function solve_euler!(rng, Xt, t0, tf, x0, f, Yt)
    N = length(Yt)
    dt = (tf - t0) / (N - 1)
    Xt[1] = x0
    for n in 2:N
        Xt[n] = Xt[n-1] + dt * f(Xt[n-1], Yt[n-1])
    end
end

function prepare_variables(Ntgt, Ns)

    nsteps = div.(Ntgt, Ns)

    deltas = Vector{Float64}(undef, length(Ns))
    trajerrors = zeros(last(Ns), length(Ns))

    Yt = Vector{Float64}(undef, Ntgt)
    Xt = Vector{Float64}(undef, Ntgt)
    XNt = Vector{Float64}(undef, last(Ns))

    return nsteps, deltas, trajerrors, Yt, Xt, XNt
end

function calculate_errors!(rng, Yt, Xt, XNt, X0, f::F, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas) where F
    for _ in 1:M
        # draw initial condition
        x0 = X0(rng)

        # generate noise sample path
        noise!(rng, Yt)

        # generate target path
        target!(rng, Xt, t0, tf, x0, f, Yt)

        # solve approximate solutions at selected time steps and update strong errors
        for (i, (nstep, N)) in enumerate(zip(nsteps, Ns))

            deltas[i] = (tf - t0) / (N - 1)

            solve_euler!(rng, XNt, t0, tf, x0, f, view(Yt, 1:nstep:1+nstep*(N-1)))

            for n in 2:N
                trajerrors[n, i] += abs(XNt[n] - Xt[1 + (n-1) * nstep])
            end
        end
    end

    # normalize errors
    trajerrors ./= M
    nothing
end

function calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
    nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

    calculate_errors!(rng, Yt, Xt, XNt, X0, f, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

    errors = maximum(trajerrors, dims=1)[1,:]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, errors, trajerrors, lc, p
end

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
    display(plt)
    filename === nothing || savefig(plt, @__DIR__() * "/img/$filename")
end

function plot_t_vs_errors(deltas, trajerrors, t0, tf, filename=nothing)
    plt = plot(title = "Evolution in time of the strong error of the Euler method\nfor each chosen fixed time step", xlabel="\$t\$", ylabel="error", titlefont=10, legend=:topleft)
    for (i, N) in enumerate(Ns)
        plot!(range(t0, tf, length=N), trajerrors[1:N, i], label="\$\\mathrm{d}t = $(round(deltas[i], sigdigits=2))\$")
    end
    display(plt)
    filename === nothing || savefig(plt, @__DIR__() * "/img/$filename")
end
