using Plots
using Random
rng = Xoshiro(123)

let t0 = 0.0, x0 = 1.0, Nmax = 3_200

    f = (u, t, p, W) -> W * u
    
    for (nfig, (M, tf)) in enumerate(
        (
            (1_000, 1.0),
            (10_000, 1.0),
            (20_000, 1.0),
            (500, 1.0)
        )
    )

        nsteps = (2^n for n in 8:-1:4)
        Ns = (div(Nmax, nstep) for nstep in nsteps)

        Wt = Matrix{Float64}(undef, Nmax, M)
        Wt[1, :] .= 0.0

        dWt = zeros(M)
        dt = tf / (Nmax - 1)
        for n in 2:Nmax
            dWt .= √dt * randn(rng, M)
            Wt[n, :] .= Wt[n-1, :] .+ dWt
        end

        Yt = Matrix{Float64}(undef, Nmax, M)
        Yt[1, :] .= x0
        It = zeros(M)

        for n in 2:Nmax
            # It .+= @view(Wt[n-1, :]) * dt
            It .+= (@view(Wt[n, :]) .+ @view(Wt[n-1,:])) * dt / 2
            Yt[n, :] .= x0 * exp.(It)
        end

        erros = Vector{Float64}(undef, length(Ns))
        deltas = Vector{Float64}(undef, length(Ns))

        for (i, (nstep, N)) in enumerate(zip(nsteps, Ns))
            local tt = range(t0, tf, length = N)
            local dt = Float64(tt.step)
            deltas[i] = dt

            local Xt = Matrix{Float64}(undef, N, M)
            Xt[1, :] .= x0

            for n in 2:N
                Xt[n, :] .= Xt[n-1, :] .* (
                    1 .+ Wt[1 + nstep * (n - 1), :] * dt
                    )
            end

            erros[i] = maximum(sum(abs, Xt - @view(Yt[1:nstep:end, :]), dims = 2)) / M
        end

        lc, p = [one.(deltas) log.(deltas)] \ log.(erros)
        linear_fit = exp(lc) * deltas .^ p

        plt = plot(xscale = :log10, yscale = :log10, xaxis = "Δt", ylims = [0.1, 10.0] .* extrema(erros), yaxis = "erro", title = "Erro forte p = $(round(p, digits=2))\nX₀ = $x0, T = $tf, M = $M", titlefont = 12, legend = :topleft)
        scatter!(plt, deltas, erros, marker = :star, label = "erro forte $M amostras")
        plot!(plt, deltas, linear_fit, linestyle = :dash, label = "ajuste linear")
        display(plt)
    end
end