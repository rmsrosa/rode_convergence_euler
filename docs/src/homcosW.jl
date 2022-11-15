# This solves dX_t/dt = -cos(5W_t) X_t as in Grune-Kloeden

using Plots
using Random
rng = Xoshiro(123)

t0 = 0.0
tf = 1.0
Nmax = 2^15 # Grune & Kloeden used Nmax = 1_000_000, which is about 2^20 = 1_048_576
M = 10_000 # Grune & Kloeden used M=1 just a single step

nsteps = collect(2^n for n in 9:-1:5)
Ns = collect(div(Nmax, nstep) for nstep in nsteps)

errorsE = zeros(length(Ns)) # errors for Euler
errorsH = zeros(length(Ns)) # errors for Heun
deltas = Vector{Float64}(undef, length(Ns))

Wt = Vector{Float64}(undef, Nmax+1)
Wt[1] = 0.0
Yt = Vector{Float64}(undef, Nmax+1)
# Euler approximation
XEt = [Vector{Float64}(undef, N) for N in Ns]
 # Heun approximation
 XHt = [Vector{Float64}(undef, N)for N in Ns]

@time for m in 1:M
    x0 = 1.0 # randn()

    dt = tf / (Nmax - 1)
    for n in 2:Nmax
        Wt[n] = Wt[n-1] + √dt * randn(rng)
    end

    Yt[1] = x0
    It = 0.0

    for n in 2:Nmax+1
        # It += -cos(5 * Wt[n-1]) * dt 
        It += -(cos(5 * Wt[n]) + cos(5 *Wt[n-1])) * dt / 2 trapezoidal
        Yt[n] = x0 * exp(It)
    end

    for (i, (nstep, N)) in enumerate(zip(nsteps, Ns))
        tt = range(t0, tf, length = N)
        dt = Float64(tt.step)
        deltas[i] = dt

        XEt[i][1] = x0
        XHt[i][1] = x0

        for n in 2:N
            XEt[i][n] = XEt[i][n-1] .* (
                1 - cos(5 * Wt[1 + nstep * (n - 1)]) * dt
            )
            XHaux = XHt[i][n-1] * (
                1 - cos(5 * Wt[1 + nstep * (n - 1)]) * dt
            )
            XHt[i][n] = XHt[i][n-1] + (
                - cos(5 * Wt[1 + nstep * (n - 1)]) * XHt[i][n-1] - cos(5 * Wt[1 + nstep * n]) * XHaux
            ) * dt / 2
        end

        errorsE[i] += maximum(abs, XEt[i] - @view(Yt[1:nstep:end-1]))
        errorsH[i] += maximum(abs, XHt[i] - @view(Yt[1:nstep:end-1]))
    end
    if mod(m, 100) == 0
        @info "$(round(100 * m/M))%"
    end
end

errorsE ./= M
errorsH ./= M

lcE, pE = [one.(deltas) log.(deltas)] \ log.(errorsE)
linear_fitE = exp(lcE) * deltas .^ pE

lcH, pH = [one.(deltas) log.(deltas)] \ log.(errorsH)
linear_fitH = exp(lcH) * deltas .^ pH

begin
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "Δt", ylims = [0.1, 10.0] .* extrema(errorsE), yaxis = "erro", title = "Strong error for \$\\mathrm{d}X_t/\\mathrm{d}t = -\\cos(5W_t) X_t\$, with \$X_0 = 1.0\$\n\$[0, T] = \$[$t0, $tf], M = $M samples\nNmax = $Nmax, Ns = $Ns", titlefont = 10, legend = :topleft)
    scatter!(plt, deltas, errorsE, marker = :star, label = "strong error Euler order p = $(round(pE, digits=2))")
    plot!(plt, deltas, linear_fitE, linestyle = :dash, label = "linear fit Euler")
    scatter!(plt, deltas, errorsH, marker = :star, label = "strong error Heun order p = $(round(pH, digits=2))")
    plot!(plt, deltas, linear_fitH, linestyle = :dash, label = "linear fit Heun")
    display(plt)
end
