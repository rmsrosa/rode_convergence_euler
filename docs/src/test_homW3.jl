# This solves dX_t/dt = W_t X_t

using Plots
using Random
rng = Xoshiro(123)

t0 = 0.0
tf = 2.0
Nmax = 2^16
M = 1_000

nsteps = collect(2^n for n in 8:-1:3)
Ns = collect(div(Nmax, nstep) for nstep in nsteps)

errors = zeros(length(Ns))
deltas = Vector{Float64}(undef, length(Ns))

for m in 1:M
    x0 = randn()

    Wt = Vector{Float64}(undef, Nmax)
    Wt[1] = 0.0

    dt = tf / (Nmax - 1)
    for n in 2:Nmax
        Wt[n] = Wt[n-1] + √dt * randn(rng)
    end

    Yt = Vector{Float64}(undef, Nmax)
    Yt[1] = x0
    It = 0.0

    for n in 2:Nmax
        # It += Wt[n-1] * dt
        It += (Wt[n] + Wt[n-1]) * dt / 2 + randn() * sqrt(dt^3) / 12
        Yt[n] = x0 * exp(It)
    end

    for (i, (nstep, N)) in enumerate(zip(nsteps, Ns))
        tt = range(t0, tf, length = N)
        dt = Float64(tt.step)
        deltas[i] = dt
        trajerrors = zeros(length(Ns), N)

        Xt = Vector{Float64}(undef, N)
        Xt[1] = x0

        for n in 2:N
            Xt[n] = Xt[n-1] .* (
                1 + Wt[1 + nstep * (n - 1)] * dt
            )
        end

        errors[i] += maximum(abs, Xt - @view(Yt[1:nstep:end]))
    end
end

errors ./= M

lc, p = [one.(deltas) log.(deltas)] \ log.(errors)
linear_fit = exp(lc) * deltas .^ p

begin
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "Δt", ylims = [0.1, 10.0] .* extrema(errors), yaxis = "erro", title = "Strong error p = $(round(p, digits=2)) for \$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$\n\$X_0 \\sim \\mathcal{N}(0, 1)\$, T = $tf, M = $M", titlefont = 12, legend = :topleft)
    scatter!(plt, deltas, errors, marker = :star, label = "strong error $M samples")
    plot!(plt, deltas, linear_fit, linestyle = :dash, label = "linear fit")
    display(plt)
end
