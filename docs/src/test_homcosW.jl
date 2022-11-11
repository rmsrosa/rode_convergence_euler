# This solves dX_t/dt = -cos(5W_t) X_t as in Grune-Kloeden

using Plots
using Random
rng = Xoshiro(123)

t0 = 0.0
tf = 1.0
Nmax = 2^20
M = 1_000

nsteps = collect(2^n for n in 18:-1:10)
Ns = collect(div(Nmax, nstep) for nstep in nsteps)

errors = zeros(length(Ns))
deltas = Vector{Float64}(undef, length(Ns))

for m in 1:M
    x0 = 1.0 # randn()

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
        It += -cos(5 *Wt[n-1]) * dt 
        # It += -(cos(5 * Wt[n]) + cos(5 *Wt[n-1])) * dt / 2 trapezoidal
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
                1 - cos(5 * Wt[1 + nstep * (n - 1)]) * dt
            )
        end

        errors[i] += maximum(abs, Xt - @view(Yt[1:nstep:end]))
    end
end

errors ./= M

lc, p = [one.(deltas) log.(deltas)] \ log.(errors)
linear_fit = exp(lc) * deltas .^ p

begin
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "Δt", ylims = [0.1, 10.0] .* extrema(errors), yaxis = "erro", title = "Strong error p = $(round(p, digits=2)) for \$\\mathrm{d}X_t/\\mathrm{d}t = -cos(5W_t) X_t\$, with \$X_0 = 1.0\$\n\$[0, T] = \$[$t0, $tf], M = $M\nNmax = $Nmax, Ns = $Ns", titlefont = 10, legend = :topleft)
    scatter!(plt, deltas, errors, marker = :star, label = "strong error $M samples")
    plot!(plt, deltas, linear_fit, linestyle = :dash, label = "linear fit")
    display(plt)
end
