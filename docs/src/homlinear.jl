# # Linear homogeneous RODE

# This checks the order of convergence of the Euler method for the linear ODE
# $$
# \begin{cases}
# dXₜ/dt = Wₜ Xₜ, \\
# X₀ ∼ 𝒩(0, 1)
# \end{cases}
# $$

using Plots
using Random
using Statistics
rng = Xoshiro(123)

t0 = 0.0
tf = 1.0
Nmax = 2^16
M = 1_000

nsteps = collect(2^n for n in 12:-1:5)
Ns = collect(div(Nmax, nstep) for nstep in nsteps)

trajerrors = zeros(last(Ns), length(Ns))
errors = zeros(length(Ns))
deltas = Vector{Float64}(undef, length(Ns))

Wt = Vector{Float64}(undef, Nmax)
Yt = Vector{Float64}(undef, Nmax)
Xt = Vector{Float64}(undef, last(Ns))

function get_errors!(rng, Wt, Yt, Xt, trajerrors, M, t0, tf, Ns, nsteps, deltas, Nmax)
    for _ in 1:M
        x0 = randn(rng)

        Wt[1] = 0.0

        Yt[1] = x0
        It = 0.0

        dt = tf / (Nmax - 1)

        for n in 2:Nmax
            Wt[n] = Wt[n-1] + √dt * randn(rng)
            It += (Wt[n] + Wt[n-1]) * dt / 2 + randn() * sqrt(dt^3) / 12
            Yt[n] = x0 * exp(It)
        end

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
end

@time get_errors!(rng, Wt, Yt, Xt, trajerrors, M, t0, tf, Ns, nsteps, deltas, Nmax)

@time for m in 1:M
    x0 = randn(rng)

    Wt[1] = 0.0

    Yt[1] = x0
    It = 0.0

    dt = tf / (Nmax - 1)

    for n in 2:Nmax
        Wt[n] = Wt[n-1] + √dt * randn(rng)
        It += (Wt[n] + Wt[n-1]) * dt / 2 + randn() * sqrt(dt^3) / 12
        Yt[n] = x0 * exp(It)
    end

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

trajerrors ./= M

errors = [mean(@view(trajerrors[1:N, i])) for (i, N) in enumerate(Ns)]
# errors ./= M

lc, p = [one.(deltas) log.(deltas)] \ log.(errors)
linear_fit = exp(lc) * deltas .^ p

begin
    plt = plot(xscale = :log10, yscale = :log10, xaxis = "Δt", ylims = [0.1, 10.0] .* extrema(errors), yaxis = "error", title = "Strong error p = $(round(p, digits=2)) with $M samples\nof the Euler method for \$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$\n\$X_0 \\sim \\mathcal{N}(0, 1)\$, on \$[0, T] = [$t0, $tf]\$", titlefont = 12, legend = :topleft)
    scatter!(plt, deltas, errors, marker = :star, label = "strong errors")
    plot!(plt, deltas, linear_fit, linestyle = :dash, label = "linear fit p = $(round(p, digits=2))")
    display(plt)
end

begin
    plt = plot(title = "Evolution of the strong error", titlefont=12, legend=:topleft)
    for (i, N) in enumerate(Ns)
        plot!(range(t0, tf, length=N), trajerrors[1:N, i], label="\$\\mathrm{d}t = $(round(deltas[i], sigdigits=2))\$")
    end
    display(plt)
end
