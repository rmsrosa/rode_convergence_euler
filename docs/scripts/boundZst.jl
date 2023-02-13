using Plots

N = 1000
η = zeros(N)
integrand = zeros(N)
integrandbound = zeros(N)
αs = 0.025:0.05:0.475
τs = (0.01, 0.1, 0.5, 1.0, 2.0)
ξfractions = (1/10, 1/2, 1)
# ξfractions = (1,)
δmax = 4.0
δslength = 500
δs = (range(δmax/δslength, δmax, length=δslength) ./ δmax) .^ 6
si = zero(δs)
sb = zero(δs)

function check_bounds!(integrand, integrandbound, η, N, αs, τs, ξfractions, δs, si, sb)
    m = 0.0
    for α = αs
        for τ in τs
            for ξfraction in ξfractions
                ξ = ξfraction * τ
                for (i, δ) in enumerate(δs)
                    s = τ + δ
                    dη = ξ / (N + 1)
                    η .= range(dη, ξ - dη, length=N)
                    integrand .= η .^ (2α) .* ((τ .- η) .^ (-α) .- (s .- η) .^ (-α)) .^ 2
                    integrandbound .= δ ^ (2α) .* (s .- η) .^ (-2α) .* (τ .- η) .^ (-2α)
                    integrandbound .= η .^ (2α) .* ((τ .- η) .^ (-2α) .- (s .- η) .^ (-2α))
                    si[i] = √(sum(integrand) * dη)
                end
                # sb .= ( ξ / τ ) ^ α * δs .^ α
                # sb .=  δs .^ α
                sb .= ξ ^ α * δs .^ (1/2 - α) ./ √(1 - 2α)
                m = max(m, maximum(si ./ sb))
                all(x < y for (x, y) in zip(si, sb)) || @warn "ops, bound failed at α = $α, τ = $τ, ξ = $ξ"
                #= plt = plot(title="error with α = $α, τ = $τ, ξ = $ξ", titlefont = 10, xlabel="δ", ylabel="error")
                plot!(plt, δs, si, label="computed")
                plot!(plt, δs, sb, label="bound") =#
                plt = plot(title="ratios α = $α, τ = $τ, ξ = $ξ", titlefont = 10, xlabel="δ", ylabel="ratio")
                plt = plot!(plt, δs, si ./ sb, legend=false)
                display(plt)
            end
        end
    end
    return m
end

@time check_bounds!(integrand, integrandbound, η, N, αs, τs, ξfractions, δs, si, sb)

check_bounds!(integrand, integrandbound, η, N, 0.25:0.25, 1.0:1.0, ξfractions, δs, si, sb)