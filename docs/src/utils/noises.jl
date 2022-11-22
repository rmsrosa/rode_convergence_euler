
function Wiener_noise(t0, tf, y0::T) where {T}
    fn = function (rng::AbstractRNG, Yt::Vector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        sqrtdt = sqrt(dt)
        Yt[1] = y0
        for n in 2:N
            Yt[n] = Yt[n-1] + sqrtdt * randn(rng)
        end
    end
    return fn
end

function GBM_noise(t0, tf, μ, σ, y0::T) where {T}
    fn = function (rng::AbstractRNG, Yt::Vector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        sqrtdt = sqrt(dt)
        a = (μ + σ^2/2)
        Yt[1] = y0
        for n in 2:N
            Yt[n] = Yt[n-1] * exp(a * dt + σ * sqrtdt * randn(rng))
        end
    end
    return fn
end

function CompoundPoisson_noise(t0, tf, λ, R)
    fn = function (rng, Yt::Vector)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        RV = Poisson(λ * dt)
        Yt[1] = 0.0
        for n in 2:N
            Ni = rand(rng, RV)
            Yt[n] = Yt[n-1]
            for _ in 1:Ni
                Yt[n] += rand(rng, R)
            end
        end
    end
end

function CompoundPoisson_noise_alt(t0, tf, λ, R)
    fn = function (rng, Yt::Vector)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        Yt[1] = zero(λ)
        i = 1
        while i < N
            i += 1
            Yt[i] = Yt[i-1]
            r = - log(rand(rng)) / λ
            while r < dt
                Yt[i] += rand(rng, R)
                r += -log(rand(rng)) / λ
            end
        end
    end
end
