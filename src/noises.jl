
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

"""
Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and stationary arrivals given by the distribution `U`.

This construction is done by drawing the number of events between consecutive times with interval `dt` by using the Poisson distribution `N(t+dt) - N(t) = Poisson(λdt)`.
"""
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

"""
Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and stationary arrivals given by the distribution `U`.

The construction is done by drawing the interarrival times.
"""
function CompoundPoisson_noise_alt(t0, tf, λ, U)
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
                Yt[i] += rand(rng, U)
                r += -log(rand(rng)) / λ
            end
        end
    end
end

function StepPoisson_noise(t0, tf, λ, R)
    fn = function (rng, Yt::Vector)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        RV = Poisson(λ * dt)
        Yt[1] = 0.0
        for n in 2:N
            Yt[n] = isodd(rand(rng, RV)) ? rand(rng, R) : Yt[n-1]
        end
    end
end

"""
Fractional Brownian motion process
"""
function fBM_noise(t0, tf, y0, N)
    cache=Vector{eltype(y0)}(undef, N)
    fn = function (rng::AbstractRNG, Yt::Vector; cache=cache)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        sqrtdt = sqrt(dt)
        Yt[1] = y0
        cache[1] = y0
        for n in 2:N
            Yt[n] = Yt[n-1] + sqrtdt * randn(rng)
            cache[n] = Yt[n]
        end
    end
    return fn
end

"""
    Generates sample paths of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
"""
function hosking(rng, T, N, H)
    
    # Diecker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) - 2*abs(k)^(2H) + abs(k+1)^(2H))
    
    # Diecker Sec. 2.1.1.
    X = [randn()]
    mu = [gamma(1, H) * X[1]]
    sigsq = [1 - (gamma(1, H)^2)]
    tau = [gamma(1, H)^2]
    
    d = [gamma(1,H)]
    
    for n in 2:N
        
        # s = \sigma_{n+1}^2
        s = sigsq[n-1] - ((gamma(n+1,H) - tau[n-1])^2) / sigsq[n-1]
        append!(sigsq, s)

        F = [i + j == n + 1 for i in 1:n, j in 1:n]
        c = [gamma(k+1,H) for k in 0:n-1]

        # d(n+1)
        phi = (gamma(n+1,H) - tau[n-1])/sigsq[n-1]
        d .= d - phi * F * d
        append!(d, phi)
        
        Xn1 = mu[n-1] + sigsq[n-1] * randn(rng)
        
        append!(X, Xn1)
        append!(mu, sum(d .* X[end:-1:begin]))
        append!(tau, sum(c .* (F * d)))
    end
    
    fBm = cumsum(X) * N^(-H)    
    return T^H * fBm
end