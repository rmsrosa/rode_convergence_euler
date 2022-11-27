
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
    CompoundPoisson_noise(t0, tf, λ, dY)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dY`.

The noise returned by the constructor yields a random sample path of ``Y_t = \\sum_{i=1}^{N_t} dY_i`` obtained by first drawing the number of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the increment is performed by adding `n` samples of the given distribution `dY`.
"""
function CompoundPoisson_noise(t0, tf, λ, dY)
    fn = function (rng, Yt::Vector)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        dN = Poisson(λ * dt)
        Yt[1] = 0.0
        for n in 2:N
            Ni = rand(rng, dN)
            Yt[n] = Yt[n-1]
            for _ in 1:Ni
                Yt[n] += rand(rng, dY)
            end
        end
    end
end

"""
    CompoundPoisson_noise_alt(t0, tf, λ, dY)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dY`.

The noise returned by the constructor yields a random sample path of ``Y_t = \\sum_{i=1}^{N_t} dY_i`` obtained by first drawing the interarrival times, along with the increments given by `dY`, during each mesh time interval.

drawing the number of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.
"""
function CompoundPoisson_noise_alt(t0, tf, λ, dY)
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
                Yt[i] += rand(rng, dY)
                r += -log(rand(rng)) / λ
            end
        end
    end
end

"""
    StepPoisson_noise(t0, tf, λ, Y)

Construct a point Poisson process on the interval `t0` to `tf`, with a point Poisson counter with rate parameter `λ` and step values given by the distribution `Y`.

The noise returned by the constructor yields a random sample path of ``Y_t = Y_{N_t}`` obtained by first drawing the number of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the next state is repeated from the previous value, if `n` is zero, or set a new sample value of `Y`, if `n` is positive. Since it is not cumulative, it doesn't make any difference, for the discretized sample, whether `n` is larger than `1` or not.
"""
function StepPoisson_noise(t0, tf, λ, Y)
    fn = function (rng, Yt::Vector)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        dN = Poisson(λ * dt)
        Yt[1] = 0.0
        for n in 2:N
            r = rand(rng, dN)
            Yt[n] = iszero(r) ? Yt[n-1] : rand(rng, R)
        end
    end
end

"""
    Transport_noise(t0, tf, f, RV, n)

Construct a transport process on the time interval `t0` to `tf`, with function `f=f(t, y)` where `y` is a random vector with dimension `n` and distribution law for each coordinate given by `RV`.

The noise returned by the constructor yields a random sample path obtained by first drawing `n` realizations of the distribution `RV` to build the sample value `y` and then defining the sample path by `Y_t = f(t, y)` for each `t` in the time mesh obtained dividing the interval from `t0` to `tf` into `n-1` intervals.
"""
function Transport_noise(t0, tf, f, RV, n)
    rv = zeros(n)
    fn = function (rng, Yt::Vector; rv = rv)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        for i in eachindex(rv)
            rv[i] = rand(rng, RV)
        end
        t = t0 - dt
        for n in 1:N
            t += dt
            Yt[n] = f(t, rv)
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