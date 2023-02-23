"""
    Wiener_noise(t0, tf, y0)

Construct a Wiener process on the interval `t0` to `tf`, with initial condition `y0`.

The noise process `noise! = Wiener_noise(t0, tf, y0)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.
    
The number of steps for the sample path is determined by the length of the given vector `Yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(Yt) - 1)`. The initial condition is `Yt[1] = y0`, corresponding to the value at time `t0`.
"""
function Wiener_noise(t0::T, tf::T, y0::T) where {T}
    fn = function (rng::AbstractRNG, Yt::AbstractVector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        sqrtdt = sqrt(dt)
        n1 = firstindex(Yt)
        Yt[n1] = y0
        for n in Iterators.drop(eachindex(Yt), 1)
            Yt[n] = Yt[n1] + sqrtdt * randn(rng)
            n1 = n
        end
    end
    return fn
end

"""
    gBm_noise(t0, tf, μ, σ, y0)

Construct a Geometric Brownian motion process on the interval `t0` to `tf`, with initial condition `y0`, drift `μ` and diffusion `σ`.

The noise process `noise! = gBm_noise(t0, tf, μ, σ, y0)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process, defined by ``dY_t = \\mu Y_t dt + \\sigma Y_t dW_t``.
    
The number of steps for the sample path is determined by the length of the given vector `Yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(Yt) - 1)`. The initial condition is `Yt[1] = y0`, corresponding to the value at time `t0`.
"""
function gBm_noise(t0::T, tf::T, μ::T, σ::T, y0::T) where {T}
    fn = function (rng::AbstractRNG, Yt::AbstractVector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        sqrtdt = sqrt(dt)
        a = (μ + σ^2/2)
        n1 = firstindex(Yt)
        Yt[n1] = y0
        for n in Iterators.drop(eachindex(Yt), 1)
            Yt[n] = Yt[n1] * exp(a * dt + σ * sqrtdt * randn(rng))
            n1 = n
        end
    end
    return fn
end

"""
    CompoundPoisson_noise(t0, tf, λ, dYlaw)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dYlaw`.

The noise process `noise! = CompoundPoisson_noise(t0, tf, λ, dYlaw)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

The noise process returned by the constructor yields a random sample path of ``Y_t = \\sum_{i=1}^{N_t} dY_i``, where ``N_t`` is the number of events up to time ``t``.

Each sample path is obtained by first drawing the number `n`of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the increment is performed by adding `n` samples of the given increment distribution `dYlaw`.
"""
function CompoundPoisson_noise(t0::T, tf::T, λ::T, dYlaw::G) where {T, G}
    fn = function (rng::AbstractRNG, Yt::AbstractVector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        dN = Poisson(λ * dt)
        n1 = firstindex(Yt)
        Yt[n1] = 0.0
        for n in Iterators.drop(eachindex(Yt), 1)
            Ni = rand(rng, dN)
            Yt[n] = Yt[n1]
            for _ in 1:Ni
                Yt[n] += rand(rng, dYlaw)
            end
            n1 = n
        end
    end
end

"""
    CompoundPoisson_noise_alt(t0, tf, λ, dYlaw)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dYlaw`.

The noise process `noise! = CompoundPoisson_noise(t0, tf, λ, dYlaw)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

The noise returned by the constructor yields a random sample path of ``Y_t = \\sum_{i=1}^{N_t} dY_i`` obtained by first drawing the interarrival times, along with the increments given by `dYlaw`, during each mesh time interval.

This is an alternative implementation to [`CompoundPoisson_noise`](@ref).
"""
function CompoundPoisson_noise_alt(t0::T, tf::T, λ::T, dYlaw::G) where {T, G}
    fn = function (rng::AbstractRNG, Yt::Vector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        Yt[1] = zero(λ)
        i = 1
        while i < N
            i += 1
            Yt[i] = Yt[i-1]
            r = - log(rand(rng)) / λ
            while r < dt
                Yt[i] += rand(rng, dYlaw)
                r += -log(rand(rng)) / λ
            end
        end
    end
end

"""
    StepPoisson_noise(t0, tf, λ, Slaw)

Construct a point Poisson process on the interval `t0` to `tf`, with a point Poisson counter with rate parameter `λ` and step values given by the distribution `Slaw`.

The noise process `noise! = StepPoisson_noise(t0, tf, λ, Slaw)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

The noise returned by the constructor yields a random sample path of ``Y_t = Y_{N_t}`` obtained by first drawing the number `n` of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the next state is repeated from the previous value, if `n` is zero, or set a new sample value of `Y`, if `n` is positive. Since it is not cumulative and it has the Markov property, it doesn't make any difference, for the discretized sample, whether `n` is larger than `1` or not.
"""
function StepPoisson_noise(t0::T, tf::T, λ::T, Slaw::G) where {T, G}
    fn = function (rng::AbstractRNG, Yt::AbstractVector{T})
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        dN = Poisson(λ * dt)
        n1 = firstindex(Yt)
        Yt[n1] = 0.0
        for n in 2:N
            Ni = rand(rng, dN)
            Yt[n] = iszero(Ni) ? Yt[n1] : rand(rng, Slaw)
            n1 = n
        end
    end
end

"""
    Transport_noise(t0, tf, f, Ylaw, n)

Construct a transport process on the time interval `t0` to `tf`, with function `f=f(t, y)` where `y` is a random vector with dimension `n` and distribution law for each coordinate given by `Ylaw`.

The noise process `noise! = Transport_noise(t0, tf, f, Ylaw, n)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

Each random sample path is obtained by first drawing `n` realizations of the distribution `Ylaw` to build the sample value `y` and then defining the sample path by `Y_t = f(t, y)` for each `t` in the time mesh obtained dividing the interval from `t0` to `tf` into `n-1` intervals.
"""
function Transport_noise(t0::T, tf::T, f::F, Ylaw::G, n::S) where {T, S, F, G}
    rv = zeros(T, n)
    fn = function (rng::AbstractRNG, Yt::AbstractVector{T}; rv::Vector{T} = rv)
        N = length(Yt)
        dt = (tf - t0) / (N - 1)
        for i in eachindex(rv)
            rv[i] = rand(rng, Ylaw)
        end
        t = t0 - dt
        for n in eachindex(Yt)
            t += dt
            Yt[n] = f(t, rv)
        end
    end
end

"""
    fBm_noise(t0, tf, y0, H, N; flags=FFTW.MEASURE)

Construct a fractional Brownian motion process on the interval `t0` to `tf`, with initial condition `y0`, Husrt parameter `H` and length up to `N`.

The noise process `noise! = fBm_noise(t0, tf, y0, H, N; flags=FFTW.MEASURE)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.
    
The number of steps for the sample path is determined by the length of the given vector `Yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(Yt) - 1)`. The initial condition is `Yt[1] = y0`, corresponding to the value at time `t0`. The length of `Yt` must be smaller than or equal to the length `N` given in the constructor and used for the pre-allocation of the auxiliary vectors.

The method implemented is the one developed by Davies and Harte and uses an FFT transform to drop the order of complexity to O(N log N). For the transform, we use `FFTW.jl`, and use the flag `flags=FFTW.MEASURE` for generating the plans. Other common flags can be passed instead.

Implementation of fractional Brownian motion via Davies-Harte method following [Dieker, T. (2004) Simulation of Fractional Brownian Motion. MSc Theses, University of Twente, Amsterdam](http://www.columbia.edu/~ad3217/fbm/thesis.pdf) and A. [B. Dieker and M. Mandjes, On spectral simulation of fractional Brownian motion, Probability in the Engineering and Informational Sciences, 17 (2003), 417-434](https://www.semanticscholar.org/paper/ON-SPECTRAL-SIMULATION-OF-FRACTIONAL-BROWNIAN-Dieker-Mandjes/b2d0d6a3d7553ae67a9f6bf0bbe21740b0914163)
"""
function fBm_noise(t0::Float64, tf::Float64, y0::Float64, H::Float64, N::Int; flags=FFTW.MEASURE)
    ispow2(N) || throw(
        ArgumentError(
            "Desired length must be a power of 2 for this implementation of the Davies-Harte method."
        )
    )
    0.0 < H < 1.0 || throw(
        ArgumentError(
            "Hurst parameter should be strictly between 0.0 and 1.0."
        )
    )
    cache_real = Vector{Float64}(undef, 2N)
    cache_complex = Vector{ComplexF64}(undef, 2N)
    cache_complex2 = Vector{ComplexF64}(undef, 2N)
    plan_inverse = plan_ifft(cache_real; flags)
    plan_direct = plan_fft(cache_complex; flags)

    fn = function (rng::AbstractRNG, Yt::AbstractVector{Float64}; cache_real::Vector{Float64}=cache_real, cache_complex::Vector{ComplexF64}=cache_complex, cache_complex2::Vector{ComplexF64}=cache_complex2, plan_inverse=plan_inverse, plan_direct=plan_direct)
        length(Yt) ≤ N || throw(
            ArgumentError(
                "length of the sample path vector should be at most that given in the construction of the fBm noise process."
            )
        )

        # covariance function in Dieker eq. (1.7)
        gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

        # the first row of the circulant matrix in Dieker eq. (2.9)
        cache_complex[1] = 1.0
        cache_complex[N+1] = 0.0
        for k in 1:N-1
            cache_complex[2N-k+1] = cache_complex[k+1] = gamma(k, H)
        end

        # square-root of eigenvalues as in Dieker eq. (2.10) - using FFTW
        mul!(cache_complex2, plan_inverse, cache_complex)
        # cache_complex .= ifft(cache_real)
        map!(r -> sqrt(2N * real(r)), cache_real, cache_complex2)

        # generate Wⱼ according to step 2 in Dieker pages 16-17
        cache_complex[1] = randn(rng)
        cache_complex[N+1] = randn(rng)
        for j in 2:N
            v1 = randn(rng)
            v2 = randn(rng)
            cache_complex[j] = (v1 + im * v2) / √2
            cache_complex[2N-j+2] = (v1 - im * v2) / √2
        end

        # multiply Wⱼ by √λⱼ to prep for DFT
        cache_complex .*= cache_real

        # Discrete Fourier transform of √λⱼ Wⱼ according to Dieker eq. (2.12) via FFTW
        mul!(cache_complex2, plan_direct, cache_complex)
        # cache_real = real(fft(cache_complex)) / √(2N)
        cache_complex2 ./= √(2N)

        map!(real, cache_real, cache_complex2)

        # Rescale from [0, N] to [0, T]
        cache_real .*= ((tf - t0)/N)^(H)

        # fGn is made of the first N values of Z 
        Yt[begin] = y0
        Yt[begin+1:end] .= view(cache_real, 2:length(Yt))
        # fBm sample Yt is made of the first N values of Z 
        cumsum!(Yt, Yt)
    end
    return fn
end


# fBm_hosking(rng, T, N, H) generates sample paths of fractional Brownian Motion using the Hosking method, but I am not sure it is correct. I focused on the Davies-Harte method instead and stopped working on this one.
function fBm_hosking(rng, T, N, H)
    
    # Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) - 2*abs(k)^(2H) + abs(k+1)^(2H))
    
    # Dieker Sec. 2.1.1.
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

#  An alternate implementation for fractional Brownian motion process via the Hosking method, but I think it is not working either. Don't think I finish implementing it.
function fBm_hosking2(rng, T, N, H)
    
    # Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

    output = Vector{Float64}(undef, N)
    phi = Vector{Float64}(undef, N)
    psi = Vector{Float64}(undef, N)
    cov = Vector{Float64}(undef, N)

    output[1] = randn(rng)
    v = 1
    phi[1] = 0.0
    for i in 1:N
        cov[i] = gamma(i-1, H)
    end
    for i in 2:N
        phi[i-1] = cov[i]
        for j in 1:i-1
            psi[j] = phi[j]
            phi[i-1] -= psi[j] * cov[i-j]
        end
        phi[i-1] /= v
        for j in 1:i-1
            phi[j] = psi[j] - phi[i-1] * psi[i-j];
        end
        v *= (1 - phi[i-1] * phi[i-1])
    
        output[i] = 0;
        for j in 1:i
            output[i] += phi[j] * output[i-j+1];
        end
        output[i] += sqrt(v) * randn(rng);
    end
    scaling = (T/N)^H
    output[1] *= scaling
    for i in 2:N
        output[i] = output[i-1] + scaling * output[i]
    end
    return output
end

"""
    fG_daviesharte(rng, T, N, H)

Generates a sample path of a fractional Gaussian noise (fGn) with Hurst parameter `H` on the interval `[0, T]` discretized over a uniform mesh with `N` points (which must be a power of 2), with random numbers generated with `rng`.

This one is not optimized with pre-allocated plans, it was only used for development and testing.
"""
function fG_daviesharte(rng::AbstractRNG, T, N, H)
    ispow2(N) || throw(
        ArgumentError(
            "desired length must be a power of 2 for this implementation of the Davies-Harte method."
        )
    )

    # covariance function in Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

    # the first row of the circulant matrix in Dieker eq. (2.9)
    row = Vector{Float64}(undef, 2N)
    row[1] = 1.0
    row[N+1] = 0.0
    for k in 1:N-1
        row[2N-k+1] = row[k+1] = gamma(k, H)
    end

    # square-root of eigenvalues as in Dieker eq. (2.10) - using FFTW
    λsqrt = sqrt.(real(ifft(row) * 2N))

    # generate Wⱼ according to step 2 in Dieker pages 16-17
    Wl = Vector{ComplexF64}(undef, 2N)
    Wl[1] = randn(rng)
    Wl[N+1] = randn(rng)
    for j in 2:N
        v1 = randn(rng)
        v2 = randn(rng)
        Wl[j] = (v1 + im * v2) / √2
        Wl[2N-j+2] = (v1 - im * v2) / √2
    end

    # multiply Wⱼ by √λⱼ to prep for DFT
    Wl .*= λsqrt

    # Discrete Fourier transform of √λⱼ Wⱼ according to Dieker eq. (2.12) via FFTW
    Z = real(fft(Wl)) / √(2N)

    # Rescale from [0, N] to [0, T]
    Z *= (T/N)^(H)

    # fBm sample is made of the first N values of Z 
    return Z[1:N]
end

"""
    fG_daviesharte_naive(rng, T, N, H)

Generates a sample path of a fractional Gaussian noise (fGn) with Hurst parameter `H` on the interval `[0, T]` discretized over a uniform mesh with `N` points (which must be a power of 2), with random numbers generated with `rng`.

This one does not use `FFTW.jl` and is pretty slow, using a naive discrete transform that is not FFT and with complexity O(N^2). It was only used for development and testing.
"""
function fG_daviesharte_naive(rng, T, N, H)
    ispow2(N) || throw(
        ArgumentError(
            "desired length must be a power of 2 for this implementation of the Davies-Harte method."
        )
    )

    # covariance function in Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

    # the first row of the circulant matrix in Dieker eq. (2.9)
    row = [[gamma(k, H) for k in 0:N-1]; 0.0; [gamma(k, H) for k in N-1:-1:1]]

    # square-root of the eigenvalues as in Dieker eq. (2.10) - straighforward inverse DFT for testing
    λsqrt = [sqrt(real(sum(row[j+1] * exp(2π * im * j * k / (2N)) for j in 0:2N-1))) for k in 0:2N-1]

    # generate Wⱼ according to step 2 in Dieker pages 16-17
    Wl = Vector{ComplexF64}(undef, 2N)
    Wl[1], Wl[N+1] = randn(rng, 2)
    for j in 2:N
        v1, v2 = randn(rng, 2)
        Wl[j] = (v1 + im * v2) / √2
        Wl[2N-j+2] = (v1 - im * v2) / √2
    end

    # multiply Wⱼ by √λⱼ to prep for DFT
    Wl .*= λsqrt

    # straighforward DFT of √λⱼ Wⱼ for testing, only first N values are relevant:
    Z = [real(sum(Wl[j+1] * exp(-2π * im * j * k / (2N))/√(2N) for j in 0:2N-1)) for k in 0:N-1]
    # Z = [real(sum(Wl[j+1] * exp(-2π * im * j * k / (2N))/√(2N) for j in 0:2N-1)) for k in 0:2N-1]

    # Rescale from [0, N] to [0, T]
    Z *= (T/N)^(H)

    # fBm sample is made of the first N values of Z
    return Z
end

"""
    fBm_daviesharte(rng, T, N, H)

Generate a sample path of a fractional Brownian motion using a sample path of a fractional Gaussian noise from [`fG_daviesharte(rng, T, N, H)`](@ref), which is not optimized with pre-allocated plans.
"""
function fBm_daviesharte(rng, T, N, H)
    Z = fG_daviesharte(rng, T, N, H)
    return [zero(Z[1]); cumsum(view(Z, 1:N-1))]
end

"""
    MultiNoise()

"""
function MultiNoise(noises...)
    fn = function (rng::AbstractRNG, Yt::Matrix)
        #yaux = similar(view(Yt, :, 1))
        for (i, noise) in enumerate(noises)
            noise(rng, view(Yt, :, i))
            #noise(rng, yaux)
            #Yt[:, i] .= yaux
        end
    end
    return fn
end
