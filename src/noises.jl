abstract type AbstractProcess end

function Random.rand!(::AbstractRNG, ::AbstractProcess, ::T) where {T} end

"""
    Wiener_noise(t0, tf, y0)

Construct a Wiener process on the interval `t0` to `tf`, with initial condition `y0`.

The noise process `noise! = Wiener_noise(t0, tf, y0)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.
    
The number of steps for the sample path is determined by the length of the given vector `Yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(Yt) - 1)`. The initial condition is `Yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct WienerProcess{T} <: AbstractProcess
    t0::T
    tf::T
    y0::T
end

function Random.rand!(rng::AbstractRNG, Y::WienerProcess{T}, yt::AbstractVector{T}) where {T}
    N = length(yt)
    dt = (Y.tf - Y.t0) / (N - 1)
    sqrtdt = sqrt(dt)
    n1 = firstindex(yt)
    yt[n1] = Y.y0
    for n in Iterators.drop(eachindex(yt), 1)
        yt[n] = yt[n1] + sqrtdt * randn(rng)
        n1 = n
    end
end

"""
    gBm_noise(t0, tf, μ, σ, y0)

Construct a Geometric Brownian motion process on the interval `t0` to `tf`, with initial condition `y0`, drift `μ` and diffusion `σ`.

The noise process `noise! = gBm_noise(t0, tf, μ, σ, y0)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process, defined by ``dY_t = \\mu Y_t dt + \\sigma Y_t dW_t``.
    
The number of steps for the sample path is determined by the length of the given vector `Yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(Yt) - 1)`. The initial condition is `Yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct GeometricBrownianMotionProcess{T} <: AbstractProcess
    t0::T
    tf::T
    y0::T
    μ::T
    σ::T
end

function Random.rand!(rng::AbstractRNG, Y::GeometricBrownianMotionProcess{T}, yt::AbstractVector{T}) where {T}
    N = length(yt)
    dt = (Y.tf - Y.t0) / (N - 1)
    sqrtdt = sqrt(dt)
    a = (Y.μ + Y.σ^2/2)
    n1 = firstindex(yt)
    yt[n1] = Y.y0
    for n in Iterators.drop(eachindex(yt), 1)
        yt[n] = yt[n1] * exp(a * dt + Y.σ * sqrtdt * randn(rng))
        n1 = n
    end
end

"""
    CompoundPoisson_noise(t0, tf, λ, dYlaw)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dYlaw`.

The noise process `noise! = CompoundPoisson_noise(t0, tf, λ, dYlaw)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

The noise process returned by the constructor yields a random sample path of ``Y_t = \\sum_{i=1}^{N_t} dY_i``, where ``N_t`` is the number of events up to time ``t``.

Each sample path is obtained by first drawing the number `n`of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the increment is performed by adding `n` samples of the given increment distribution `dYlaw`.
"""
struct CompoundPoissonProcess{T, G} <: AbstractProcess
    t0::T
    tf::T
    y0::T
    λ::T
    dYlaw::G
end

function Random.rand!(rng::AbstractRNG, Y::CompoundPoissonProcess{T, G}, yt::AbstractVector{T}) where {T, G}
    N = length(yt)
    dt = (Y.tf - Y.t0) / (N - 1)
    dN = Poisson(Y.λ * dt)
    n1 = firstindex(yt)
    yt[n1] = Y.y0
    for n in Iterators.drop(eachindex(yt), 1)
        Ni = rand(rng, dN)
        yt[n] = yt[n1]
        for _ in 1:Ni
            yt[n] += rand(rng, Y.dYlaw)
        end
        n1 = n
    end
end

"""
    CompoundPoisson_noise_alt(t0, tf, λ, dYlaw)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dYlaw`.

The noise process `noise! = CompoundPoisson_noise(t0, tf, λ, dYlaw)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

The noise returned by the constructor yields a random sample path of ``Y_t = \\sum_{i=1}^{N_t} dY_i`` obtained by first drawing the interarrival times, along with the increments given by `dYlaw`, during each mesh time interval.

This is an alternative implementation to [`CompoundPoisson_noise`](@ref).
"""
struct CompoundPoissonProcessAlt{T, G} <: AbstractProcess
    t0::T
    tf::T
    y0::T
    λ::T
    dYlaw::G
end

function Random.rand!(rng::AbstractRNG, Y::CompoundPoissonProcessAlt{T, G}, yt::Vector{T}) where {T, G}
    N = length(yt)
    dt = (tf - t0) / (N - 1)
    n1 = firstindex(yt)
    yt[n1] = Y.y0
    while ni < N
        Yn1 = yt[n1]
        ni += 1
        yt[i] = Yn1
        r = - log(rand(rng)) / Y.λ
        while r < dt
            yt[ni] += rand(rng, Y.dYlaw)
            r += -log(rand(rng)) / Y.λ
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
struct PoissonStepProcess{T, G} <: AbstractProcess
    t0::T
    tf::T
    y0::T
    λ::T
    Slaw::G
end

function Random.rand!(rng::AbstractRNG, Y::PoissonStepProcess{T, G}, yt::AbstractVector{T}) where {T, G}
    N = length(yt)
    dt = (Y.tf - Y.t0) / (N - 1)
    dN = Poisson(Y.λ * dt)
    n1 = firstindex(yt)
    yt[n1] = Y.y0
    for n in Iterators.drop(eachindex(yt), 1)
        Ni = rand(rng, dN)
        yt[n] = iszero(Ni) ? yt[n1] : rand(rng, Y.Slaw)
        n1 = n
    end
end


"""
    Transport_noise(t0, tf, f, Ylaw, n)

Construct a transport process on the time interval `t0` to `tf`, with function `f=f(t, y)` where `y` is a random vector with dimension `n` and distribution law for each coordinate given by `Ylaw`.

The noise process `noise! = Transport_noise(t0, tf, f, Ylaw, n)` returned by the constructor is a function that takes a RNG `rng` and a pre-allocated vector `Yt` and, upon each call to `noise!(rng, Yt)`, mutates the vector `Yt`, filling it up with a new sample path of the process.

Each random sample path is obtained by first drawing `n` realizations of the distribution `Ylaw` to build the sample value `y` and then defining the sample path by `Y_t = f(t, y)` for each `t` in the time mesh obtained dividing the interval from `t0` to `tf` into `n-1` intervals.
"""
struct TransportProcess{T, F, G} <: AbstractProcess
    t0::T
    tf::T
    Ylaw::G
    f::F
    rv::Vector{T}
    TransportProcess(t0::T, tf::T, Ylaw::G, f::F, ny::Int64) where {T, F, G} = new{T, F, G}(t0, tf, Ylaw, f, zeros(T, ny))
end

struct Bar{T}
    x::T
    v::Vector{T}
    function Bar(x::T, n::Int64) where {T} 
        z = zeros(T, n)
        new{T}(x, z)
    end
end

function Random.rand!(rng::AbstractRNG, Y::TransportProcess{T, F, G}, yt::AbstractVector{T}) where {T, F, G}
    N = length(yt)
    dt = (Y.tf - Y.t0) / (N - 1)
    for i in eachindex(Y.rv)
        Y.rv[i] = rand(rng, Y.Ylaw)
    end
    # rand!(rng, Y.Ylaw, Y.rv) # Beta allocates but others don't; maybe a bug in Beta
    t = Y.t0 - dt
    for n in eachindex(yt)
        t += dt
        yt[n] = Y.f(t, Y.rv)
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
struct FractionalBrownianMotionProcess{P1, P2} <: AbstractProcess
    t0::Float64
    tf::Float64
    y0::Float64
    H::Float64
    N::Int
    cache_real::Vector{Float64}
    cache_complex::Vector{ComplexF64}
    cache_complex2::Vector{ComplexF64}
    plan_inverse::P1
    plan_direct::P2

    function FractionalBrownianMotionProcess(t0::Float64, tf::Float64, y0::Float64, H::Float64, N::Int; flags=FFTW.MEASURE)
        ispow2(N) || throw(
            ArgumentError(
                "Desired maximum length `N` must be a power of 2 for this implementation of the Davies-Harte method."
            )
        )
        0.0 < H < 1.0 || throw(
            ArgumentError(
                "Hurst parameter `H` should be strictly between 0.0 and 1.0."
            )
        )

        cache_real = Vector{Float64}(undef, 2N)
        cache_complex = Vector{ComplexF64}(undef, 2N)
        cache_complex2 = Vector{ComplexF64}(undef, 2N)
        plan_inverse = plan_ifft(cache_real; flags)
        plan_direct = plan_fft(cache_complex; flags)

        new{typeof(plan_inverse), typeof(plan_direct)}(t0, tf, y0, H, N, cache_real, cache_complex, cache_complex2, plan_inverse, plan_direct)
    end
end

function Random.rand!(rng::AbstractRNG, Y::FractionalBrownianMotionProcess{P1, P2}, yt::AbstractVector{T}) where {T, P1, P2}

    length(yt) ≤ Y.N || throw(
        ArgumentError(
            "length of the sample path vector should be at most that given in the construction of the fBm noise process."
        )
    )

    t0 = Y.t0
    tf = Y.tf
    H = Y.H
    y0 = Y.y0
    N = Y.N
    plan_inverse = Y.plan_inverse
    plan_direct = Y.plan_direct

    # covariance function in Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

    # the first row of the circulant matrix in Dieker eq. (2.9)
    Y.cache_complex[1] = 1.0
    Y.cache_complex[N+1] = 0.0
    for k in 1:N-1
        Y.cache_complex[2N-k+1] = Y.cache_complex[k+1] = gamma(k, H)
    end

    # square-root of eigenvalues as in Dieker eq. (2.10) - using FFTW
    mul!(Y.cache_complex2, plan_inverse, Y.cache_complex)
    # Y.cache_complex .= ifft(Y.cache_real)
    map!(r -> sqrt(2N * real(r)), Y.cache_real, Y.cache_complex2)

    # generate Wⱼ according to step 2 in Dieker pages 16-17
    Y.cache_complex[1] = randn(rng)
    Y.cache_complex[N+1] = randn(rng)
    for j in 2:N
        v1 = randn(rng)
        v2 = randn(rng)
        Y.cache_complex[j] = (v1 + im * v2) / √2
        Y.cache_complex[2N-j+2] = (v1 - im * v2) / √2
    end

    # multiply Wⱼ by √λⱼ to prep for DFT
    Y.cache_complex .*= Y.cache_real

    # Discrete Fourier transform of √λⱼ Wⱼ according to Dieker eq. (2.12) via FFTW
    mul!(Y.cache_complex2, plan_direct, Y.cache_complex)
    # Y.cache_real = real(fft(Y.cache_complex)) / √(2N)
    Y.cache_complex2 ./= √(2N)

    map!(real, Y.cache_real, Y.cache_complex2)

    # Rescale from [0, N] to [0, T]
    Y.cache_real .*= ((tf - t0)/N)^(H)

    # fGn is made of the first N values of Z 
    yt[begin] = y0
    yt[begin+1:end] .= view(Y.cache_real, 2:length(yt))
    # fBm sample yt is made of the first N values of Z 
    cumsum!(yt, yt)
end

"""
    MultiNoise()

"""
function MultiProcess_noise(noises...)
    fn = function (rng::AbstractRNG, yt::AbstractMatrix; noises::Tuple=noises)
        axes(eachcol(yt)) == axes(noises) || throw(
            DimensionMismatch("Columns of `yt` and list of noises in `noises` must match indices; got $(axes(eachcol(yt))) and $(axes(noises)).")
        )
        for (noise, yti) in zip(noises, eachcol(yt))
            noise(rng, yti)
        end
    end
    return fn
end
