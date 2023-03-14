"""
    AbstractProcess{T, N}

Abstract super type for every noise process, with parameter `N` being either `Univariate` or `Multivariate` and `T` being the eltype of the process.
    
The following aliases are also defined:

* `UnivariateProcess{T} = AbstractProcess{T, Univariate}`
* `MultivariateProcess{T} = AbstractProcess{T, Multivariate}`

The parameter types are borrowed from Distributions.Univariate and Distributions.Multivariate.
"""
abstract type AbstractProcess{T, N} end

"""
    UnivariateProcess{T}

Supertype for univariate noise processes.

Alias for `AbstractProcess{T, Univariate}`.
"""
const UnivariateProcess{T} = AbstractProcess{T, Univariate} where {T}

"""
    MultivariateProcess{T}

Supertype for multivariate noise processes.

Alias for `AbstractProcess{T, Multivariate}`.
"""
const MultivariateProcess{T} = AbstractProcess{T, Multivariate} where {T}

Base.eltype(::AbstractProcess{T}) where {T} = T

Base.length(noise::UnivariateProcess) = 1

"""
    rand!(rng::AbstractRNG, noise::AbstractProcess{T}, yt::VecOrMat{T})

Generate sample paths of the `noise` process.

Populate the vector or matrix `yt` with a sample path of the process `noise`, with random numbers generated from `rng`. See each noise type for details.
"""
function Random.rand!(::AbstractRNG, noise::AbstractProcess{T}, ::S) where {T, S}
    throw(
        ArgumentError(
            "`rand!(rng, noise, yt)` not implemente for `noise` of type $(typeof(noise))"
        )
    )
end

"""
    WienerProcess(t0, tf, y0)

Construct a Wiener process on the interval `t0` to `tf`, with initial condition `y0`.

The noise process `noise = WienerProcess(t0, tf, y0)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.

Since, by definition, ``\\Delta W_t \\sim \\mathcal{N}(0, t)``, a sample path is constructed recursively by solving the recursive relation
```math
W_{t_i} = W_{t_{i-1}} + \\sqrt{dt} z_i, \\qquad i = 1, \\ldots,
```
where at each step `z_i` is drawn from a standard Normal distribution.
"""
struct WienerProcess{T} <: UnivariateProcess{T}
    t0::T
    tf::T
    y0::T
end

function Random.rand!(rng::AbstractRNG, noise::WienerProcess{T}, yt::AbstractVector{T}) where {T}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    sqrtdt = sqrt(dt)
    i1 = firstindex(yt)
    yt[i1] = noise.y0
    for i in Iterators.drop(eachindex(yt), 1)
        yt[i] = yt[i1] + sqrtdt * randn(rng)
        i1 = i
    end
end

"""
    OrnsteinUhlenbeckProcess(t0, tf, ν, σ, y0)

Construct an Ornstein Uhlenbeck process ``O_t`` on the interval `t0` to `tf`, with initial condition `y0`, drift `-θ` and diffusion `σ`, as defined by the equation
```math
\\mathrm{d}O_t = -\\nu O_t \\;\\mathrm{d}t + \\sigma \\;\\mathrm{d}W_t.
```

The solution is
```math
O_t = e^{-\\nu t}O_0 + \\sigma \\int_0^t e^{-\\nu (t - s)}\\;\\mathrm{d}W_s.
```

The noise process `noise = OrnsteinUhlenbeckProcess(t0, tf, μ, σ, y0)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.

Notice the integral term is a Normal random variable with zero mean and variance
```math
\\mathbb{E}\\left[ \\left( \\sigma \\int_0^t e^{-\\nu (t - s)} \\;\\mathrm{d}W_s\\right)^2\\right] = \\frac{\\sigma^2}{2\\nu}\\left( 1 - e^{-2\\nu t} \\right).
```

Thus, a sample path is constructed with exact distribution by solving the recursion relation
```math
O_{t_i} = e^{-\\nu \\Delta t} O_{t_{i-1}} + \\frac{\\sigma}{\\sqrt{2\\nu}} \\sqrt{1 - e^{-2\\nu \\Delta t}} z_i, \\qquad i = 1, \\ldots,
```
where at each time step `z_i` is drawn from a standard Normal distribution.
"""
struct OrnsteinUhlenbeckProcess{T} <: UnivariateProcess{T}
    t0::T
    tf::T
    y0::T
    ν::T
    σ::T
end

function Random.rand!(rng::AbstractRNG, noise::OrnsteinUhlenbeckProcess{T}, yt::AbstractVector{T}) where {T}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    a = exp(-noise.ν * dt)
    b = noise.σ * √((1 - a ^ 2) / (2 * noise.ν))
    i1 = firstindex(yt)
    yt[i1] = noise.y0
    for i in Iterators.drop(eachindex(yt), 1)
        yt[i] = a * yt[i1] + b * randn(rng)
        i1 = i
    end
end

"""
    GeometricBrownianMotionProcess(t0, tf, μ, σ, y0)

Construct a Geometric Brownian motion process ``Y_t`` on the interval `t0` to `tf`, with initial condition `y0`, drift `μ` and diffusion `σ`, as defined by
```math
\\mathrm{d}Y_t = \\mu Y_t \\;\\mathrm{d}t + \\sigma Y_t \\;\\mathrm{d}W_t.
```

The noise process `noise = GeometricBrownianMotionProcess(t0, tf, μ, σ, y0)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct GeometricBrownianMotionProcess{T} <: UnivariateProcess{T}
    t0::T
    tf::T
    y0::T
    μ::T
    σ::T
end

function Random.rand!(rng::AbstractRNG, noise::GeometricBrownianMotionProcess{T}, yt::AbstractVector{T}) where {T}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    a = (noise.μ + noise.σ^2/2) * dt
    b = noise.σ * sqrt(dt)
    i1 = firstindex(yt)
    yt[i1] = noise.y0
    for i in Iterators.drop(eachindex(yt), 1)
        yt[i] = yt[i1] * exp(a + b * randn(rng))
        i1 = i
    end
end

"""
    CompoundPoissonProcess(t0, tf, λ, dylaw)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dylaw`.

The noise process `noise = CompoundPoissonProcess(t0, tf, λ, dylaw)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.

The noise process returned by the constructor yields a random sample path of
```math
Y_t = \\sum_{i=1}^{N_t} \\;\\mathrm{d}Y_i,
```
where ``N_t`` is the number of events up to time ``t``.

Each sample path is obtained by first drawing the number `n` of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the increment is performed by adding `n` samples of the given increment distribution `dylaw`.

The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is set to `yt[1] = 0`, corresponding to the value at time `t0`.
"""
struct CompoundPoissonProcess{T, G} <: UnivariateProcess{T}
    t0::T
    tf::T
    λ::T
    dylaw::G
end

function Random.rand!(rng::AbstractRNG, noise::CompoundPoissonProcess{T, G}, yt::AbstractVector{T}) where {T, G}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    dnlaw = Poisson(noise.λ * dt)
    i1 = firstindex(yt)
    yt[i1] = zero(T)
    for i in Iterators.drop(eachindex(yt), 1)
        numi = rand(rng, dnlaw)
        yt[i] = yt[i1]
        for _ in 1:numi
            yt[i] += rand(rng, noise.dylaw)
        end
        i1 = i
    end
end

"""
    CompoundPoissonProcessAlt(t0, tf, λ, dylaw)

Construct a Compound Poisson process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ` and increments given by the distribution `dylaw`.

The noise process `noise = CompoundPoissonProcessAlt(t0, tf, λ, dylaw)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.

The noise returned by the constructor yields a random sample path by first drawing the interarrival times, along with the increments given by `dylaw`, during each mesh time interval.

This is an alternative implementation to [`CompoundPoissonProcess`](@ref).
"""
struct CompoundPoissonProcessAlt{T, G} <: UnivariateProcess{T}
    t0::T
    tf::T
    λ::T
    dylaw::G
end

function Random.rand!(rng::AbstractRNG, noise::CompoundPoissonProcessAlt{T, G}, yt::Vector{T}) where {T, G}
    n = length(yt)
    dt = (tf - t0) / (n - 1)
    n1 = firstindex(yt)
    yt[n1] = zero(T)
    while ni < n
        Yn1 = yt[n1]
        ni += 1
        yt[i] = Yn1
        r = - log(rand(rng)) / noise.λ
        while r < dt
            yt[ni] += rand(rng, noise.dylaw)
            r -= log(rand(rng)) / noise.λ
        end
    end
end

"""
    PoissonStepProcess(t0, tf, λ, steplaw)

Construct a point Poisson process on the interval `t0` to `tf`, with a point Poisson counter with rate parameter `λ` and step values given by the distribution `steplaw`.

The noise process `noise = PoissonStepProcess(t0, tf, λ, steplaw)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.

The noise returned by the constructor yields a random sample path of ``Y_t = Y_{N_t}`` obtained by first drawing the number `n` of events between consecutive times with interval `dt` according to the Poisson distribution `n = N(t+dt) - N(t) = Poisson(λdt)`.

Then, based on the number `n` of events, the next state is repeated from the previous value, if `n` is zero, or set to a new sample value of `Y`, if `n` is positive. Since it is not cumulative and it has the Markov property, it doesn't make any difference, for the discretized sample, whether `n` is larger than `1` or not.

The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct PoissonStepProcess{T, G} <: UnivariateProcess{T}
    t0::T
    tf::T
    λ::T
    steplaw::G
end

function Random.rand!(rng::AbstractRNG, noise::PoissonStepProcess{T, G}, yt::AbstractVector{T}) where {T, G}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    dnlaw = Poisson(noise.λ * dt)
    i1 = firstindex(yt)
    yt[i1] = zero(T)
    for i in Iterators.drop(eachindex(yt), 1)
        numi = rand(rng, dnlaw)
        yt[i] = iszero(numi) ? yt[i1] : rand(rng, noise.steplaw)
        i1 = i
    end
end

"""
    TransportProcess(t0, tf, f, ylaw, n)

Construct a transport process on the time interval `t0` to `tf`, with function `f=f(t, y)` where `y` is a random vector with dimension `n` and distribution law for each coordinate given by `ylaw`.

The noise process `noise = TransportProcess(t0, tf, f, ylaw, n)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.

Each random sample path is obtained by first drawing `n` realizations of the distribution `ylaw` to build the sample value `y` and then defining the sample path by `Y_t = f(t, y)` for each `t` in the time mesh obtained dividing the interval from `t0` to `tf` into `n-1` intervals.

The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct TransportProcess{T, F, G, N} <: UnivariateProcess{T}
    t0::T
    tf::T
    ylaw::G
    f::F
    rv::Array{T, N}
    function TransportProcess(t0::T, tf::T, ylaw::G, f::F, ny::Int64) where {T, F, G}
        N = length(ylaw)
        rv = ylaw isa ContinuousUnivariateDistribution ? zeros(T, ny) : zeros(T, ny, N)
        new{T, F, G, N}(t0, tf, ylaw, f, rv)
    end
end

function Random.rand!(rng::AbstractRNG, noise::TransportProcess{T, F, G}, yt::AbstractVector{T}) where {T, F, G}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    if noise.ylaw isa ContinuousUnivariateDistribution
        for i in eachindex(noise.rv)
            @inbounds noise.rv[i] = rand(rng, noise.ylaw)
        end
    else
        for i in eachindex(eachrow(noise.rv))
            rand!(rng, noise.ylaw, view(noise.rv, i, :))
        end
    end
    # rand!(rng, noise.ylaw, noise.rv) # Most distributions don't allocate but Beta does (see https://github.com/JuliaStats/Distributions.jl/pull/1281), so I prefer to roll out the loop explicitly
    t = noise.t0 - dt
    for j in eachindex(yt)
        t += dt
        @inbounds yt[j] = noise.f(t, noise.rv)
    end
end

"""
    FractionalBrownianMotionProcess(t0, tf, y0, hurst, len; flags=FFTW.MEASURE)

Construct a fractional Brownian motion process on the interval `t0` to `tf`, with initial condition `y0`, Hurst parameter `hurst` and length up to `len`.

The noise process `noise = FractionalBrownianMotionProcess(t0, tf, y0, hurst, len; flags=FFTW.MEASURE)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`. The length of `yt` must be smaller than or equal to the length `len` given in the constructor and used for the pre-allocation of the auxiliary vectors.

The method implemented is the one developed by Davies and Harte and uses an FFT transform to drop the order of complexity to O(N log N). For the transform, we use `FFTW.jl`, and use the flag `flags=FFTW.MEASURE` for generating the plans. Other common flags can be passed instead.

This implementation of fractional Brownian motion via Davies-Harte method follows [Dieker, T. (2004) Simulation of Fractional Brownian Motion. MSc Theses, University of Twente, Amsterdam](http://www.columbia.edu/~ad3217/fbm/thesis.pdf) and A. [B. Dieker and M. Mandjes, On spectral simulation of fractional Brownian motion, Probability in the Engineering and Informational Sciences, 17 (2003), 417-434](https://www.semanticscholar.org/paper/ON-SPECTRAL-SIMULATION-OF-FRACTIONAL-BROWNIAN-Dieker-Mandjes/b2d0d6a3d7553ae67a9f6bf0bbe21740b0914163)
"""
struct FractionalBrownianMotionProcess{P1, P2} <: UnivariateProcess{Float64}
    t0::Float64
    tf::Float64
    y0::Float64
    hurst::Float64
    len::Int
    cache_real::Vector{Float64}
    cache_complex::Vector{ComplexF64}
    cache_complex2::Vector{ComplexF64}
    plan_inverse::P1
    plan_direct::P2

    function FractionalBrownianMotionProcess(t0::Float64, tf::Float64, y0::Float64, hurst::Float64, len::Int; flags=FFTW.MEASURE)
        ispow2(len) || throw(
            ArgumentError(
                "Desired maximum length `len` must be a power of 2 for this implementation of the Davies-Harte method."
            )
        )
        0.0 < hurst < 1.0 || throw(
            ArgumentError(
                "Hurst parameter should be strictly between 0.0 and 1.0."
            )
        )

        cache_real = Vector{Float64}(undef, 2len)
        cache_complex = Vector{ComplexF64}(undef, 2len)
        cache_complex2 = Vector{ComplexF64}(undef, 2len)
        plan_inverse = plan_ifft(cache_real; flags)
        plan_direct = plan_fft(cache_complex; flags)

        new{typeof(plan_inverse), typeof(plan_direct)}(t0, tf, y0, hurst, len, cache_real, cache_complex, cache_complex2, plan_inverse, plan_direct)
    end
end

function Random.rand!(rng::AbstractRNG, noise::FractionalBrownianMotionProcess, yt::AbstractVector{Float64})

    length(yt) ≤ noise.len || throw(
        ArgumentError(
            "length of the sample path vector should be at most that given in the construction of the fBm noise process."
        )
    )

    t0 = noise.t0
    tf = noise.tf
    H = noise.hurst
    y0 = noise.y0
    n = noise.len
    plan_inverse = noise.plan_inverse
    plan_direct = noise.plan_direct

    # covariance function in Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

    # the first row of the circulant matrix in Dieker eq. (2.9)
    noise.cache_complex[1] = 1.0
    noise.cache_complex[n+1] = 0.0
    for k in 1:n-1
        noise.cache_complex[2n-k+1] = noise.cache_complex[k+1] = gamma(k, H)
    end

    # square-root of eigenvalues as in Dieker eq. (2.10) - using FFTW
    mul!(noise.cache_complex2, plan_inverse, noise.cache_complex)
    map!(r -> sqrt(2n * real(r)), noise.cache_real, noise.cache_complex2)

    # generate Wⱼ according to step 2 in Dieker pages 16-17
    noise.cache_complex[1] = randn(rng)
    noise.cache_complex[n+1] = randn(rng)
    for j in 2:n
        v1 = randn(rng)
        v2 = randn(rng)
        noise.cache_complex[j] = (v1 + im * v2) / √2
        noise.cache_complex[2n-j+2] = (v1 - im * v2) / √2
    end

    # multiply Wⱼ by √λⱼ to prep for DFT
    noise.cache_complex .*= noise.cache_real

    # Discrete Fourier transform of √λⱼ Wⱼ according to Dieker eq. (2.12) via FFTW
    mul!(noise.cache_complex2, plan_direct, noise.cache_complex)
    noise.cache_complex2 ./= √(2n)

    map!(real, noise.cache_real, noise.cache_complex2)

    # Rescale from [0, N] to [0, T]
    noise.cache_real .*= ((tf - t0)/n)^(H)

    # fGn is made of the first n values of Z 
    yt[begin] = y0
    yt[begin+1:end] .= view(noise.cache_real, 2:length(yt))
    # fBm sample yt is made of the first N values of Z 
    cumsum!(yt, yt)
end

"""
    ProductProcess(noises...)

Construct a multivariate process from independent univariate processes.

The noise process `noise = ProductProcess(noises...)` returned by the constructor is a subtype of `AbstractNoise{Multivariate}`.

Sample paths are obtained by populating a pre-allocated matrix `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the number of rows of the given matrix `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (size(yt, 1) - 1)`.
    
Each columns of `yt` is populated with a sample path from each univariate process in `noise`.
"""
struct ProductProcess{T, D} <: MultivariateProcess{T}
    processes::D
    function ProductProcess(p::D) where {D <: Tuple{Vararg{UnivariateProcess}}} 
        isempty(p) && error(
            "ProductProcess must consist of at least one univariate process"
        )
        all(pi -> pi isa UnivariateProcess, p) || error(
            "each and every element must be a univariate process"
        )
        T = eltype(first(p))
        all(pi -> eltype(pi) == T, p) || error(
            "all processes must have same eltype"
        )

        return new{T, D}(p)
    end
end

ProductProcess(p::UnivariateProcess...) = ProductProcess(p)

Base.length(noise::ProductProcess) = length(noise.processes)

#= function Random.rand!(rng::AbstractRNG, noise::ProductProcess{T}, yt::AbstractMatrix{T}) where {T}
    axes(eachcol(yt)) == axes(noise.processes) || throw(
        DimensionMismatch("Columns of `mat` must have same length and match the indices of `noise.processes`.")
    )
    for (i, yti) in enumerate(eachcol(yt))
        rand!(rng, noise.processes[i], yti)
    end
end =#

# `ProductProcess` was allocating a little when drawing `rand!(rng, noise, yt)` with different types of processes in `noise::ProductProcess`.
# It was just a small overhead, apparently not affecting performance.
# It could be due to failed inference and/or type instability.
# Anyway, I changed it to the generated function below, with specialized rolled out loop and it is fine, now, non-allocating.

@generated function Random.rand!(rng::AbstractRNG, noise::ProductProcess{T, D}, yt::AbstractMatrix{T}) where {T, D}
    n = length(D.parameters)
    ex = quote
            axes(eachcol(yt)) == axes(noise.processes) || throw(
            DimensionMismatch("Columns of `mat` must have same length and match the indices of `noise.processes`.")
        )
    end
    for i in 1:n
        ex = quote
            $ex
            rand!(rng, noise.processes[$i], view(yt, :, $i))
        end
    end
    ex = quote
        $ex
        nothing
    end
    return ex
end