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
            "`rand!(rng, noise, yt)` not implemented for `noise` of type $(typeof(noise))"
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
W_{t_i} = W_{t_{i-1}} + \\sqrt{\\mathrm{dt}} z_i, \\qquad i = 1, \\ldots,
```
where at each step ``z_i`` is drawn from a standard Normal distribution.
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
    OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ)

Construct an Ornstein Uhlenbeck process ``O_t`` on the interval `t0` to `tf`, with initial condition `y0`, drift `-ν` and diffusion `σ`, as defined by the equation
```math
\\mathrm{d}O_t = -\\nu O_t \\;\\mathrm{d}t + \\sigma \\;\\mathrm{d}W_t.
```

The solution is
```math
O_t = e^{-\\nu t}O_0 + \\sigma \\int_0^t e^{-\\nu (t - s)}\\;\\mathrm{d}W_s.
```

The noise process `noise = OrnsteinUhlenbeckProcess(t0, tf, y0, ν, σ)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

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
where at each time step ``z_i`` is drawn from a standard Normal distribution.

The Ornstein-Uhlenbeck process has mean, variance, and covariance given by
```math
    \\mathbb{E}[O_t] = O_0 e^{-\\nu t}, \\mathrm{Var}[O_t] = \\frac{\\sigma^2}{2\\nu}, \\quad \\mathrm{Cov}[O_tO_s] = \\frac{\\sigma^2}{2\\nu} e^{-\\nu |t - s|}.
```
so that ``O_t`` and ``O_s`` are significantly correlated only when ``|t - s| \\lesssim \\tau``, where ``\\tau = 1/\\nu`` is a characteristic time scale for the process. When ``\\tau \\rightarrow 0``, i.e. ``\\nu \\rightarrow \\infty``,  with ``\\sigma / \\nu = \\tau\\sigma \\rightarrow 1``, this approximates a Gaussian white noise.
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
    GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)

Construct a Geometric Brownian motion process ``Y_t`` on the interval `t0` to `tf`, with initial condition `y0`, drift `μ` and diffusion `σ`, as defined by
```math
\\mathrm{d}Y_t = \\mu Y_t \\;\\mathrm{d}t + \\sigma Y_t \\;\\mathrm{d}W_t.
```

The noise process `noise = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

The exact solution is given by
```math
Y_t = Y_0 e^{ \\left(\\mu - \\frac{\\sigma^2}{2}\\right) t + \\sigma W_t) }.
```

The discretized solution sample is computed recursively via
```math
Y_{t_j} = Y_{t_{j-1}} e^{ \\left(\\mu - \\frac{\\sigma^2}{2}\\right) \\Delta t + \\sigma \\sqrt{\\Delta t} Z_j) },
```
where ``Z_t \\sim \\mathcal{N}(0, 1)``.

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
    HomogeneousLinearItoProcess(t0, tf, y0, primitive_a, primitive_bsquare)

Construct a homogeneous linear Itô process noise ``Y_t`` on the interval `t0` to `tf`, with initial condition `y0`, as defined by the equation
```math
\\mathrm{d}Y_t = a(t) Y_t \\;\\mathrm{d}t + b(t) Y_t \\;\\mathrm{d}W_t,
```
provided the primitive of ``a=a(t)`` and the primitive of ``b^2 = b(t)^2`` are given, via `primitive_a` and `primitive_bsquare`, respectively.

The noise process `noise = HomogeneousLinearItoProcess(t0, tf, y0, primitive_a, primitive_bsquare)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

The exact solution has the form
```math
Y_t = y_0 e^{\\int_0^t (a(s) - \\frac{b(s)^2}{2}) \\;\\mathrm{d}s + \\int_0^t b(s) \\;\\mathrm{d}W_s}.
```

The basic statistics for this process can be computed by first computing the statistics for its logarithm, which satisfies
```math
\\mathbb{E}\\left[ \\ln Y_t \\right] = \\ln y_0 + \\int_0^t (a(s) - \\frac{b(s)^2}{2}) \\;\\mathrm{d}s,
```
and
```math
\\mathrm{Var}\\left( \\ln Y_t \\right) = \\int_0^t b(s)^2 \\;\\mathrm{d}s.
```
Then, since ``\\ln Y_t`` is Gaussian, ``Y_t`` is log-normal with
```math
\\mathbb{E}\\left[ Y_t \\right] = y_0 e^{ \\int_0^t a(s) \\;\\mathrm{d}s},
```
and
```math
\\mathrm{Var}\\left( Y_t \\right) = y_0^2 e^{\\int_0^t 2a(s) \\;\\mathrm{d}s}\\left( e^{\\int_0^t b(s)^2 \\;\\mathrm{d}s} - 1 \\right).
```

A distributionally exact solution is computed on the mesh points in a recursive manner by
```math
Y_{t_j} = Y_{t_{j-1}} e^{(p_a(t_j) - p_a(t_{j-1})) - (p_{b^2}(t_j) - p_{b^2}(t_{j-1})/2 + Z_j)}, \\qquad j = 1, \\ldots,
```
with ``Y_0 = y_0``, and where ``p_a = p_a(t)`` is the given primitive of ``a=a(t)``, ``p_{b^2} = p_{b^2}(t)`` is the given primitive of ``b^2 = b(t)^2``, and ``Z_j \\sim \\mathcal{N}(0, p_{b^2}(t_j) - p_{b^2}(t_{j-1}))``.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct HomogeneousLinearItoProcess{T, A, B} <: UnivariateProcess{T}
    t0::T
    tf::T
    y0::T
    primitive_a::A
    primitive_bsquare::B
end

function Random.rand!(rng::AbstractRNG, noise::HomogeneousLinearItoProcess{T}, yt::AbstractVector{T}) where {T}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    pa = noise.primitive_a
    pb2 = noise.primitive_bsquare
    i1 = firstindex(yt)
    yt[i1] = noise.y0
    tj = tj1 = noise.t0
    for i in Iterators.drop(eachindex(yt), 1)
        tj += dt
        dpa = pa(tj) - pa(tj1)
        dpb2 = pb2(tj) - pb2(tj1)
        yt[i] = yt[i1] * exp(dpa + dpb2/2 + √dpb2 * randn(rng))
        i1 = i
        tj1 = tj
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

Then, based on the number `n` of events, the increment is performed by adding `n` samples of the given increment distribution `dylaw`.

The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is set to `yt[1] = 0`, corresponding to the value at time `t0`.
"""
struct CompoundPoissonProcess{T, G} <: UnivariateProcess{T}
    t0::T
    tf::T
    λ::T
    dylaw::G
end

function Random.rand!(rng::AbstractRNG, noise::CompoundPoissonProcess{T}, yt::AbstractVector{T}) where {T}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    ti = ti1 = noise.t0
    ti -= log(rand(rng)) / noise.λ
    ni1 = firstindex(yt)
    yt[ni1] = zero(T)
    for ni in Iterators.drop(eachindex(yt), 1)
        yt[ni] = yt[ni1]
        ti1 += dt
        while ti ≤ ti1
            yt[ni] += rand(rng, noise.dylaw)
            ti -= log(rand(rng)) / noise.λ
        end
        ni1 = ni
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
    ExponentialHawkesProcess(t0, tf, λ₀, a, δ, dylaw)

Construct an Exponentially Decaying Hawkes process on the interval `t0` to `tf`, with point Poisson counter with rate parameter `λ`, jump increments given by the distribution `dylaw`, and exponential decay with rate `δ`. 

An exponentially decaying Hawkes process is a self-exciting point process ``\\lambda_t``, representing a time-dependent intensity rate for an inhomogenous Poisson counter with an initial intensity ``\\lambda_0``, a reversion level ``a`` with ``\\lambda_0 \\geq a \\geq 0``, an exponential decay with rate ``\\delta > 0``, and positive stationary random jump increments ``S_k``, at each arrival time ``T_k``. The process is define by

```math
    \\lambda_t = a + (\\lambda_0 - a) e^{-\\delta (t-t_0)} + \\sum_{t_0 \\leq T_k < t} S_k e^{-\\delta (t - T_k)}, \\quad t \\geq t_0.
```

The noise process `noise = ExponentialHawkesProcess(t0, tf, λ, δ, dylaw)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.

The noise returned by the constructor yields a random sample path by first drawing the interarrival times, along with the increments given by `dylaw`, during each mesh time interval, and then applying the exponential decay.

This implementation of the Hawkes process follows [A. Dassius and H. Zhao, Exact simulation of Hawkes process with exponentially decaying intensity, Electron. Commun. Probab. 18 (2013), no. 62, 1-13.](https://projecteuclid.org/journals/electronic-communications-in-probability/volume-18/issue-none/Exact-simulation-of-Hawkes-process-with-exponentially-decayingintensity/10.1214/ECP.v18-2717.full)
"""
struct ExponentialHawkesProcess{T, G} <: UnivariateProcess{T}
    t0::T
    tf::T
    λ₀::T
    a::T
    δ::T
    dylaw::G
    function ExponentialHawkesProcess(t0::T, tf::T, λ₀::T, a::T, δ::T, dylaw::G) where {T, G}
        ( λ₀ ≥ a ≥ zero(T)) || error("Parameters must satisfy `λ₀ ≥ a ≥ 0`")
        dylaw isa Distribution && support(dylaw).lb ≥ zero(T) || error("Distribution `dylaw` must be nonnegative")
        δ > 0 || error("Decay rate `δ` must be positive")
        tf > t0 || error("Final time `tf` must be greater than initial time `t0`")
        new{T, G}(t0, tf, λ₀, a, δ, dylaw)
    end
end

function Random.rand!(rng::AbstractRNG, noise::ExponentialHawkesProcess{T}, yt::AbstractVector{T}) where {T}
    δ = noise.δ
    a = noise.a
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    ti1 = noise.t0
    ti = ti1 + dt
    ni1 = firstindex(yt)
    yt[ni1] = noise.λ₀ # initial intensity
    t = ti1 # starting arrival
    d = 1 + δ * log( rand(rng) ) / (yt[ni1] - a)
    s = -log(rand(rng)) / a
    t += d ≤ 0.0 ? s : min(s, -log(d) / δ)
    for ni in Iterators.drop(eachindex(yt), 1)
        yt[ni] = (yt[ni1] - a) * exp( -δ * min(dt, t - ti1) ) + a # update decay to time t or ti = ti1 + dt, whichever is first
        while t ≤ ti
            yt[ni] += rand(rng, noise.dylaw) 
            d = 1 + δ * log( rand(rng) ) / (yt[ni] - a)
            s = -log(rand(rng)) / a
            r = d ≤ 0.0 ? s : min(s, -log(d) / δ)
            yt[ni] = (yt[ni] - a) * exp( -δ * min(ti - t, r) ) + a # update decay to time t + r or ti, whichever is first
            t += r
        end
        ni1 = ni
        ti1 = ti
        ti += dt
    end
end

"""
    TransportProcess(t0, tf, ylaw, f, d)

Construct a transport process on the time interval `t0` to `tf`, with function `f=f(t, y)` where `y` is a random vector with dimension `d` and distribution law for each coordinate given by `ylaw`.

The noise process `noise = TransportProcess(t0, tf, ylaw, f, d)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.

Each random sample path is obtained by first drawing `d` realizations of the distribution `ylaw` to build the sample value `y` and then defining the sample path by `Y_t = f(t, y)` for each `t` in the time mesh.

The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`.
"""
struct TransportProcess{T, F, G, N} <: UnivariateProcess{T}
    t0::T
    tf::T
    ylaw::G
    f::F
    rv::Array{T, N}
    function TransportProcess(t0::T, tf::T, ylaw::G, f::F, d::Int64) where {T, F, G}
        N = ylaw isa UnivariateDistribution ? 1 : 2
        rv = ylaw isa UnivariateDistribution ? zeros(T, d) : zeros(T, length(ylaw), d)
        new{T, F, G, N}(t0, tf, ylaw, f, rv)
    end
end

function Random.rand!(rng::AbstractRNG, noise::TransportProcess{T}, yt::AbstractVector{T}) where {T}
    n = length(yt)
    dt = (noise.tf - noise.t0) / (n - 1)
    # Most distributions don't allocate but Beta does (see https://github.com/JuliaStats/Distributions.jl/pull/1281), so beware of that, or thing about getting back with the rooled out explicit loop.
    rand!(rng, noise.ylaw, noise.rv)
    t = noise.t0 - dt
    for j in eachindex(yt)
        t += dt
        @inbounds yt[j] = noise.f(t, noise.rv)
    end
end

"""
    FractionalBrownianMotionProcess(t0, tf, y0, hurst, d; flags=FFTW.MEASURE)

Construct a fractional Brownian motion process on the interval `t0` to `tf`, with initial condition `y0`, Hurst parameter `hurst` and length up to `d`.

The noise process `noise = FractionalBrownianMotionProcess(t0, tf, y0, hurst, d; flags=FFTW.MEASURE)` returned by the constructor is a subtype of `AbstractNoise{Univariate}`.

Sample paths are obtained by populating a pre-allocated vector `yt` with the sample path, via `rand!(rng, noise, yt)`.
    
The number of steps for the sample path is determined by the length of the given vector `yt`, and the time steps are uniform and calculated according to `dt = (tf - t0) / (length(yt) - 1)`. The initial condition is `yt[1] = y0`, corresponding to the value at time `t0`. The length of `yt` must be smaller than or equal to the length `d` given in the constructor and used for the pre-allocation of the auxiliary vectors.

The method implemented is the one developed by Davies and Harte and uses an FFT transform to drop the order of complexity to O(N log N). For the transform, we use `FFTW.jl`, and use the flag `flags=FFTW.MEASURE` for generating the plans. Other common flags can be passed instead.

This implementation of fractional Brownian motion via Davies-Harte method follows [Dieker, T. (2004) Simulation of Fractional Brownian Motion. MSc Theses, University of Twente, Amsterdam](http://www.columbia.edu/~ad3217/fbm/thesis.pdf) and [A. B. Dieker and M. Mandjes, On spectral simulation of fractional Brownian motion, Probability in the Engineering and Informational Sciences, 17 (2003), 417-434](https://www.semanticscholar.org/paper/ON-SPECTRAL-SIMULATION-OF-FRACTIONAL-BROWNIAN-Dieker-Mandjes/b2d0d6a3d7553ae67a9f6bf0bbe21740b0914163)
"""
struct FractionalBrownianMotionProcess{P1, P2} <: UnivariateProcess{Float64}
    t0::Float64
    tf::Float64
    y0::Float64
    hurst::Float64
    d::Int
    cache_real::Vector{Float64}
    cache_complex::Vector{ComplexF64}
    cache_complex2::Vector{ComplexF64}
    plan_inverse::P1
    plan_direct::P2

    function FractionalBrownianMotionProcess(t0::Float64, tf::Float64, y0::Float64, hurst::Float64, d::Int; flags=FFTW.MEASURE)
        ispow2(d) || throw(
            ArgumentError(
                "Desired maximum length `d` must be a power of 2 for this implementation of the Davies-Harte method."
            )
        )
        0.0 < hurst < 1.0 || throw(
            ArgumentError(
                "Hurst parameter should be strictly between 0.0 and 1.0."
            )
        )

        cache_real = Vector{Float64}(undef, 2d)
        cache_complex = Vector{ComplexF64}(undef, 2d)
        cache_complex2 = Vector{ComplexF64}(undef, 2d)
        plan_inverse = plan_ifft(cache_real; flags)
        plan_direct = plan_fft(cache_complex; flags)

        new{typeof(plan_inverse), typeof(plan_direct)}(t0, tf, y0, hurst, d, cache_real, cache_complex, cache_complex2, plan_inverse, plan_direct)
    end
end

function Random.rand!(rng::AbstractRNG, noise::FractionalBrownianMotionProcess, yt::AbstractVector{Float64})

    length(yt) -1 ≤ noise.d || throw(
        ArgumentError(
            "length of the sample path vector should be at most that given in the construction of the fBm noise process."
        )
    )

    t0 = noise.t0
    tf = noise.tf
    H = noise.hurst
    y0 = noise.y0
    d = noise.d
    plan_inverse = noise.plan_inverse
    plan_direct = noise.plan_direct

    # covariance function in Dieker eq. (1.7)
    gamma = (k, H) -> 0.5 * (abs(k-1)^(2H) + abs(k+1)^(2H)) - abs(k)^(2H)

    # the first row of the circulant matrix in Dieker eq. (2.9)
    noise.cache_complex[1] = 1.0
    noise.cache_complex[d+1] = 0.0
    for k in 1:d-1
        noise.cache_complex[2d-k+1] = noise.cache_complex[k+1] = gamma(k, H)
    end

    # square-root of eigenvalues as in Dieker eq. (2.10) - using FFTW
    mul!(noise.cache_complex2, plan_inverse, noise.cache_complex)
    map!(r -> sqrt(2d * real(r)), noise.cache_real, noise.cache_complex2)

    # generate Wⱼ according to step 2 in Dieker pages 16-17
    noise.cache_complex[1] = randn(rng)
    noise.cache_complex[d+1] = randn(rng)
    for j in 2:d
        v1 = randn(rng)
        v2 = randn(rng)
        noise.cache_complex[j] = (v1 + im * v2) / √2
        noise.cache_complex[2d-j+2] = (v1 - im * v2) / √2
    end

    # multiply Wⱼ by √λⱼ to prep for DFT
    noise.cache_complex .*= noise.cache_real

    # Discrete Fourier transform of √λⱼ Wⱼ according to Dieker eq. (2.12) via FFTW
    mul!(noise.cache_complex2, plan_direct, noise.cache_complex)
    noise.cache_complex2 ./= √(2d)

    map!(real, noise.cache_real, noise.cache_complex2)

    # Rescale from [0, N] to [0, T]
    noise.cache_real .*= ((tf - t0)/d)^(H)

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