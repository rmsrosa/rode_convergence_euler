struct ConvergenceSuite{T, N0, NY, F1, F2, F3}
    t0::T
    tf::T
    x0law::ContinuousDistribution{N0}
    f::F1
    noise::AbstractProcess{NY}
    target!::F2
    method!::F3
    ntgt::Int
    ns::Vector{Int}
    m::Int
    yt::VecOrMat{T}
    xt::VecOrMat{T}
    xnt::VecOrMat{T}
    function ConvergenceSuite(t0::T, tf::T, x0law::ContinuousDistribution{N0}, f::F1, noise::AbstractProcess{NY}, target!::F2, method::F3, ntgt::Int, ns::Vector{Int}, m::Int) where {T, N0, NY, F1, F2, F3}
        ( ntgt > 0 && all(>(0), ns) ) || throw(
            ArgumentError(
                "`ntgt` and `ns` arguments must be positive integers."
            )
        )
        all(mod(ntgt, n) == 0 for n in ns) || throw(
            ArgumentError(
                "The length `ntgt = $ntgt` should be divisible by any of the lengths in `ns=$ns`"
            )
        )
    
        if N0 == Univariate
            xt = Vector{Float64}(undef, ntgt)
            xnt = Vector{Float64}(undef, last(ns))
        elseif N0 == Multivariate
            nx = length(x0law)
            xt = Matrix{Float64}(undef, ntgt, nx)
            xnt = Matrix{Float64}(undef, last(ns), nx)
        else
            throw(
                ArgumentError(
                    "`xlaw` should be either `ContinuousUnivariateDistribution` or `ContinuousMultivariateDistribution`."
                )
            )
        end
        if NY == Univariate
            yt = Vector{Float64}(undef, ntgt)
        elseif NY == Multivariate
            yt = Matrix{Float64}(undef, ntgt, length(noise))
        else
            throw(
                ArgumentError(
                    "`noise` should be either a `UnivariateProcess` or a `MultivariateProcess`."
                )
            )
        end

        return new{T, N0, NY, F1, F2, F3}(t0, tf, x0law, f, noise, target!, method, ntgt, ns, m, yt, xt, xnt)
    end
end

struct ConvergenceResults{T, S}
    suite::S
    deltas::Vector{T}
    trajerrors::Matrix{T}
    errors::Vector{T}
    lc::T
    p::T
end

function solve!(rng, suite::ConvergenceSuite)
    nsteps = div.(suite.ntgt, suite.ns)
    deltas = (suite.tf - suite.t0) ./ (suite.ns .- 1)
    trajerrors = zeros(last(suite.ns), length(suite.ns))

    calculate_errors!(rng, trajerrors, suite.yt, suite.xt, suite.xnt, suite.x0law, suite.f, suite.noise, suite.target!, suite.m, suite.t0, suite.tf, suite.ns, nsteps, deltas)

    errors = maximum(trajerrors, dims=1)[1, :]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    results = ConvergenceResults(suite, deltas, trajerrors, errors, lc, p)
    return results
end