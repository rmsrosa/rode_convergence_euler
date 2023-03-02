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
    nsteps::Vector{Int}
    deltas::Vector{T}
    trajerrors::Matrix{T}
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
    
        nsteps = div.(ntgt, ns)
    
        deltas = Vector{Float64}(undef, length(ns))
        trajerrors = zeros(last(ns), length(ns))
    
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

        return new{T, N0, NY, F1, F2, F3}(t0, tf, x0law, f, noise, target!, method, ntgt, ns, m, nsteps, deltas, trajerrors, yt, xt, xnt)
    end
end
