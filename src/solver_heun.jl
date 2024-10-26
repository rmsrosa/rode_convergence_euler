"""
    RandomHeun(T::DataType=Float64, n::Int=0)

Instantiate a `RandomHeun` method including two cache vectors of length `n` for a non-allocating solver via the Heun method, solved by `solve!(xt, t0, tf, x0, f, yt, params, ::RandomHeun))`.

Set `n` to `0` for solving a scalar equation and set `n` to the length of the system (e.g. the length of the initial condition).
"""
struct RandomHeun{T, N} <: RODEMethod{N}
    cachex1::Vector{T}
    cachex2::Vector{T}
    function RandomHeun(T::DataType, n::Int)
        n ≥ 0 || error(
            "size must be non-negative"
        )
        cachex1 = Vector{T}(undef, n)
        cachex2 = Vector{T}(undef, n)
        N = n == 0 ? Univariate : Multivariate
        return new{T, N}(cachex1, cachex2)
    end
end

RandomHeun(T::DataType) = RandomHeun(T, 0)
RandomHeun(n::Int) = RandomHeun(Float64, n)
RandomHeun() = RandomHeun(Float64, 0)

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}, params::Q, ::RandomHeun{T, Univariate}) where {T, F, Q}
    # scalar solution, scalar noise
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("vectors `xt` and `yt` must match indices")
    )
    N = length(xt) - 1 # mesh intervals
    dt = (tf - t0) / N
    i1 = firstindex(xt)
    xt[i1] = x0
    ti1 = t0
    for i in Iterators.drop(eachindex(xt, yt), 1)
        fn1 = f(ti1, xt[i1], yt[i1], params)
        xtnaux = xt[i1] + dt * fn1
        ti1 += dt
        xt[i] = xt[i1] + dt * (fn1 + f(ti1, xtnaux, yt[i], params)) / 2
        i1 = i
    end
end
