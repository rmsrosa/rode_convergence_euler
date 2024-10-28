"""
    RandomHeun(T::DataType=Float64, n::Int=0)

Instantiate a `RandomHeun` method including three cache vectors of length `n` for a non-allocating solver via the Heun method, solved by `solve!(xt, t0, tf, x0, f, yt, params, ::RandomHeun))`.

Set `n` to `0` when solving a scalar equation and set `n` to the length of the initial vector when solving a system of equations.
"""
struct RandomHeun{T, N} <: RODEMethod{N}
    cachex1::Vector{T}
    cachef1::Vector{T}
    cachef::Vector{T}
    function RandomHeun(T::DataType, n::Int)
        n ≥ 0 || error(
            "size must be non-negative"
        )
        cachex1 = Vector{T}(undef, n)
        cachef1 = Vector{T}(undef, n)
        cachef = Vector{T}(undef, n)
        N = n == 0 ? Univariate : Multivariate
        return new{T, N}(cachex1, cachef1, cachef)
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

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractMatrix{T}, params::Q, ::RandomHeun{T, Univariate}) where {T, F, Q}
    # scalar solution, vector noise
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The vector `xt` and the rows of the matrix `yt` must match indices")
    )
    N = length(xt) - 1 # mesh intervals
    dt = (tf - t0) / N
    i1 = firstindex(xt)
    xt[i1] = x0
    ti1 = t0
    for i in Iterators.drop(eachindex(xt), 1)
        fn1 = f(ti1, xt[i1], view(yt, i1, :), params)
        xtnaux = xt[i1] + dt * fn1
        ti1 += dt
        xt[i] = xt[i1] + dt * (fn1 + f(ti1, xtnaux, view(yt, i, :), params)) / 2
        i1 = i
    end
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractVector{T}, params::Q, method::RandomHeun{T, Multivariate}) where {T, F, Q}
    # vector solution, scalar noise
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The rows of the matrix `xt` and the vector `yt` must match indices")
    )
    axes(xt, 2) == axes(x0, 1) || throw(
        ArgumentError(
            "Column of `xt` and `x0` must match indices."
        )
    )
    n = size(xt, 1) - 1 # mesh intervals
    dt = (tf - t0) / n
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0
    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)
        f(method.cachef1, ti1, view(xt, i1, :), yt[i1], params)
        for j in eachindex(axes(xt, 2), axes(method.cachex1, 1))
            @inbounds method.cachex1[j] = xt[i1, j] + dt * method.cachef1[j]
        end
        ti1 += dt
        f(method.cachef, ti1, method.cachex1, yt[i], params)
        for j in eachindex(axes(xt, 2), axes(method.cachef1, 1), axes(method.cachef, 1))
            @inbounds xt[i, j] = xt[i1, j] + dt * ( method.cachef1[j] + method.cachef[j] ) / 2
        end
        i1 = i
    end
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractMatrix{T}, params::Q, method::RandomHeun{T, Multivariate}) where {T, F, Q}
    # vector solution, vector noise
    axes(xt, 1) == axes(yt, 1) || error(
        "The rows of the matrices `xt` and `yt` must match indices"
    )
    axes(xt, 2) == axes(x0, 1) || error(
        "Columns of `xt` and `x0` must match indices"
    )
    cacheaxis = axes(method.cachex1, 1)
    ( cacheaxis isa Base.OneTo{Int64} && axes(method.cachef1, 1) == cacheaxis && axes(method.cachef, 1) == cacheaxis ) || error(
        "All cache vectors should have the same length."
    )
    ( axes(xt, 2) isa Base.OneTo{Int64} && size(method.cachex1, 1) ≥ size(xt, 2)) || error(
        "row-length of the cache vectors should be greater than or equal to the column-size of `xt`"
    )
    n = size(xt, 1) - 1 # mesh intervals
    dt = (tf - t0) / n
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0

    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)


        f(method.cachef1, ti1, view(xt, i1, :), view(yt, i1, :), params)
        for j in eachindex(axes(xt, 2), axes(method.cachex1, 1))
            @inbounds method.cachex1[j] = xt[i1, j] + dt * method.cachef1[j]
        end
        ti1 += dt
        f(method.cachef, ti1, method.cachex1, view(yt, i, :), params)
        for j in eachindex(axes(xt, 2), axes(method.cachef1, 1), axes(method.cachef, 1))
            @inbounds xt[i, j] = xt[i1, j] + dt * ( method.cachef1[j] + method.cachef[j] ) / 2
        end
        i1 = i
    end
end