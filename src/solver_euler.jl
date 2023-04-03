"""
    RandomEuler(T::DataType=Float64, n::Int=0)

Instantiate a `RandomEuler` method including a cache vector of length `n` for a non-allocating solver via the Euler method, solved by `solve!(xt, t0, tf, x0, f::F, yt::AbstractVector{T}, ::RandomEuler))]`

Set `n` to `0` for solving a scalar equation and set `n` to the length of the system (e.g. the length of the initial condition).
"""
struct RandomEuler{T, N} <: RODEMethod{N}
    cachex::Vector{T}
    function RandomEuler(T::DataType, n::Int)
        n ≥ 0 || error(
            "size must be non-negative"
        )
        cachex = Vector{T}(undef, n)
        N = n == 0 ? Univariate : Multivariate
        return new{T, N}(cachex)
    end
end

RandomEuler(T::DataType) = RandomEuler(T, 0)
RandomEuler(n::Int) = RandomEuler(Float64, n)
RandomEuler() = RandomEuler(Float64, 0)

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}, ::RandomEuler{T, Univariate}) where {T, F}
    # scalar solution, scalar noise
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )

    N = length(xt)
    dt = (tf - t0) / (N - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    ti1 = t0
    for i in Iterators.drop(eachindex(xt, yt), 1)
        xt[i] = xt[i1] + dt * f(ti1, xt[i1], yt[i1])
        i1 = i
        ti1 += dt
    end
end

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractMatrix{T}, ::RandomEuler{T, Univariate}) where {T, F}
    # scalar solution, vector noise
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The vector `xt` and the rows of the matrix `yt` must match indices")
    )
    n = length(xt)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    ti1 = t0
    for i in Iterators.drop(eachindex(xt), 1)
        xt[i] = xt[i1] + dt * f(ti1, xt[i1], view(yt, i1, :))
        i1 = i
        ti1 += dt
    end
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractVector{T}, method::RandomEuler{T, Multivariate}) where {T, F}
    # vector solution, scalar noise
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The rows of the matrix `xt` and the vector `yt` must match indices")
    )
    axes(xt, 2) == axes(x0, 1) || throw(
        ArgumentError(
            "Column of `xt` and `x0` must match indices."
        )
    )
    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0
    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)
        f(method.cachex, ti1, view(xt, i-1, :), yt[i-1])
        for j in eachindex(axes(xt, 2), axes(method.cachex, 1))
            @inbounds xt[i, j] = xt[i-1, j] + dt * method.cachex[j]
        end
        i1 = i
        ti1 += dt
    end
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractMatrix{T}, method::RandomEuler{T, Multivariate}) where {T, F}
    # vector solution, vector noise
    axes(xt, 1) == axes(yt, 1) || error(
        "The rows of the matrices `xt` and `yt` must match indices"
    )
    axes(xt, 2) == axes(x0, 1) || error(
        "Columns of `xt` and `x0` must match indices"
    )
    ( axes(xt, 2) isa Base.OneTo{Int64} && axes(method.cachex, 1) isa Base.OneTo{Int64} && size(method.cachex, 1) ≥ size(xt, 2)) || error(
        "row-length of the cache vector `method.cachex` should be greater than or equal to the column-size of `xt`"
    )
    axes(xt, 2)
    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0
    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)
        f(method.cachex, ti1, view(xt, i-1, :), view(yt, i-1, :))
        for j in eachindex(axes(xt, 2))
            @inbounds xt[i, j] = xt[i-1, j] + dt * method.cachex[j]
        end
        i1 = i
        ti1 += dt
    end
end
