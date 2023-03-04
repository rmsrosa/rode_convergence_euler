abstract type RODEMethod{N1, N2} end

struct RandomEuler{T, N1, N2} <: RODEMethod{N1, N2}
    cachex::Vector{T}
    cachey::Vector{T}
    function RandomEuler(T::DataType, n::NTuple{2, Int})
        all(n .≥ 0) || error(
            "dimensions must be non-negative"
        )
        cachex = Vector{T}(undef, n[1])
        cachey = Vector{T}(undef, n[2])
        N1 = n[1] == 0 ? Univariate : Multivariate
        N2 = n[2] == 0 ? Univariate : Multivariate
        return new{T, N1, N2}(cachex, cachey)
    end
end

RandomEuler(T::DataType) = RandomEuler(T, (0, 0))
RandomEuler(T::DataType, n::Int...) = RandomEuler(T, n)
RandomEuler(n::NTuple{2, Int}) = RandomEuler(Float64, n)
RandomEuler(n::Int...) = RandomEuler(Float64, n)
RandomEuler() = RandomEuler(Float64, (0, 0))

struct RandomHeun{T, N1, N2} <: RODEMethod{N1, N2}
    cachex1::Vector{T}
    cachex2::Vector{T}
    cachey::Vector{T}
    function RandomHeun(T::DataType, n::NTuple{2, Int})
        all(n .≥ 0) || error(
            "dimensions must be non-negative"
        )
        cachex1 = Vector{T}(undef, n[1])
        cachex2 = Vector{T}(undef, n[1])
        cachey = Vector{T}(undef, n[2])
        N1 = n[1] == 0 ? Univariate : Multivariate
        N2 = n[2] == 0 ? Univariate : Multivariate
        return new{T, N1, N2}(cachex1, cachex2, cachey)
    end
end

"""
    solve!(xt, t0:, tf, x0, f, yt, method::RandomEuler)

Solve inplace, via Euler method, (a sample path of) the (R)ODE `dx_t/dt = f(t, x_t, y_t),` for an unknown `x_t` and a given (noise path) `y_t`, with the following arguments:

* a function `f(t, x, y)`, if `x` is a scalar, or `f(dx, t, x, y)`, if `x` is a vector;
* a scalar or vector initial condition `x0`;
* a time interval `t0` to `tf`;
* a sample path `yt` of a "noise", either a vector (for scalar noise) or a matrix (for vectorial noise).

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `n` of the vector `xt` via `dt = (tf - t0) / (n - 1)`.

The noise `yt` should be of the same (row) length as `xt`.
"""
# scalar solution, scalar noise
function solve!(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, ::RandomEuler{T, Univariate, Univariate}) where {T, F}
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

# scalar solution, vector noise
function solve!(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Matrix{T}, ::RandomEuler{T, Univariate, Multivariate}) where {T, F}
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

# vector solution, scalar noise
function solve!(xt::Matrix{T}, t0::T, tf::T, x0::Vector{T}, f::F, yt::Vector{T}, method::RandomEuler{T, Multivariate, Univariate}) where {T, F}
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

# vector solution, vector noise
function solve!(xt::Matrix{T}, t0::T, tf::T, x0::Vector{T}, f::F, yt::Matrix{T}, method::RandomEuler{T, Multivariate, Multivariate}) where {T, F}
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The rows of the matrices `xt` and `yt` must match indices")
    )
    axes(xt, 2) == axes(x0, 1) || throw(
        ArgumentError(
            "Columns of `xt` and `x0` must match indices"
        )
    )
    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0
    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)
        f(method.cachex, ti1, view(xt, i-1, :), view(yt, i-1, :))
        for j in eachindex(axes(xt, 2), axes(method.cachex, 1))
            @inbounds xt[i, j] = xt[i-1, j] + dt * method.cachex[j]
        end
        i1 = i
        ti1 += dt
    end
end

"""
    solve!(xt, t0, tf, x0, f, yt, method::RandomHeun)

Solve inplace, via Heun method, (a sample path of) the scalar (R)ODE `dx_t/dt = f(t, x_t, y_t),` with a given scalar noise `y_t`, with the following arguments:

* a function `f=f(t, x, y)`;
* a scalar initial condition `x0`;
* a time interval `t0` to `tf`;
* a vector sample path `yt` of a "noise".

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `n` of the vector `xt` via `dt = (tf - t0) / (n - 1)`.

The noise vector `yt` should be of the same length as `xt`.
"""
function solve!(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, ::RandomHeun{T, Univariate, Univariate}) where {T, F}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("vectors `xt` and `yt` must match indices")
    )
    n = length(xt)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    ti1 = t0
    for i in Iterators.drop(eachindex(xt, yt), 1)
        fn1 = f(ti1, xt[i1], yt[i1])
        xtnaux = xt[i1] + dt * fn1
        ti1 += dt
        xt[i] = xt[i1] + dt * (fn1 + f(ti1, xtnaux, yt[i])) / 2
        i1 = i
    end
end
