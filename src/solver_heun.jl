"""
    RandomHeun(T::DataType=Float64, n::Int=0)

Instantiate a `RandomHeun` method including two cache vectors of length `n` for a non-allocating solver via the Heun method (see [solve!(xt, t0, tf, x0, f::F, yt::Vector{T}, ::RandomHeun))](@ref)).

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
function solve!(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, ::RandomHeun{T, Univariate}) where {T, F}
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
