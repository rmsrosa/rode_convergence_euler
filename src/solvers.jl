"""
    solve_euler!(rng, xt, t0, tf, x0, f, yt)

Solve inplace, via Euler method, (a sample path of) the (R)ODE `dx_t/dt = f(t, x_t, y_t),` for an unknown `x_t` and a given (noise path) `y_t`, with the following arguments:

* a function `f(t, x, y)`, if `x` is a scalar, or `f(dx, t, x, y)`, if `x` is a vector;
* a scalar or vector initial condition `x0`;
* a time interval `t0` to `tf`;
* a sample path `yt` of a "noise", either an vector (for scalar noise) or a matrix (for vectorial noise).

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `n` of the vector `xt` via `dt = (tf - t0) / (n - 1)`.

The noise `yt` should be of the same (row) length as `xt`.
"""
# scalar unknown, scalar noise
function solve_euler!(rng::AbstractRNG, xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}) where {T, F}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices; got $(axes(xt)) and $(axes(yt)).")
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

# scalar unknown, vector noise
function solve_euler!(rng::AbstractRNG, xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractMatrix{T}) where {T, F}
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The vector `xt` and the rows of the matrix `yt` must match indices; got $(axes(xt, 1)) and $(axes(yt, 1)).")
    )
    n = length(xt)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    ti1 = t0
    for i in Iterators.drop(eachindex(xt), 1)
        xt[i] = xt[i1] + dt * f(ti1, xt[i1], yt[i1, :])
        i1 = i
        ti1 += dt
    end
end

# vector unknown, scalar noise
function solve_euler!(rng::AbstractRNG, xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractVector{T}) where {T, F}
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The rows of the matrix `xt` and the vector `yt` must match indices; got $(axes(xt, 1)) and $(axes(yt, 1)).")
    )
    axes(xt, 2) == axes(x0, 1) || throw(
        ArgumentError(
            "Column of `xt` and `x0` must match indices; got $(axes(xt, 2)) and $(axes(x0, 1))"
        )
    )
    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0
    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)
        # use row of xt as a temporary cache variable for dX
        f(view(xt, i, :), ti1, view(xt, i-1, :), yt[i-1])
        xt[i, :] .= view(xt, i-1, :) .+ dt * view(xt, i, :)
        i1 = i
        ti1 += dt
    end
end

# vector unknown, vector noise
function solve_euler!(rng::AbstractRNG, xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractMatrix{T}) where {T, F}
    axes(xt, 1) == axes(yt, 1) || throw(
        DimensionMismatch("The rows of the matrices `xt` and `yt` must match indices; got $(axes(xt, 1)) and $(axes(yt, 1)).")
    )
    axes(xt, 2) == axes(x0, 1) || throw(
        ArgumentError(
            "Columns of `xt` and `x0` must match indices; got $(axes(xt, 2)) and $(axes(x0, 1))"
        )
    )
    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(axes(xt, 1))
    xt[i1, :] .= x0
    ti1 = t0
    for i in Iterators.drop(eachindex(axes(xt, 1), axes(yt, 1)), 1)
        # use row of xt as a temporary cache variable for dX
        f(view(xt, i, :), ti1, view(xt, i-1, :), yt[i-1, :])
        xt[i, :] .= view(xt, i-1, :) .+ dt * view(xt, i, :)
        i1 = i
        ti1 += dt
    end
end

"""
    solve_heun!(rng, xt, t0, tf, x0, f, yt)

Solve inplace, via Heun method, (a sample path of) the scalar (R)ODE `dx_t/dt = f(t, x_t, y_t),` with a given scalar noise `y_t`, with the following arguments:

* a function `f=f(t, x, y)`;
* a scalar initial condition `x0`;
* a time interval `t0` to `tf`;
* a vector sample path `yt` of a "noise".

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `n` of the vector `xt` via `dt = (tf - t0) / (n - 1)`.

The noise vector `yt` should be of the same length as `xt`.
"""
function solve_heun!(rng::AbstractRNG, xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Union{Vector{T}, SubArray{T, 1, Vector{T}, Tuple{StepRange{Int64, Int64}}, true}}) where {T, F}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("vectors `xt` and `yt` must match indices; got $(axes(xt)) and $(axes(yt)).")
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
