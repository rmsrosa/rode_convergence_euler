"""
    solve_euler!(rng, xt, t0, tf, x0, f, yt)

Solve inplace, via Euler method, (a sample path of) the (R)ODE `dx_t/dt = f(t, x_t, y_t),` with the following data:

* function `f=f(t, x, y)`;
* initial condition `x0`;
* time interval `t0` to `tf`;
* a sample path `yt` of a "noise".

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `N` of the vector `xt` via `dt = (tf - t0) / (N - 1)`.

The noise vector `yt` is expected to be at least of the same length as `xt`.
"""
function solve_euler!(rng::AbstractRNG, xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}) where {T, F}
    N = length(yt)
    length(xt) ≥ N || throw(
        ArgumentError(
            "Length of `xt` should be at least that of `yt` but got $(length(xt)) and $N"
        )
    )
    dt = (tf - t0) / (N - 1)
    xt[1] = x0
    tn1 = t0
    for n in 2:N
        xt[n] = xt[n-1] + dt * f(tn1, xt[n-1], yt[n-1])
        tn1 += dt
    end
end

function solve_euler!(rng::AbstractRNG, xt::Matrix{T}, t0::T, tf::T, x0::Vector{T}, f::F, yt::Vector{T}) where {T, F}
    N = length(yt)
    size(xt, 1) ≥ N || throw(
        ArgumentError(
            "Row length of `xt` should be at least that of `yt` but got $(size(xt,1)) and $N"
        )
    )
    size(xt, 2) == length(x0) || throw(
        ArgumentError(
            "Column length of `xt` should the same as that of `x0` but got $(size(xt,2)) and $(length(x0))"
        )
    )
    dt = (tf - t0) / (N - 1)
    xt[1] .= x0
    tn1 = t0
    for n in 2:N
        # use row of xt as an auxiliary cache variable for dX
        f(view(xt, n, :), tn1, view(xt, n-1, :), yt[n-1])
        xt[n] .= xt[n-1, :] .+ dt * xt[N, :]
        tn1 += dt
    end
end

"""
    solve_heun!(rng, xt, t0, tf, x0, f, yt)

Solve inplace, via Heun method, (a sample path of) the (R)ODE `dx_t/dt = f(t, x_t, y_t),` with the following data:

* function `f=f(t, x, y)`;
* initial condition `x0`;
* time interval `t0` to `tf`;
* a sample path `yt` of a "noise".

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `N` of the vector `xt` via `dt = (tf - t0) / (N - 1)`.

The noise vector `yt` is expected to be at least of the same length as `xt`.
"""
function solve_heun!(rng::AbstractRNG, xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}) where {T, F}
    N = length(yt)
    length(xt) ≥ N || throw(
        ArgumentError(
            "Length of `xt` should be at least that of `yt` but got $(length(xt)) and $N"
        )
    )
    dt = (tf - t0) / (N - 1)
    xt[1] = x0
    tn1 = t0
    for n in 2:N
        fn1 = f(tn1, xt[n-1], yt[n-1])
        xtnaux = xt[n-1] + dt * fn1
        tn1 += dt
        xt[n] = xt[n-1] + dt * (fn1 + f(tn1, xtnaux, yt[n])) / 2
    end
end
