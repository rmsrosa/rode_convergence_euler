"""
    RODEMethod{N}

Abstract supertype for the methods for solving a Random ODE.
"""
abstract type RODEMethod{N} end

"""
    CustomMethod{F, P, N} <: RODEMethod{N}

Custom method for solving a Random ODE. It has two fields:

* `solver`: a function `solver(xt, t0, tf, x0, f, yt, params)` to solve, on the interval `t0` to `tf`, a Random ODE with right hand side `f`, "noise" sample path `yt`, initial condition `x0` and extra paramters `params`;
* `params`: any argument or series of arguments necessary for the custom solver.

Aliases:

* `CustomUnivariateMethod{F, P} = CustomMethod{F, P, Univariate}`
* `CustomMultivariateMethod{F, P} = CustomMethod{F, P, Multivariate}`
"""
struct CustomMethod{F, P, N} <: RODEMethod{N}
    solver::F
    params::P
end

const CustomUnivariateMethod{F, P} = CustomMethod{F, P, Univariate} where {F, P}

const CustomMultivariateMethod{F, P} = CustomMethod{F, P, Multivariate} where {F, P}

CustomMethod{N}(solver::F, params::P) where {F, P, N} = CustomMethod{F, P, N}(solver, params)

CustomUnivariateMethod(solver::F, params::P) where {F, P} = CustomMethod{F, P, Univariate}(solver, params)

CustomMultivariateMethod(solver::F, params::P) where {F, P} = CustomMethod{F, P, Multivariate}(solver, params)

"""
    solve!(xt, t0, tf, x0, f, yt, method)

Solve a random ODE with the provided `method`.

More precisely, sove, inplace, (a sample path of) the (R)ODE `dx_t/dt = f(t, x_t, y_t),` for an unknown `x_t` and a given (noise path) `y_t`, with the following arguments:

* a function `f(t, x, y)`, if `x` is a scalar, or `f(dx, t, x, y)`, if `x` is a vector;
* a scalar or vector initial condition `x0`;
* a time interval `t0` to `tf`;
* a sample path `yt` of a "noise", either a vector (for scalar noise) or a matrix (for vectorial noise).
* a numerical `method`, either `RandomEuler()` for a scalar equation, `RandomEuler(n)` for an n-dimensional system of equations, or `RandomHeun()` for a scalar equation.

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length `n` of the vector `xt` via `dt = (tf - t0) / (n - 1)`.

The noise `yt` should be of the same (row) length as `xt`.
"""
function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}, method::CustomUnivariateMethod) where {T, F}
    method.solver(xt, t0, tf, x0, f, yt, method.params)
end

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractMatrix{T}, method::CustomUnivariateMethod) where {T, F}
    method.solver(xt, t0, tf, x0, f, yt, method.params)
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractVector{T}, method::CustomMultivariateMethod) where {T, F}
    method.solver(xt, t0, tf, x0, f, yt, method.params)
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractMatrix{T}, method::CustomMultivariateMethod) where {T, F}
    method.solver(xt, t0, tf, x0, f, yt, method.params)
end