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

* CustomUnivariateMethod{F, P} = CustomMethod{F, P, Univariate}
* CustomMultivariateMethod{F, P} = CustomMethod{F, P, Multivariate}
"""
struct CustomMethod{F, P, N} <: RODEMethod{N}
    solver::F
    params::P
end

const CustomUnivariateMethod{F, P} = CustomMethod{F, P, Univariate}
const CustomMultivariateMethod{F, P} = CustomMethod{F, P, Multivariate}

CustomMethod{N}(solver::F, params::P) where {F, P, N} = CustomMethod{F, P, N}(solver, params)

CustomMethod(solver::F, params::P) where {F, P} = CustomMethod{F, P, Univariate}(solver, params)

CustomMethod(solver::F, params::P, n::Int) where {F, P} = CustomMethod{F, P, n > 0 ? Multivariate : Univariate}(solver, params)

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}, method::CustomMethod) where {T, F}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )
    method.solver(xt, t0, tf, x0, f, yt, method.params)
end
