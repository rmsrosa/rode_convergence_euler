"""
    RODEMethod{N}

Abstract supertype for the methods for solving a Random ODE.
"""
abstract type RODEMethod{N} end

"""
    CustomMethod{F, P, N} <: RODEMethod{N}

Custom method for solving a Random ODE. It has two fields:

* `solver`: a function `solver(xt, t0, tf, x0, f, params, yt, solver_params)` to solve, on the interval `t0` to `tf`, a Random ODE with right hand side `f`, "noise" sample path `yt`, initial condition `x0`, parameters `params` for the function `f`, and extra parameters `solver_params` for the custom solver;
* `solver_params`: any argument or series of arguments necessary for the custom solver.

Aliases:

* `CustomUnivariateMethod{F, P} = CustomMethod{F, P, Univariate}`
* `CustomMultivariateMethod{F, P} = CustomMethod{F, P, Multivariate}`
"""
struct CustomMethod{F, P, N} <: RODEMethod{N}
    solver::F
    solver_params::P
end

const CustomUnivariateMethod{F, P} = CustomMethod{F, P, Univariate} where {F, P}

const CustomMultivariateMethod{F, P} = CustomMethod{F, P, Multivariate} where {F, P}

CustomMethod{N}(solver::F, solver_params::P) where {F, P, N} = CustomMethod{F, P, N}(solver, solver_params)

CustomUnivariateMethod(solver::F, solver_params::P) where {F, P} = CustomMethod{F, P, Univariate}(solver, solver_params)

CustomMultivariateMethod(solver::F, solver_params::P) where {F, P} = CustomMethod{F, P, Multivariate}(solver, solver_params)

"""
    solve!(xt, t0, tf, x0, f, params, yt, method)

Solve a random ODE with the provided `method`.

More precisely, sove, inplace, a sample path of the (R)ODE
```math
    \\begin{cases}
        \\displaystyle\\frac{\\mathrm{d}X_t}{\\mathrm{d}t} = f(t, X_t, Y_t), & t_0 \\leq t \\leq t_f, \\\\
        X_{t_0} = X_0,
    \\end{cases}
```
with the following arguments:

* a function `f(t, x, y, params)`, if `x` is a scalar, or `f(dx, t, x, y, params)`, if `x` is a vector;
* a scalar or vector sample initial condition `x0`;
* a time interval `t0` to `tf`;
* a sample path `yt` of a "noise", either a vector (for scalar noise) or a matrix (for vectorial noise);
* parameters `params` for the function `f`;
* a numerical `method`, either `RandomEuler()` for a scalar equation, `RandomEuler(n)` for an n-dimensional system of equations, or `RandomHeun()` for a scalar equation.

The values of `xt` are updated with the computed solution values.

The time step is obtained from the length of the vector `xt` via `dt = (tf - t0) / (lenght(xt) - 1)`.

The noise `yt` should be of the same (row) length as `xt`.
"""
function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}, params::Q, method::CustomUnivariateMethod) where {T, F, Q}
    method.solver(xt, t0, tf, x0, f, yt, params, method.solver_params)
end

function solve!(xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractMatrix{T}, params::Q, method::CustomUnivariateMethod) where {T, F, Q}
    method.solver(xt, t0, tf, x0, f, yt, params, method.solver_params)
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractVector{T}, params::Q, method::CustomMultivariateMethod) where {T, F, Q}
    method.solver(xt, t0, tf, x0, f, yt, params, method.solver_params)
end

function solve!(xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractMatrix{T}, params::Q, method::CustomMultivariateMethod) where {T, F, Q}
    method.solver(xt, t0, tf, x0, f, yt, params, method.solver_params)
end