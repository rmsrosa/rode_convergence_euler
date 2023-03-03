"""
    ConvergenceSuite(t0, tf, x0law, f, noise, target!, method!, ntgt, ns, m)

Gather the data needed for computing the convergence error for a given RODE.
```math
    \\begin{cases}
        \\frac{\\mathrm{d}X_t}{\\mathrm{d}t} = f(t, X_t, Y_t), & t_0 \\leq t \\leq t_f \\\\
        X_{t_0} = X_0.
    \\end{cases}
```

The data comprises of the following:
* the initial and final times `t0` and `tf`;
* the univariate or multivariate distribution `x0law` for the initial condition ``X_0``;
* the right-hand-side term `f` for the equation, either in the out-of-place form `f=f(t, x, y)`, for a scalar equation (i.e. with a univariate initial condition `x0law`), or in the in-place form `f=f(dx, t, x, y)`, for a system of equations (i.e. with a multivariate initial condition `x0law`);
* the univariate or multivariate process `noise` for the noise term ``Y_t``;
* the function `target!` to compute the target solution for the error calculation via `target!(rng, xt, t0, tf, x0, f, yt)`;
* the function `method!` with the given method to approximate the solution, typically the [`euler_method!`](@ref), also in the form `method!(rng, xt, t0, tf, x0, f, yt)`;
* the number `ntgt` of mesh points in the fine mesh on which the target solution will be computed;
* the vector `ns` with a list of numbers of mesh points to compute the approximate solutions;
* the number `m` of sample paths to be computed for estimating the strong error via Monte Carlo method.

Besides these data obtained from the supplied arguments, a few cache vectors or matrices are created:
* a vector or matrix `yt` to hold the sample paths of the noise on the finest mesh, with length or row-length being `ntgt` and the shape depending on whether the noise is univariate or multivariate;
* a vector or matrix `xt` to hold the sample paths of the target solution, on the finest mesh, with length or row-length being `ntgt` and the shape depending on whether the law for the initial condition being univariate or multivariate;
* a vector or matrix `xnt` to hold the sample paths of the approximate solution, with length or row-length being the maximum of those in `ns` and the shape depending on whether the law for the initial condition being univariate or multivariate.
* ``

The actual error is obtained by solving a ConvergenceSuite via [`solve(rng, suite)`](@ref), with a given RNG.
"""
struct ConvergenceSuite{T, D, P, F1, F2, F3}
    t0::T
    tf::T
    x0law::D
    f::F1
    noise::P
    target!::F2
    method!::F3
    ntgt::Int
    ns::Vector{Int}
    m::Int
    yt::VecOrMat{T} # cache
    xt::VecOrMat{T} # cache
    xnt::VecOrMat{T} # cache
    function ConvergenceSuite(t0::T, tf::T, x0law::D, f::F1, noise::P, target!::F2, method::F3, ntgt::Int, ns::Vector{Int}, m::Int) where {T, D, P, F1, F2, F3}
        ( ntgt > 0 && all(>(0), ns) ) || error(
            "`ntgt` and `ns` arguments must be positive integers."

        )
        all(mod(ntgt, n) == 0 for n in ns) || error(
            "The length of `ntgt` should be divisible by any of the lengths in `ns`"
        )
    
        if D <: ContinuousUnivariateDistribution
            xt = Vector{Float64}(undef, ntgt)
            xnt = Vector{Float64}(undef, last(ns))
        elseif D <: ContinuousMultivariateDistribution
            nx = length(x0law)
            xt = Matrix{Float64}(undef, ntgt, nx)
            xnt = Matrix{Float64}(undef, last(ns), nx)
        else
            error(
                "`xlaw` should be either `ContinuousUnivariateDistribution` or `ContinuousMultivariateDistribution`."
            )
        end
        if P <: UnivariateProcess
            yt = Vector{Float64}(undef, ntgt)
        elseif P <: MultivariateProcess
            yt = Matrix{Float64}(undef, ntgt, length(noise))
        else
            error(
                "`noise` should be either a `UnivariateProcess` or a `MultivariateProcess`."
            )
        end

        return new{T, D, P, F1, F2, F3}(t0, tf, x0law, f, noise, target!, method, ntgt, ns, m, yt, xt, xnt)
    end
end

struct ConvergenceResults{T, S}
    suite::S
    deltas::Vector{T}
    trajerrors::Matrix{T}
    errors::Vector{T}
    lc::T
    p::T
    vandermonde::Matrix{T} # cache
    logerrors::Vector{T} # cache
end

function init(suite::ConvergenceSuite{T}) where {T}
    t0 = suite.t0
    tf = suite.tf
    ns = suite.ns

    deltas = (tf - t0) ./ (ns .- 1)
    vandermonde = [one.(deltas) log.(deltas)]
    trajerrors = zeros(last(ns), length(ns))

    errors = Vector{T}(undef, last(ns))
    logerrors = Vector{T}(undef, last(ns))
    lc = zero(T)
    p = zero(T)

    results = ConvergenceResults(suite, deltas, trajerrors, errors, lc, p, logerrors, vandermonde)

    return results
end

function calculate_trajerrors!(rng, trajerrors::Matrix{T}, suite::ConvergenceSuite{T, D, P}) where {T, D, P}
    t0 = suite.t0
    tf = suite.tf
    x0law = suite.x0law
    f = suite.f
    noise = suite.noise
    target! = suite.target!
    method! = suite.method!
    ntgt = suite.ntgt
    ns = suite.ns
    m = suite.m
    yt = suite.yt
    xt = suite.xt
    xnt = suite.xnt

    for _ in 1:m
        # draw initial condition
        if D <: ContinuousUnivariateDistribution
            xt[1] = rand(rng, x0law)
        else
            rand!(rng, x0law, view(xt, 1, :)) 
        end

        # generate noise sample path
        rand!(rng, noise, yt)

        # generate target path
        if D <: ContinuousUnivariateDistribution
            target!(rng, xt, t0, tf, xt[1], f, yt)
        else
            target!(rng, xt, t0, tf, view(xt, 1, :), f, yt)            
        end

        # solve approximate solutions at selected time steps and update strong errors
        #for (i, (nstep, nsi)) in enumerate(zip(nsteps, ns))
        for (i, nsi) in enumerate(ns)

            nstep = div(ntgt, nsi)

            if D <: ContinuousUnivariateDistribution && P <: UnivariateProcess
                solve_euler!(rng, view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1)))
            elseif D <: ContinuousUnivariateDistribution
                solve_euler!(rng, view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1), :))
            elseif P <: UnivariateProcess
                solve_euler!(rng, view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1)))
            else
                solve_euler!(rng, view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1), :))
            end

            for n in 2:nsi
                if D <: ContinuousUnivariateDistribution
                    trajerrors[n, i] += abs(xnt[n] - xt[1 + (n-1) * nstep])
                else
                    for j in eachindex(axes(xnt, 2))
                        trajerrors[n, i] += abs(xnt[n, j] - xt[1 + (n-1) * nstep, j])
                    end
                end
            end
        end
    end

    # normalize trajectory errors
    trajerrors ./= m

    return nothing
end

function solve(rng::AbstractRNG, suite::ConvergenceSuite{T}) where {T}

    trajerrors = zeros(T, last(suite.ns), length(suite.ns))

    calculate_trajerrors!(rng, trajerrors, suite)
    
    errors = maximum(trajerrors, dims=1)[1, :]
    deltas = (suite.tf - suite.t0) ./ (suite.ns .- 1)

    # fit to errors ∼ C Δtᵖ with lc = ln(C)
    vandermonde = [one.(deltas) log.(deltas)]
    logerrors = log.(errors)
    lc, p =  vandermonde \ logerrors

    # return results as `ConvergenceResults`
    results = ConvergenceResults(suite, deltas, trajerrors, errors, lc, p, vandermonde, logerrors)
    return results
end
