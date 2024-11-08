"""
    ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m, ks)

Gather the data needed for computing the convergence error for a given RODE of the form
```math
    \\begin{cases}
        \\displaystyle\\frac{\\mathrm{d}X_t}{\\mathrm{d}t} = f(t, X_t, Y_t), & t_0 \\leq t \\leq t_f \\\\
        X_{t_0} = X_0.
    \\end{cases}
```

The data comprises of the following:

* the initial and final times `t0` and `tf`;
* the univariate or multivariate distribution `x0law` for the initial condition ``X_0``;
* the right-hand-side term `f` for the equation, either in the out-of-place form `f=f(t, x, y, params)`, for a scalar equation (i.e. with a univariate initial condition `x0law`), or in the in-place form `f=f(dx, t, x, y, params)`, for a system of equations (i.e. with a multivariate initial condition `x0law`);
* the univariate or multivariate process `noise` for the noise term ``Y_t``;
* parameters  `params` for the function `f`;
* the method `target` to compute the target solution for the error calculation via `solve!(xt, t0, tf, x0, f, yt, target)`, typically `EulerMethod()` with a much finer resolution with `ntgt` mesh points or the order of the square of the highest number of mesh points in `ns` (see below) or a lower resolution `CustomMethod()` with an exact distribution of the possible solutions conditioned to the already computed noise points;
* the `method` to approximate the solution, typically the `EulerMethod()`, also in the form `solve!(xt, t0, tf, x0, f, yt, method)`;
* the number `ntgt` of mesh points in the fine mesh on which the target solution will be computed;
* the vector `ns` with a list of numbers of mesh points to compute the approximate solutions;
* the number `m` of sample paths to be computed for estimating the strong error via Monte Carlo method.
* the range of steps `ks` to be used in case one approximates a Random PDE with an increasing number of spatial discretization points, so for each `n` in `ns`, one uses a range `begin:k:end` for the points in the spatial discretization, which defaults to `k=[1]` in the case of a scalar or of a genuine system of RODEs;

Besides these data obtained from the supplied arguments, a few cache vectors or matrices are created:
* a vector or matrix `yt` to hold the sample paths of the noise on the finest mesh, with length or row-length being `ntgt` and the shape depending on whether the noise is univariate or multivariate;
* a vector or matrix `xt` to hold the sample paths of the target solution, on the finest mesh, with length or row-length being `ntgt` and the shape depending on whether the law for the initial condition being univariate or multivariate;
* a vector or matrix `xnt` to hold the sample paths of the approximate solution, with length or row-length being the maximum of those in `ns` and the shape depending on whether the law for the initial condition being univariate or multivariate.

The actual error is obtained by solving a ConvergenceSuite via `solve(rng::AbstractRNG, suite::ConvergenceSuite{T})` with a given RNG.
"""
struct ConvergenceSuite{T, D, P, Q, F, N1, N2, M1, M2}
    t0::T
    tf::T
    x0law::D
    f::F
    noise::P
    params::Q
    target::M1
    method::M2
    ntgt::Int
    ns::Vector{Int}
    m::Int
    ks::Vector{Int}
    yt::Array{T, N1} # cache
    xt::Array{T, N2} # cache
    xnt::Array{T, N2} # cache
    function ConvergenceSuite(t0::T, tf::T, x0law::D, f::F, noise::P, params::Q, target::M1, method::M2, ntgt::Int, ns::Vector{Int}, m::Int, ks::Vector{Int}=ones(Int, length(ns))) where {T, D, P, Q, F, M1, M2}
        ( ntgt > 0 && m > 0 ) || error(
            "`ntgt` and `m` arguments must be positive integers."

        )
        ( all(>(0), ns) && all(>(0), ks) ) || error(
            "all elements in `ns` and `ks` must be positive integers."

        )
        ( length(ns) == length(ks) ) || error(
            "`ns` and `ks` must have same length"
        )
        all(mod(ntgt, n) == 0 for n in ns) || error(
            "The length of `ntgt` should be divisible by any of the lengths in `ns`"
        )
    
        ( M1 <: RODEMethod && M2 <: RODEMethod ) || error(
            "The `target` and `method` solver methods should be subtypes of `RODEMethod`"
        )
        if D <: UnivariateDistribution
            xt = Vector{T}(undef, ntgt + 1)
            xnt = Vector{T}(undef, last(ns) + 1)
            N2 = 1
        elseif D <: MultivariateDistribution
            nx = length(x0law)
            xt = Matrix{T}(undef, ntgt + 1, nx)
            xnt = Matrix{T}(undef, last(ns) + 1, nx)
            N2 = 2
        else
            error(
                "`x0law` should be either `UnivariateDistribution` or `MultivariateDistribution`."
            )
        end
        if P <: UnivariateProcess
            yt = Vector{T}(undef, ntgt + 1)
            N1 = 1
        elseif P <: MultivariateProcess
            yt = Matrix{T}(undef, ntgt + 1, length(noise))
            N1 = 2
        else
            error(
                "`noise` should be either a `UnivariateProcess` or a `MultivariateProcess`."
            )
        end

        return new{T, D, P, Q, F, N1, N2, M1, M2}(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m, ks, yt, xt, xnt)
    end
end

"""
    ConvergenceResult{T, S}(deltas::Vector{T}, trajerrors::Matrix{T}, trajstderrs::Matrix{T}, errors::Vector{T}, stderrs::Vector{T}, lc::T, p::T, pmin::T, pmax::T) where {T, S}

Stores the result of `solve(rng, suite)` with fields
* `deltas`: the time steps associated with the number of mesh points in the vector `suite.ns`;
* `trajerrors`: a matrix where each column corresponds to the strong error, along the trajectory, at each mesh resolution determined by `suite.ns`, i.e. `trajerrors[i, k]` is the error at time ``t_0 + i \\Delta t``, for the time step ``\\Delta t = (t_f - t_0) / (n - 1)`` associated with the kth element `n = suite.ns[k]`;
* `trajstderrs`: a matrix with the corresponding standard error for each entry in `trajerrors`;
* `errors`: the maximum, along the trajectory, of the `trajerrors`;
* `stderrs`: the corresponding standard error for the Monte-Carlo estimate of the strong `errors`;
* `lc`: the logarithm ``\\log(C)`` of the multiplicative constant in the fitted error `CΔtᵖ`;
* `p`: the estimated order of the strong convergence;
* `pmin` and `pmax`: the 95% confidence interval for `p`;
"""
struct ConvergenceResult{T}
    deltas::Vector{T}
    trajerrors::Matrix{T}
    trajstderrs::Matrix{T}
    errors::Vector{T}
    stderrs::Vector{T}
    lc::T
    p::T
    pmin::T
    pmax::T
end

"""
    calculate_trajerrors!(rng, trajerrors, trajstderrs, suite)

Calculate the strong error and the standard error of the suite at each time step along the trajectory.

The strong errors are stored in the provided `trajerrors` matrix, while the standar errors are stored in the provided `trajstderrs`. All the info is given in the ConvergenceSuite `suite`. The RNG seed is given in `rng`.

This method is used when solving a `ConvergenceSuite`.
"""
function calculate_trajerrors!(rng, trajerrors::Matrix{T}, trajstderrs::Matrix{T}, suite::ConvergenceSuite{T, D, P}) where {T, D, P}
    t0 = suite.t0
    tf = suite.tf
    x0law = suite.x0law
    f = suite.f
    noise = suite.noise
    params = suite.params
    target = suite.target
    method = suite.method
    ntgt = suite.ntgt
    ns = suite.ns
    m = suite.m
    ks = suite.ks
    yt = suite.yt
    xt = suite.xt
    xnt = suite.xnt

    for _ in 1:m
        # draw initial condition
        if D <: UnivariateDistribution
            xt[1] = rand(rng, x0law)
        else
            rand!(rng, x0law, view(xt, 1, :)) 
        end

        # generate noise sample path
        rand!(rng, noise, yt)

        # generate target path
        if D <: UnivariateDistribution
            solve!(xt, t0, tf, xt[1], f, yt, params, target)
        else
            solve!(xt, t0, tf, view(xt, 1, :), f, yt, params, target)            
        end

        # solve approximate solutions at selected time steps and update strong errors
        for (i, nsi) in enumerate(ns)

            nstep = div(ntgt, nsi)
            kstep = ks[i]

            if D <: UnivariateDistribution && P <: UnivariateProcess
                solve!(view(xnt, 1:nsi+1), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*nsi), params, method)
            elseif D <: UnivariateDistribution
                solve!(view(xnt, 1:nsi+1), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*nsi, :), params, method)
            elseif P <: UnivariateProcess
                solve!(view(xnt, 1:nsi+1, 1:kstep:size(xnt,2)), t0, tf, view(xt, 1, 1:kstep:size(xnt,2)), f, view(yt, 1:nstep:1+nstep*nsi), params, method)
            else
                solve!(view(xnt, 1:nsi+1, 1:kstep:size(xnt,2)), t0, tf, view(xt, 1, 1:kstep:size(xnt,2)), f, view(yt, 1:nstep:1+nstep*nsi, :), params, method)
            end

            for n in 2:nsi+1
                if D <: UnivariateDistribution
                    trajerrors[n, i] += abs(xnt[n] - xt[1 + (n-1) * nstep])
                    trajstderrs[n, i] += abs2(xnt[n] - xt[1 + (n-1) * nstep])
                else
                    for j in 1:kstep:size(xnt,2)
                        trajerrors[n, i] += abs(xnt[n, j] - xt[1 + (n-1) * nstep, j]) * kstep
                        trajstderrs[n, i] += abs2(xnt[n, j] - xt[1 + (n-1) * nstep, j]) * kstep
                    end
                end
            end
        end
    end

    # compute mean and standard error from first and second moments
    # in theory, argument should be non-negative, but use `abs`
    # because of round-off errors
    for i in eachindex(trajerrors, trajstderrs)
        trajstderrs[i] = sqrt( abs( trajstderrs[i] - trajerrors[i] ^ 2 / m ) / (m - 1) / m )
        trajerrors[i] /= m
    end

    return nothing
end

"""
    solve(rng, suite::ConvergenceSuite)

Compute the strong errors and the order of convergence of the given ConvergenceSuite `suite`, with the provided RNG seed `rng`.

The result is returned in the form of a [`ConvergenceResult`](@ref).
"""
function solve(rng::AbstractRNG, suite::ConvergenceSuite{T}) where {T}

    trajerrors = zeros(T, last(suite.ns) + 1, length(suite.ns))
    trajstderrs = zeros(T, last(suite.ns) + 1, length(suite.ns))

    calculate_trajerrors!(rng, trajerrors, trajstderrs, suite)
    
    errors = maximum(trajerrors, dims=1)[1, :]
    stderrs = maximum(trajstderrs, dims=1)[1, :]
    errorsminmax = Dict(
        :min => maximum(trajerrors .- trajstderrs, dims=1)[1,:],
        :max => maximum(trajerrors .+ trajstderrs, dims=1)[1,:]
    )
    deltas = (suite.tf - suite.t0) ./ suite.ns

    # fit to errors ∼ C Δtᵖ with lc = ln(C)
    vandermonde = [one.(deltas) log.(deltas)]
    logerrors = log.(errors)
    lc, p =  vandermonde \ logerrors

    # uncertainty in `p`: 95% confidence interval based on 
    # standard errors of Monte Carlo approximation of the errors
    lcsandps = vandermonde \ reduce(hcat, [log(errorsminmax[ci][i]) for (i, ci) in enumerate(c)] for c in Iterators.ProductIterator(Tuple(Iterators.repeated((:min,:max), length(errors)))))

    pmin, pmax = extrema(lcsandps[2, :])

    # return `result` as a `ConvergenceResult`
    result = ConvergenceResult(deltas, trajerrors, trajstderrs, errors, stderrs, lc, p, pmin, pmax)
    return result
end

# Think about a non-allocating solver for the convergence result/# By adding `trajerrors`, `vandermonde`, `logerrors`, `logdeltas` and `inv(vandermonde' * vandermonde)[2, 2]` as cache arguments to ConvergenceResult
# Adding an `result = init(suite)` with whatever initialization is necessary
# Adding a non-allocating solve!(rng, result, suite)
# Then rewrite `solve` to call `init` and then `solve!`.

"""
    generate_error_table(result, suite, info)

Generate the markdown table with the data for the strong errors. 
    
This is obteined from `result.errors`, with time steps `result.deltas` and lengths `suite.ns`, and the provided `info` for the problem, where `info` is given as a namedtuple with String fields `info.equation`, `info.ic`, and `info.noise`, some of which may be taken from `suite`.
"""
function generate_error_table(result::ConvergenceResult, suite:: ConvergenceSuite, info::NamedTuple=(equation = "RODE", ic=string(nameof(typeof(suite.x0law))), noise=string(nameof(typeof(suite.noise)))))
    t0 = suite.t0
    tf = suite.tf
    ns = suite.ns
    m = suite.m
    ntgt = suite.ntgt
    deltas = result.deltas
    errors = result.errors
    stderrs = result.stderrs
    table = "    \\begin{center}
        \\begin{tabular}[htb]{|r|l|l|l|}
            \\hline N & dt & error & std err \\\\
            \\hline \\hline\n"
    for (n, dt, error, stderr) in zip(
        ns,
        round.(deltas, sigdigits=3),
        round.(errors, sigdigits=3),
        round.(stderrs, sigdigits=3)
    )
        table *= "            $n & $dt & $error & $stderr \\\\\n"
    end
    table *= "            \\hline
        \\end{tabular}
    \\end{center}
    
    \\bigskip

    \\caption{Mesh points (N), time steps (dt), strong error (error), and standard error (std err) of the Euler method for $(info.equation) for each mesh resolution \$N\$, with initial condition $(info.ic) and $(info.noise), on the time interval \$I = [$t0, $tf]\$, based on \$M = $(m)\$ sample paths for each fixed time step, with the target solution calculated with \$$ntgt\$ points. The order of strong convergence is estimated to be \$p = $(round(result.p, digits=3))\$, with the 95\\% confidence interval \$[$(round(result.pmin, digits=4)), $(round(result.pmax, digits=4))]\$.}"
    return table
end