"""
    ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)

Gather the data needed for computing the convergence error for a given RODE of the form
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
* the method `target` to compute the target solution for the error calculation via `solve!(xt, t0, tf, x0, f, yt, target)`, typically `EulerMethod()` with a much finer resolution with `ntgt` mesh points or the order of the square of the highest number of mesh points in `ns` (see below) or a lower resolution `CustomMethod() with an exact distribution of the possible solutions conditioned to the already computed noise points;
* the `method` to approximate the solution, typically the `EulerMethod()`, also in the form `solve!(xt, t0, tf, x0, f, yt, method)`;
* the number `ntgt` of mesh points in the fine mesh on which the target solution will be computed;
* the vector `ns` with a list of numbers of mesh points to compute the approximate solutions;
* the number `m` of sample paths to be computed for estimating the strong error via Monte Carlo method.

Besides these data obtained from the supplied arguments, a few cache vectors or matrices are created:
* a vector or matrix `yt` to hold the sample paths of the noise on the finest mesh, with length or row-length being `ntgt` and the shape depending on whether the noise is univariate or multivariate;
* a vector or matrix `xt` to hold the sample paths of the target solution, on the finest mesh, with length or row-length being `ntgt` and the shape depending on whether the law for the initial condition being univariate or multivariate;
* a vector or matrix `xnt` to hold the sample paths of the approximate solution, with length or row-length being the maximum of those in `ns` and the shape depending on whether the law for the initial condition being univariate or multivariate.

The actual error is obtained by solving a ConvergenceSuite via `solve(rng::AbstractRNG, suite::ConvergenceSuite{T})` with a given RNG.
"""
struct ConvergenceSuite{T, D, P, F, N1, N2, M1, M2}
    t0::T
    tf::T
    x0law::D
    f::F
    noise::P
    target::M1
    method::M2
    ntgt::Int
    ns::Vector{Int}
    m::Int
    yt::Array{T, N1} # cache
    xt::Array{T, N2} # cache
    xnt::Array{T, N2} # cache
    function ConvergenceSuite(t0::T, tf::T, x0law::D, f::F, noise::P, target::M1, method::M2, ntgt::Int, ns::Vector{Int}, m::Int) where {T, D, P, F, M1, M2}
        ( ntgt > 0 && all(>(0), ns) ) || error(
            "`ntgt` and `ns` arguments must be positive integers."

        )
        all(mod(ntgt, n) == 0 for n in ns) || error(
            "The length of `ntgt` should be divisible by any of the lengths in `ns`"
        )
    
        ( M1 <: RODEMethod && M2 <: RODEMethod ) || error(
            "The `target` and `method` solver methods should be subtypes of `RODEMethod`"
        )
        if D <: UnivariateDistribution
            xt = Vector{T}(undef, ntgt)
            xnt = Vector{T}(undef, last(ns))
            N2 = 1
        elseif D <: MultivariateDistribution
            nx = length(x0law)
            xt = Matrix{T}(undef, ntgt, nx)
            xnt = Matrix{T}(undef, last(ns), nx)
            N2 = 2
        else
            error(
                "`xlaw` should be either `UnivariateDistribution` or `MultivariateDistribution`."
            )
        end
        if P <: UnivariateProcess
            yt = Vector{T}(undef, ntgt)
            N1 = 1
        elseif P <: MultivariateProcess
            yt = Matrix{T}(undef, ntgt, length(noise))
            N1 = 2
        else
            error(
                "`noise` should be either a `UnivariateProcess` or a `MultivariateProcess`."
            )
        end

        return new{T, D, P, F, N1, N2, M1, M2}(t0, tf, x0law, f, noise, target, method, ntgt, ns, m, yt, xt, xnt)
    end
end

"""
    ConvergenceResult{T, S}(suite::S, deltas::Vector{T}, trajerrors::Matrix{T}, errors::Vector{T}, lc::T, p::T,eps::T) where {T, S}

Stores the result of `solve(rng, suite)` with fields
* `suite`: the `ConvergenceSuite` which is to be solved for;
* `deltas`: the time steps associated with the number of mesh points in the vector `suite.ns`;
* `trajerrors`: a matrix where each column corresponds to the strong error, along the trajectory, at each mesh resolution determined by `suite.ns`, i.e. `trajerrors[i, k]` is the error at time ``t_0 + i \\Delta t``, for the time step ``\\Delta t = (t_f - t_0) / (n - 1)`` associated with the kth element `n = suite.ns[k]`;
* `errors`: the maximum, along the trajectory, of the `trajerrors`;
* `lc`: the logarithm ``\\log(C)`` of the multiplicative constant in the fitted error `CΔtᵖ`;
* `p`: the order of the strong convergence;
* `eps`: an estimate of the half-width of the 95% confidence interval for `p` in the least square fit as a maximum likelyhood estimate.
"""
struct ConvergenceResult{T, S}
    suite::S
    deltas::Vector{T}
    trajerrors::Matrix{T}
    errors::Vector{T}
    lc::T
    p::T
    eps::T
end

"""
    calculate_trajerrors!(rng, trajerrors, suite)

Calculate the strong error at each time step along the trajectory.
"""
function calculate_trajerrors!(rng, trajerrors::Matrix{T}, suite::ConvergenceSuite{T, D, P}) where {T, D, P}
    t0 = suite.t0
    tf = suite.tf
    x0law = suite.x0law
    f = suite.f
    noise = suite.noise
    target = suite.target
    method = suite.method
    ntgt = suite.ntgt
    ns = suite.ns
    m = suite.m
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
            solve!(xt, t0, tf, xt[1], f, yt, target)
        else
            solve!(xt, t0, tf, view(xt, 1, :), f, yt, target)            
        end

        # solve approximate solutions at selected time steps and update strong errors
        #for (i, (nstep, nsi)) in enumerate(zip(nsteps, ns))
        for (i, nsi) in enumerate(ns)

            nstep = div(ntgt, nsi)

            if D <: UnivariateDistribution && P <: UnivariateProcess
                solve!(view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1)), method)
            elseif D <: UnivariateDistribution
                solve!(view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1), :), method)
            elseif P <: UnivariateProcess
                solve!(view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1)), method)
            else
                solve!(view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1), :), method)
            end

            for n in 2:nsi
                if D <: UnivariateDistribution
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

"""
    solve(rng, suite::ConvergenceSuite)

Compute the strong errors and the order of convergence.
"""
function solve(rng::AbstractRNG, suite::ConvergenceSuite{T}) where {T}

    trajerrors = zeros(T, last(suite.ns), length(suite.ns))

    calculate_trajerrors!(rng, trajerrors, suite)
    
    errors = maximum(trajerrors, dims=1)[1, :]
    deltas = (suite.tf - suite.t0) ./ (suite.ns .- 1)

    # fit to errors ∼ C Δtᵖ with lc = ln(C)
    vandermonde = [one.(deltas) log.(deltas)]
    logerrors = log.(errors)
    lc, p =  vandermonde \ logerrors

    # uncertainty in `p`: 95% confidence interval (p - eps, p + eps) where

    standard_error = √(sum(abs2, logerrors .- ( lc .+ p .* log.(deltas))) / (length(deltas) - 2))
    eps = 2 * standard_error * inv(vandermonde' * vandermonde)[2, 2]

    # return results as a `ConvergenceResult`
    results = ConvergenceResult(suite, deltas, trajerrors, errors, lc, p, eps)
    return results
end

# Think about a non-allocating solver for the convergence results/# By adding `trajerrors`, `vandermonde`, `logerrors`, `logdeltas` and `inv(vandermonde' * vandermonde)[2, 2]` as cache arguments to ConvergenceResult
# Adding an `results = init(suite)` with whatever initialization is necessary
# Adding a non-allocating solve!(rng, results, suite)
# Then rewrite `solve` to call `init` and then `solve!`.

"""
    generate_error_table(results, info)

Generate the markdown table with the data for the strong `results.errors` obtained with time steps `results.deltas` and lengths `results.suite.ns`, and the provided `info` for the problem, where `info` is given as a namedtuple with String fields `info.equation`, `info.ic`, and `info.noise`.
"""
function generate_error_table(results::ConvergenceResult, info::NamedTuple=(equation = "RODE", ic=string(nameof(typeof(results.suite.x0law))), noise=string(nameof(typeof(results.suite.noise)))))
    t0 = results.suite.t0
    tf = results.suite.tf
    ns = results.suite.ns
    m = results.suite.m
    ntgt = results.suite.ntgt
    deltas = results.deltas
    errors = results.errors
    table = "    \\begin{tabular}[htb]{|r|l|l|}
        \\hline N & dt & error\\\\
        \\hline \\hline\n"
    for (n, dt, error) in zip(ns, round.(deltas, sigdigits=3), round.(errors, sigdigits=3))
        table *= "        $n & $dt & $error \\\\\n"
    end
    table *= "        \\hline
    \\end{tabular}
    \\bigskip

    \\caption{Mesh points (N), time steps (dt), and strong error (error) of the Euler method for $(info.equation), with initial condition $(info.ic) and $(info.noise), on the time interval ($t0, $tf), based on \$m = $(m)\$ sample paths for each fixed time step, with the target solution calculated with \$$ntgt\$ points.}"
    return table
end