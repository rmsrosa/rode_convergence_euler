"""
    prepare_variables(ntgt, ns; nx = 0, ny = 0)

Pre-allocate all the necessary variables used in estimating the strong error.

Since no computation is actually made, we just need the length `ntgt` for the target solution, a list `ns` of lengths for the approximations, and the sizes of the solution values and the noise.

This returns the variables `nsteps`, `deltas`, `trajerrors`, `yt`, `xt`, `xnt`, where

1. The arrays `yt` and `xt` have (row) length `ntgt`, able to hold the values of the noise and the target solution computed in the fine mesh, and could be vectors or matrices, depending on whether `nx == 0` or `nx > 0` and on whether `ny == 0` or `ny > 0`;
2. A vector/matrix `xnt` with the largest (row) length in `ns`, ready to hold any of the approximations with length in `ns`;
3. A vector `trajerrors` of length `ns` able to hold the strong error of each of the approximations with length `ns`;
4. A vector `deltas` with length `ns` able to hold the time steps to be used in each of the approximations;
5. A vector `nsteps` where each `nsteps[i]` gives the steps in the fine mesh to coincide with the corresponding steps in the coarse mesh, i.e. `nsteps[i]*ns[i] = ntgt`;

The length `ntgt` should be divisible by any of the lengths in `ns`, with both positive, while `nx` and `ny` just need to be non-negative integers.
"""
function prepare_variables(ntgt::Int, ns::Vector{Int}; nx::Int = 0, ny::Int = 0)

    (ntgt > 0 && all(>(0), ns) && nx ≥ 0 && ny ≥ 0 ) || throw(
        ArgumentError(
            "`ntgt` and `ns` arguments must be positive  and `nx` and `ny` keyword arguments must be non-negative."
        )
    )
    all(mod(ntgt, n) == 0 for n in ns) || throw(
        ArgumentError(
            "The length `ntgt = $ntgt` should be divisible by any of the lengths in `ns=$ns`"
        )
    )

    nsteps = div.(ntgt, ns)

    deltas = Vector{Float64}(undef, length(ns))
    trajerrors = zeros(last(ns), length(ns))

    yt = ny == 0 ? Vector{Float64}(undef, ntgt) : Matrix{Float64}(undef, ntgt, ny)
    xt = nx == 0 ? Vector{Float64}(undef, ntgt) : Matrix{Float64}(undef, ntgt, nx)
    xnt = nx == 0 ? Vector{Float64}(undef, last(ns)) : Matrix{Float64}(undef, last(ns), nx)

    return nsteps, deltas, trajerrors, yt, xt, xnt
end

"""
    calculate_errors!(rng, trajerrors, yt, xt, xnt, x0law, f, noise, target!, M, t0, tf, ns, nsteps, deltas)

Calculate the strong error for the Euler method.

When `x` is a scalar, the equation is a scalar equation assumed to be given with right hand side `f`, which must be a function with three arguments, `f=f(t, x, y)`, where `t` is time, `x` is the unknown, and `y` is the noise. When `x` is a vector, corresponding to a system of equations, the function is assumed to have four arguments, `f=f(dx, t, x, y)`, modifying the derivative `dx` in place, of same size as `x`. The initial condition is a random variable drawn from the distribution law `x0law` via `rand(rng, x0law)`, in the scalar case, or via `rand!(rng, x0law, x0)`, in the case of systems. The time span is assumed to be from `t0` to `tf`.

A sample path of the noise is computed via `rand!(rng, noise, yt)`, which is a function that takes a RNG, the noise process `noise`, and a pre-allocated vector/matrix to hold the values of the sample path.

The strong error is computed with respect to the `target!` solution, computed via `target!(rng, xt, t0, tf, x0, f, yt)`, which fills up the pre-allocated vectors/matrices `xt` and `yt` with the solution and the associate noise path, respectively.

The strong errors are computed for each approximation with length in the vector `ns` of lengths and stored in the vector `trajerrors`. The associated time steps are stored in the vector `deltas`.

The strong errors are computed via Monte Carlo method, with the number of realizations defined by the argument `M`.
"""
function calculate_errors!(rng::AbstractRNG, trajerrors::Matrix{T}, yt::VecOrMat{T}, xt::VecOrMat{T}, xnt::VecOrMat{T}, x0law::ContinuousDistribution{N0}, f::F, noise::AbstractProcess{NY}, target!::G, m::Int, t0::T, tf::T, ns::Vector{Int}, nsteps::Vector{Int}, deltas::Vector{T}) where {T, N0, NY, F, G}

    for _ in 1:m
        # draw initial condition
        if N0 == Multivariate
            rand!(rng, x0law, view(xt, 1, :))
        else
            xt[1] = rand(rng, x0law)
        end

        # generate noise sample path
        rand!(rng, noise, yt)

        # generate target path
        if N0 == Multivariate
            target!(rng, xt, t0, tf, view(xt, 1, :), f, yt)
        else
            target!(rng, xt, t0, tf, xt[1], f, yt)
        end

        # solve approximate solutions at selected time steps and update strong errors
        for (i, (nstep, nsi)) in enumerate(zip(nsteps, ns))

            deltas[i] = (tf - t0) / (nsi - 1)

            if N0 == Multivariate && NY == Multivariate
                solve_euler!(rng, view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1), :))
            elseif N0 == Multivariate
                solve_euler!(rng, view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1)))
            elseif NY == Multivariate
                solve_euler!(rng, view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1), :))
            else
                solve_euler!(rng, view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1)))
            end

            for n in 2:nsi
                if N0 == Multivariate
                    for j in eachindex(axes(xnt, 2))
                        trajerrors[n, i] += abs(xnt[n, j] - xt[1 + (n-1) * nstep, j])
                    end
                else
                    trajerrors[n, i] += abs(xnt[n] - xt[1 + (n-1) * nstep])
                end
            end
        end
    end

    # normalize errors
    trajerrors ./= m
    return yt, xt, xnt
end

"""
    calculate_errors(rng, t0, tf, x0law, f, noise, target!, ntgt, ns, m)

Calculate the strong error for the Euler method.

The equation is assumed to be given with right hand side `f`, which must be a function of three variables `f=f(t, x, y)`, where `t` is time, `x` is the unknown, and `y` is the noise. The initial condition is a random variable drawn from the distribution law `x0law` via `rand(rng, x0law)`. The time span is assumed to be from `t0` to `tf`.

The noise is computed by drawing from `noise`, which is a AbstractNoise that generates sample paths from `rand!(rng, noise, yt)` from a RNG and on a pre-allocated vector to hold the values of the noise.

The strong error is computed with respect to the `target!` solution, computed via `target!(rng, xt, t0, tf, x0, f, yt)`, which fills up pre-allocated vectors `xt` and `yt` with the solution and the associate noise, respectively.

The strong errors are computed for each approximation with length in the vector `ns` of lengths and stored in the vector `trajerrors`. The associated time steps are stored in the vector `deltas`.

The strong errors are computed via Monte Carlo method, with the number of realizations defined by the argument `M`.

What this function do is actually to call [`prepare_variables`](@ref) to pre-allocate the necessary variables and next to call [`calculate_errors!`](@ref) to mutate the pre-allocated vectors.
"""
function calculate_errors(rng::AbstractRNG, t0::T, tf::T, x0law::ContinuousDistribution{N0}, f::F, noise::AbstractProcess{NY}, target!::G, ntgt::Int, ns::Vector{Int}, m::Int) where {T, N0, NY, F, G}
    nx = N0 == Multivariate ? length(x0law) : 0
    ny = NY == Multivariate ? length(noise) : 0
    nsteps, deltas, trajerrors, yt, xt, xnt = prepare_variables(ntgt, ns; nx, ny)

    calculate_errors!(rng, trajerrors, yt, xt, xnt, x0law, f, noise, target!, m, t0, tf, ns, nsteps, deltas)

    errors = maximum(trajerrors, dims=1)[1, :]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, errors, trajerrors, lc, p
end
