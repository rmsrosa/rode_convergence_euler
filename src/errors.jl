"""
    prepare_variables(Ntgt, Ns)

Pre-allocate all the variables used in estimating the strong error.

Since no computation is actually made, we just need the length `Ntgt` for the target solution, and a list `Ns` of lengths for the approximations.

This returns the variables `nsteps`, `deltas`, `trajerrors`, `Yt`, `Xt`, `XNt`, where

1. The vectors `Yt` and `Xt` have length `Ntgt`, able to hold the values of the noise and the target solution computed in the fine mesh;
2. A vector `XNt` with the largest length in `Ns`, ready to hold any of the approximations with length in `Ns`;
3. A vector `trajerrors` of length `Ns` able to hold the strong error of each of the approximations with length `Ns`;
4. A vector `deltas` with length `Ns` able to hold the time steps to be used in each of the approximations;
5. A vector `nsteps` where each `nsteps[i]` gives the steps in the fine mesh to coincide with the corresponding steps in the coarse mesh, i.e. `nsteps[i]*Ns[i] = Ntgt`;

The length `Ntgt` should be divisible by any of the lengths in `Ns`.
"""
function prepare_variables(Ntgt::Int, Ns::Vector{Int})

    all(mod(Ntgt, N) == 0 for N in Ns) || throw(
        ArgumentError(
            "The length `Ntgt = $Ntgt` should be divisible by any of the lengths in `Ns=$Ns`"
        )
    )

    nsteps = div.(Ntgt, Ns)

    deltas = Vector{Float64}(undef, length(Ns))
    trajerrors = zeros(last(Ns), length(Ns))

    Yt = Vector{Float64}(undef, Ntgt)
    Xt = Vector{Float64}(undef, Ntgt)
    XNt = Vector{Float64}(undef, last(Ns))

    return nsteps, deltas, trajerrors, Yt, Xt, XNt
end

"""
    calculate_errors!(rng, Yt, Xt, XNt, X0law, f, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

Calculate the strong error for the Euler method.

The equation is assumed to be given with right hand side `f`, which must be a function of three variables `f=f(t, x, y)`, where `t` is time, `x` is the unknown, and `y` is the noise. The initial condition is a random variable drawn from the distribution law `X0law` via `rand(rng, X0law)`. The time span is assumed to be from `t0` to `tf`.

The noise is computed via `noise!`, which is a function that takes a RNG and a pre-allocated vector to hold the values of the noise.

The strong error is computed with respect to the `target!` solution, computed via `target!(rng, Xt, t0, tf, x0, f, Yt)`, which fills up the pre-allocated vectors `Xt` and `Yt` with the solution and the associate noise, respectively.

The strong errors are computed for each approximation with length in the vector `Ns` of lengths and stored in the vector `trajerrors`. The associated time steps are stored in the vector `deltas`.

The strong errors are computed via Monte Carlo method, with the number of realizations defined by the argument `M`.
"""
function calculate_errors!(rng, Yt, Xt, XNt, X0law, f::F, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas) where F
    
    for _ in 1:M
        # draw initial condition
        x0 = rand(rng, X0law)

        # generate noise sample path
        noise!(rng, Yt)

        # generate target path
        target!(rng, Xt, t0, tf, x0, f, Yt)

        # solve approximate solutions at selected time steps and update strong errors
        for (i, (nstep, N)) in enumerate(zip(nsteps, Ns))

            deltas[i] = (tf - t0) / (N - 1)

            solve_euler!(rng, XNt, t0, tf, x0, f, view(Yt, 1:nstep:1+nstep*(N-1)))

            for n in 2:N
                trajerrors[n, i] += abs(XNt[n] - Xt[1 + (n-1) * nstep])
            end
        end
    end

    # normalize errors
    trajerrors ./= M
    return Yt, Xt, XNt
end

"""
    calculate_errors(rng, t0, tf, X0law, f, noise!, target!, Ntgt, Ns, M)

Calculate the strong error for the Euler method.

The equation is assumed to be given with right hand side `f`, which must be a function of three variables `f=f(t, x, y)`, where `t` is time, `x` is the unknown, and `y` is the noise. The initial condition is a random variable drawn from the distribution law `X0law` via `rand(rng, X0law)`. The time span is assumed to be from `t0` to `tf`.

The noise is computed via `noise!`, which is a function that takes a RNG and a pre-allocated vector to hold the values of the noise.

The strong error is computed with respect to the `target!` solution, computed via `target!(rng, Xt, t0, tf, x0, f, Yt)`, which fills up pre-allocated vectors `Xt` and `Yt` with the solution and the associate noise, respectively.

The strong errors are computed for each approximation with length in the vector `Ns` of lengths and stored in the vector `trajerrors`. The associated time steps are stored in the vector `deltas`.

The strong errors are computed via Monte Carlo method, with the number of realizations defined by the argument `M`.

What this function do is actually to call [`prepare_variables`](@ref) to pre-allocate the necessary variables and next to call [`calculate_errors!`](@ref) to mutate the pre-allocated vectors.
"""
function calculate_errors(rng, t0, tf, X0law, f, noise!, target!, Ntgt, Ns, M)
    nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

    calculate_errors!(rng, Yt, Xt, XNt, X0law, f, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

    errors = maximum(trajerrors, dims=1)[1,:]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, errors, trajerrors, lc, p
end
