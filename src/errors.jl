"""
    prepare_variables(Ntgt, Ns; nx = 0, ny = 0)

Pre-allocate all the necessary variables used in estimating the strong error.

Since no computation is actually made, we just need the length `Ntgt` for the target solution, a list `Ns` of lengths for the approximations, and the sizes of the solution values and the noise.

This returns the variables `nsteps`, `deltas`, `trajerrors`, `Yt`, `Xt`, `XNt`, where

1. The arrays `Yt` and `Xt` have (row) length `Ntgt`, able to hold the values of the noise and the target solution computed in the fine mesh, and could be vectors or matrices, depending on whether `nx == 0` or `nx > 0` and on whether `ny == 0` or `ny > 0`;
2. A vector/matrix `XNt` with the largest (row) length in `Ns`, ready to hold any of the approximations with length in `Ns`;
3. A vector `trajerrors` of length `Ns` able to hold the strong error of each of the approximations with length `Ns`;
4. A vector `deltas` with length `Ns` able to hold the time steps to be used in each of the approximations;
5. A vector `nsteps` where each `nsteps[i]` gives the steps in the fine mesh to coincide with the corresponding steps in the coarse mesh, i.e. `nsteps[i]*Ns[i] = Ntgt`;

The length `Ntgt` should be divisible by any of the lengths in `Ns`, with both positive, while `nx` and `ny` just need to be non-negative integers.
"""
function prepare_variables(Ntgt::Int, Ns::Vector{Int}; nx::Int = 0, ny::Int = 0)

    (Ntgt > 0 && all(>(0), Ns) && nx ≥ 0 && ny ≥ 0 ) || throw(
        ArgumentError(
            "`Ntgt` and `Ns` arguments must be positive  and `nx` and `ny` keyword arguments must be non-negative."
        )
    )
    all(mod(Ntgt, N) == 0 for N in Ns) || throw(
        ArgumentError(
            "The length `Ntgt = $Ntgt` should be divisible by any of the lengths in `Ns=$Ns`"
        )
    )

    nsteps = div.(Ntgt, Ns)

    deltas = Vector{Float64}(undef, length(Ns))
    trajerrors = zeros(last(Ns), length(Ns))

    Yt = ny == 0 ? Vector{Float64}(undef, Ntgt) : Matrix{Float64}(undef, Ntgt, ny)
    Xt = nx == 0 ? Vector{Float64}(undef, Ntgt) : Matrix{Float64}(undef, Ntgt, nx)
    XNt = nx == 0 ? Vector{Float64}(undef, last(Ns)) : Matrix{Float64}(undef, last(Ns), nx)

    return nsteps, deltas, trajerrors, Yt, Xt, XNt
end

"""
    calculate_errors!(rng, Yt, Xt, XNt, X0law, f, noise, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

Calculate the strong error for the Euler method.

When `x` is a scalar, the equation is a scalar equation assumed to be given with right hand side `f`, which must be a function with three arguments, `f=f(t, x, y)`, where `t` is time, `x` is the unknown, and `y` is the noise. When `x` is a vector, corresponding to a system of equations, the function is assumed to have four arguments, `f=f(dx, t, x, y)`, modifying the derivative `dx` in place, of same size as `x`. The initial condition is a random variable drawn from the distribution law `X0law` via `rand(rng, X0law)`, in the scalar case, or via `rand!(rng, X0law, x0)`, in the case of systems. The time span is assumed to be from `t0` to `tf`.

A sample path of the noise is computed via `rand!(rng, noise, yt)`, which is a function that takes a RNG, the noise process `noise`, and a pre-allocated vector/matrix to hold the values of the sample path.

The strong error is computed with respect to the `target!` solution, computed via `target!(rng, Xt, t0, tf, x0, f, Yt)`, which fills up the pre-allocated vectors/matrices `Xt` and `Yt` with the solution and the associate noise path, respectively.

The strong errors are computed for each approximation with length in the vector `Ns` of lengths and stored in the vector `trajerrors`. The associated time steps are stored in the vector `deltas`.

The strong errors are computed via Monte Carlo method, with the number of realizations defined by the argument `M`.
"""
function calculate_errors!(rng::AbstractRNG, Yt::VecOrMat{T}, Xt::VecOrMat{T}, XNt::VecOrMat{T}, X0law::ContinuousDistribution{N0}, f::F, noise::AbstractProcess{NY}, target!::G, trajerrors::Matrix{T}, M::Int, t0::T, tf::T, Ns::Vector{Int}, nsteps::Vector{Int}, deltas::Vector{T}) where {T, N0, NY, F, G}
    
    # get whether a system 
    xisinplace = N0 == Multivariate
    yisinplace = NY == Multivariate
    for _ in 1:M
        # draw initial condition
        if N0 == Multivariate
            rand!(rng, X0law, view(Xt, 1, :))
        else
            Xt[1] = rand(rng, X0law)
        end

        # generate noise sample path
        rand!(rng, noise, Yt)

        # generate target path
        if N0 == Multivariate
            target!(rng, Xt, t0, tf, view(Xt, 1, :), f, Yt)
        else
            target!(rng, Xt, t0, tf, Xt[1], f, Yt)
        end

        # solve approximate solutions at selected time steps and update strong errors
        for (i, (nstep, Nsi)) in enumerate(zip(nsteps, Ns))

            deltas[i] = (tf - t0) / (Nsi - 1)

            if N0 == Multivariate && NY == Multivariate
                solve_euler!(rng, view(XNt, 1:Nsi, :), t0, tf, view(Xt, 1, :), f, view(Yt, 1:nstep:1+nstep*(Nsi-1), :))
            elseif N0 == Multivariate
                solve_euler!(rng, view(XNt, 1:Nsi, :), t0, tf, view(Xt, 1, :), f, view(Yt, 1:nstep:1+nstep*(Nsi-1)))
            elseif NY == Multivariate
                solve_euler!(rng, view(XNt, 1:Nsi), t0, tf, Xt[1], f, view(Yt, 1:nstep:1+nstep*(Nsi-1), :))
            else
                solve_euler!(rng, view(XNt, 1:Nsi), t0, tf, Xt[1], f, view(Yt, 1:nstep:1+nstep*(Nsi-1)))
            end

            for n in 2:Nsi
                if N0 == Multivariate
                    for j in eachindex(axes(XNt, 2))
                        trajerrors[n, i] += abs(XNt[n, j] - Xt[1 + (n-1) * nstep, j])
                    end
                else
                    trajerrors[n, i] += abs(XNt[n] - Xt[1 + (n-1) * nstep])
                end
            end
        end
    end

    # normalize errors
    trajerrors ./= M
    return Yt, Xt, XNt
end

"""
    calculate_errors(rng, t0, tf, X0law, f, noise, target!, Ntgt, Ns, M)

Calculate the strong error for the Euler method.

The equation is assumed to be given with right hand side `f`, which must be a function of three variables `f=f(t, x, y)`, where `t` is time, `x` is the unknown, and `y` is the noise. The initial condition is a random variable drawn from the distribution law `X0law` via `rand(rng, X0law)`. The time span is assumed to be from `t0` to `tf`.

The noise is computed by drawing from `noise`, which is a AbstractNoise that generates sample paths from `rand!(rng, noise, Yt)` from a RNG and on a pre-allocated vector to hold the values of the noise.

The strong error is computed with respect to the `target!` solution, computed via `target!(rng, Xt, t0, tf, x0, f, Yt)`, which fills up pre-allocated vectors `Xt` and `Yt` with the solution and the associate noise, respectively.

The strong errors are computed for each approximation with length in the vector `Ns` of lengths and stored in the vector `trajerrors`. The associated time steps are stored in the vector `deltas`.

The strong errors are computed via Monte Carlo method, with the number of realizations defined by the argument `M`.

What this function do is actually to call [`prepare_variables`](@ref) to pre-allocate the necessary variables and next to call [`calculate_errors!`](@ref) to mutate the pre-allocated vectors.
"""
function calculate_errors(rng::AbstractRNG, t0::T, tf::T, X0law::ContinuousDistribution{N0}, f::F, noise::AbstractProcess{NY}, target!::G, Ntgt::Int, Ns::Vector{Int}, M::Int) where {T, N0, NY, F, G}
    nx = N0 == Multivariate ? length(X0law) : 0
    ny = NY == Multivariate ? length(noise) : 0
    nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns; nx, ny)

    calculate_errors!(rng, Yt, Xt, XNt, X0law, f, noise, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

    errors = maximum(trajerrors, dims=1)[1, :]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, errors, trajerrors, lc, p
end
