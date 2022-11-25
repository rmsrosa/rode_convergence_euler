function prepare_variables(Ntgt, Ns)

    nsteps = div.(Ntgt, Ns)

    deltas = Vector{Float64}(undef, length(Ns))
    trajerrors = zeros(last(Ns), length(Ns))

    Yt = Vector{Float64}(undef, Ntgt)
    Xt = Vector{Float64}(undef, Ntgt)
    XNt = Vector{Float64}(undef, last(Ns))

    return nsteps, deltas, trajerrors, Yt, Xt, XNt
end

function calculate_errors!(rng, Yt, Xt, XNt, X0, f::F, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas) where F
    for _ in 1:M
        # draw initial condition
        x0 = rand(rng, X0)

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

function calculate_errors(rng, t0, tf, X0, f, noise!, target!, Ntgt, Ns, M)
    nsteps, deltas, trajerrors, Yt, Xt, XNt = prepare_variables(Ntgt, Ns)

    calculate_errors!(rng, Yt, Xt, XNt, X0, f, noise!, target!, trajerrors, M, t0, tf, Ns, nsteps, deltas)

    errors = maximum(trajerrors, dims=1)[1,:]

    lc, p = [one.(deltas) log.(deltas)] \ log.(errors)

    return deltas, errors, trajerrors, lc, p
end
