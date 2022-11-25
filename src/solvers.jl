"""
    solve_euler!(rng, Xt, t0, tf, x0, f, Yt)

Solve inplace, via Euler method, the RODE `dX_t/dt = f(t, X_t, Y_t),` with the following data:

* function `f=f(t, x, y)`;
* initial condition `x0`;
* time interval `t0` to `tf`;
* noise `Yt`.

The values of `Xt` are updated with the computed solution values.

The time step is obtained from the length `N` of the vector `Xt` via `dt = (tf - t0) / (N - 1)`.

The noise vector `Yt` is expected to be of the same length of `Xt`.
"""
function solve_euler!(rng, Xt, t0, tf, x0, f, Yt)
    N = length(Yt)
    dt = (tf - t0) / (N - 1)
    length(Xt) ≥ length(Xt) || throw(
        ArgumentError(
            "Length of `Xt` should be at least that of `Yt` but got $(length(Xt)) and $N"
        )
    )
    Xt[1] = x0
    tn1 = t0
    for n in 2:N
        Xt[n] = Xt[n-1] + dt * f(tn1, Xt[n-1], Yt[n-1])
        tn1 += dt
    end
end

"""
    solve_euler!(rng, Xt, t0, tf, x0, f, Yt)

Solve inplace, via Heun method, the RODE `dX_t/dt = f(t, X_t, Y_t),` with the following data:

* function `f=f(t, x, y)`;
* initial condition `x0`;
* time interval `t0` to `tf`;
* noise `Yt`.

The values of `Xt` are updated with the computed solution values.

The time step is obtained from the length `N` of the vector `Xt` via `dt = (tf - t0) / (N - 1)`.

The noise vector `Yt` is expected to be of the same length of `Xt`.
"""
function solve_heun!(rng, Xt, t0, tf, x0, f, Yt)
    N = length(Yt)
    dt = (tf - t0) / (N - 1)
    Xt[1] = x0
    tn1 = t0
    for n in 2:N
        fn1 = f(tn1, Xt[n-1], Yt[n-1])
        Xtnaux= Xt[n-1] + dt * fn1
        tn1 += dt
        Xt[n] = Xt[n-1] + dt * (fn1 + f(tn1, Xtnaux, Yt[n])) / 2
    end
end