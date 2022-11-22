function solve_euler!(rng, Xt, t0, tf, x0, f, Yt)
    N = length(Yt)
    dt = (tf - t0) / (N - 1)
    Xt[1] = x0
    for n in 2:N
        Xt[n] = Xt[n-1] + dt * f(Xt[n-1], Yt[n-1])
    end
end

function solve_heun!(rng, Xt, t0, tf, x0, f, Yt)
    N = length(Yt)
    dt = (tf - t0) / (N - 1)
    Xt[1] = x0
    for n in 2:N
        Xtnaux= Xt[n-1] + dt * f(Xt[n-1], Yt[n-1])
        Xt[n] = Xt[n-1] + dt * (f(Xt[n-1], Yt[n-1]) + f(Xtnaux, Yt[n])) / 2
    end
end