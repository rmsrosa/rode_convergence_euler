using Random
using BenchmarkTools

function integrate!(Xt, f, t0, tf, x0)
    N = length(Xt)
    dt = (tf - t0) / (N - 1)
    Xt[1] = x0
    for n in 2:N
        Xt[n] = Xt[n-1] + dt * f(Xt[n-1])
        #Xt[n] = Xt[n-1] - dt * Xt[n-1]
    end
end

function solve!(rng, Xt, f::F, t0, tf) where F
    x0 = rand(rng)
    integrate!(Xt, f, t0, tf, x0)
end

rng = Xoshiro(123)
t0 = 0.0
tf = 1.0
f(x) = -x

Xt = zeros(2^6)

solve!(rng, Xt, f, t0, tf)

@btime solve!($rng, $Xt, $f, $t0, $tf)

@code_warntype solve!(rng, Xt,  f, t0, tf)
@profview solve!(rng, Xt, f, t0, tf)

@profview_allocs solve!(rng, Xt, f, t0, tf)

