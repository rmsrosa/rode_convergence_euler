using Random, LinearAlgebra, Statistics
using Distributions, BenchmarkTools
using RODEConvergence
using Test
using Plots

t0 = 0.0
tf = 2.0
n = 2^16
m = 5_000
ythf = Vector{Float64}(undef, m)
ytf = Vector{Float64}(undef, m)
yt = Vector{Float64}(undef, n)

ntgt = 2^16
ns = 2 .^ (4:8)
m = 1_000

rng = Xoshiro(123)
x0law = Normal()
y0 = 0.2
μ = 0.3
σ = 0.2
λ = 10.0
α = 2.0
β = 15.0
dylaw = Normal(μ, σ)
steplaw = Beta(α, β)
nr = 10
fy = (t, r) -> mapreduce(ri -> cbrt(sin(t/ri)), +, r) / length(r)
ylaw = Beta(α/2, β)
hurst = 0.25
noise = ProductProcess(
    WienerProcess(t0, tf, y0),
    GeometricBrownianMotionProcess(t0, tf, y0, μ, σ),
    CompoundPoissonProcess(t0, tf, λ, ylaw),
    PoissonStepProcess(t0, tf, λ, steplaw),
    TransportProcess(t0, tf, ylaw, fy, nr),
    FractionalBrownianMotionProcess(t0, tf, y0, hurst, n)
)

ymt = Matrix{Float64}(undef, n, length(noise))

rand!(rng, noise, ymt)

plot(ymt[:, 5])
plot(ymt)

for noisei in noise.processes
    @info first(split(string(noisei), "{"))
    @btime rand!($rng, $noisei, $yt)
end
@info first(split(string(noise), "{"))
@btime rand!($rng, $noise, $ymt)

noise = ProductProcess(PoissonStepProcess(t0, tf, λ, steplaw))
ymt = Matrix{Float64}(undef, n, length(noise))

rand!(rng, noise, ymt)

plot(ymt)


f = (t, x, y) -> -x + mean(y)

@btime ConvergenceSuite(t0, tf, x0law, f, noise, solve_euler!, solve_euler!, ntgt, ns, m);

suite = ConvergenceSuite(t0, tf, x0law, f, noise, solve_euler!, solve_euler!, ntgt, ns, m)

@btime solve(rng, suite);

trajerrors = zeros(last(suite.ns), length(suite.ns))

@btime RODEConvergence.calculate_trajerrors!($rng, $trajerrors, $suite)

function try1!(rng, trajerrors::Matrix{T}, suite::ConvergenceSuite{T, Univariate, Univariate, F1, F2, F3}) where {T, N0, NY, F1, F2, F3}
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
        #if N0 == Multivariate
        #    rand!(rng, x0law, view(xt, 1, :))
        #else
            xt[1] = rand(rng, x0law)
        #end

        # generate noise sample path
        #rand!(rng, noise, yt)

        # generate target path
        #if N0 == Multivariate
        #    target!(rng, xt, t0, tf, view(xt, 1, :), f, yt)
        #else
        #    target!(rng, xt, t0, tf, xt[1], f, yt)
        #end

        # solve approximate solutions at selected time steps and update strong errors
        #= for (i, nsi) in enumerate(ns)

            nstep = div(ntgt, nsi)

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
        end =#
    end

    # normalize trajectory errors
    #trajerrors ./= m

    return nothing
end

@btime try1!($rng, $trajerrors, $suite)