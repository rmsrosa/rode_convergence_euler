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

@btime solve!(rng, suite);