# We test the solvers with the equation $x' = f(t, x, y)$
# and initial condition $x(0) = x_0$, where
# $f(t, x, y) = (y + cos(t))x$ and "noise" $y(t) = cos(t)$.
# The solution is $x(t) = x0 e^{2\sin(t)}$.
# For vectorial unknown and vectorial noises, we simply repeat the equations and/or noise

# This is for testing the CustomMethod at the end of the test.
# There is some scoping problem that gives me extra allocations if I define this inside the test
custom_solver = function(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, params::P) where {T, F, P}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )

    n = length(xt) - 1
    dt = (tf - t0) / n
    i1 = firstindex(xt)
    xt[i1] = x0
    integral = 0.0
    for i in Iterators.drop(eachindex(xt, yt), 1)
        integral += (yt[i] + yt[i1]) * dt / 2
        if params isa AbstractRNG
            integral += sqrt(dt^3 / 12) * randn(params)
        end
        xt[i] = x0 * exp(integral)
        i1 = i
    end
end

@testset "Solvers" begin
    t0 = 0.0
    tf = 1.5
    n = 2^8
    tt = range(t0, tf, length=n+1)
    @testset "scalar/scalar Euler" begin
        x0 = 0.5
        f = (t, x, y) -> ( y + cos(t) ) * x
        yt = cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        sol = x0 * exp.( 2 * sin.(tt))
        method = RandomEuler()
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0
    end
    @testset "scalar/vector Euler" begin
        x0 = 0.5
        f = (t, x, y) -> ( sum(y) + cos(t) ) * x
        yt = [0.3 0.7] .* cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        sol = x0 * exp.( 2 * sin.(tt))
        method = RandomEuler()
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0
    end
    @testset "vector/scalar Euler" begin
        x0 = [0.2, 0.3]
        f! = (dx, t, x, y) -> (dx .= ( y + cos(t) ) .* x)
        yt = cos.(tt)
        xt = Matrix{Float64}(undef, n + 1, length(x0))
        sol = [x0[1] x0[2]] .* exp.( 2 * sin.(tt))
        method = RandomEuler(2)
        @test_nowarn solve!(xt, t0, tf, x0, f!, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f!, $yt, $method)) == 0
    end
    @testset "vector/vector Euler" begin
        x0 = [0.2, 0.3]
        f! = (dx, t, x, y) -> (dx .= ( sum(y) + cos(t) ) .* x)
        yt = [0.2 0.2 0.6] .* cos.(tt)
        xt = Matrix{Float64}(undef, n + 1, length(x0))
        sol = [x0[1] x0[2]] .* exp.( 2 * sin.(tt))
        method = RandomEuler(2)
        @test_nowarn solve!(xt, t0, tf, x0, f!, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f!, $yt, $method)) == 0
    end
    @testset "scalar/scalar Heun" begin
        x0 = 0.5
        f = (t, x, y) -> ( y + cos(t) ) * x
        yt = cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        sol = x0 * exp.( 2 * sin.(tt))
        method = RandomHeun()
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0
    end

    @testset "User solver 1" begin
        rng = Xoshiro(123)
        x0 = 0.5
        f = (t, x, y) -> y * x
        yt = cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        sol = x0 * exp.( sin.(tt))

        method = CustomUnivariateMethod(custom_solver, nothing)
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0
    end

    @testset "User solver 2" begin
        rng = Xoshiro(123)    
        x0 = 0.5
        f = (t, x, y) -> y * x
        noise = WienerProcess(t0, tf, 0.0)
        yt = Vector{Float64}(undef, n + 1)
        rand!(rng, noise, yt)
        xt = Vector{Float64}(undef, n + 1)
        dt = (tf - t0) / n
        sol = x0 * exp.( cumsum(yt) * dt)
    
        method = RandomEuler()
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0

        method = CustomUnivariateMethod(custom_solver, rng)
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0
    end
end
