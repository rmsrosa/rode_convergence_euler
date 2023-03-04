@testset "Test solvers" begin
    # We test the solvers with the equation $x' = f(t, x, y)$
    # and initial condition $x(0) = x_0$, where
    # $f(t, x, y) = (y + cos(t))x$ and "noise" $y(t) = cos(t)$.
    # The solution is $x(t) = x0 e^{2\sin(t)}$.
    # For vectorial unknown and vectorial noises, we simply repeat the equations and/or noise
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 2.0
    n = 2^8
    tt = range(t0, tf, length=n)
    @testset "scalar/scalar Euler" begin
        x0 = 0.5
        f = (t, x, y) -> ( y + cos(t) ) * x
        yt = cos.(tt)
        xt = Vector{Float64}(undef, n)
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
        xt = Vector{Float64}(undef, n)
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
        xt = Matrix{Float64}(undef, n, length(x0))
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
        xt = Matrix{Float64}(undef, n, length(x0))
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
        xt = Vector{Float64}(undef, n)
        sol = x0 * exp.( 2 * sin.(tt))
        method = RandomHeun()
        @test_nowarn solve!(xt, t0, tf, x0, f, yt, method)
        @test maximum(abs, xt .- sol) < 0.05
        @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $method)) == 0
    end
end