@testset "Test solvers" begin
    # We test the solvers with the equation $x' = f(t, x, y)$
    # and initial condition $x(0) = x_0$, where
    # $f(t, x, y) = (y + cos(t))x$ and "noise" $y(t) = cos(t)$.
    # The solution is $x(t) = x0 e^{2\sin(t)}$.
    # For vectorial unknown and vectorial noises, we simply repeat the equations and/or noise
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 2.0
    N = 2^8
    tt = range(t0, tf, length=N)
    @testset "scalar/scalar Euler" begin
        x0 = 0.5
        f = (t, x, y) -> ( y + cos(t) ) * x
        Yt = cos.(tt)
        Xt = Vector{Float64}(undef, N)
        sol = x0 * exp.( 2 * sin.(tt))
        @test_nowarn solve_euler!(rng, Xt, t0, tf, x0, f, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
    @testset "scalar/scalar Heun" begin
        x0 = 0.5
        f = (t, x, y) -> ( y + cos(t) ) * x
        Yt = cos.(tt)
        Xt = Vector{Float64}(undef, N)
        sol = x0 * exp.( 2 * sin.(tt))
        @test_nowarn solve_heun!(rng, Xt, t0, tf, x0, f, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
    @testset "scalar/vector Euler" begin
        x0 = 0.5
        f = (t, x, y) -> ( sum(y) + cos(t) ) * x
        Yt = [0.3 0.7] .* cos.(tt)
        Xt = Vector{Float64}(undef, N)
        sol = x0 * exp.( 2 * sin.(tt))
        @test_nowarn solve_euler!(rng, Xt, t0, tf, x0, f, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
    @testset "vector/scalar Euler" begin
        x0 = [0.2, 0.3]
        f! = (dx, t, x, y) -> (dx .= ( y + cos(t) ) * x)
        Yt = cos.(tt)
        Xt = Matrix{Float64}(undef, N, length(x0))
        sol = [x0[1] x0[2]] .* exp.( 2 * sin.(tt))
        @test_nowarn solve_euler!(rng, Xt, t0, tf, x0, f!, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
    @testset "vector/vector Euler" begin
        x0 = [0.2, 0.3]
        f! = (dx, t, x, y) -> (dx .= ( sum(y) + cos(t) ) * x)
        Yt = [0.2 0.2 0.6] .* cos.(tt)
        Xt = Matrix{Float64}(undef, N, length(x0))
        sol = [x0[1] x0[2]] .* exp.( 2 * sin.(tt))
        @test_nowarn solve_euler!(rng, Xt, t0, tf, x0, f!, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
end