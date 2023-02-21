@testset "Test solvers" begin
    # We test the solvers with the equation $x' = f(t, x, y)$
    # and initial condition $x(0) = x_0$, where
    # $f(t, x, y) = (y + cos(t))x$ and "noise" $y(t) = cos(t)$.
    # The solution is $x(t) = x0 e^{2\sin(t)}$.
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 2.0
    x0 = 0.5
    f = (t, x, y) -> ( y + cos(t) ) * x
    N = 2^8
    tt = range(t0, tf, length=N)
    Yt = cos.(tt)
    Xt = Vector{Float64}(undef, N)
    sol = x0 * exp.( 2 * sin.(tt))
    @testset "Euler" begin
        @test_nowarn solve_euler!(rng, Xt, t0, tf, x0, f, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
    @testset "Heun" begin
        @test_nowarn solve_heun!(rng, Xt, t0, tf, x0, f, Yt)
        @test maximum(abs, Xt .- sol) < 0.05
    end
end