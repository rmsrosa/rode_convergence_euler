@testset "Test convergence" begin
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 1.0
    ntgt = 2^14
    ns = 2 .^ (4:7)
    m = 500

    @testset "Scalar/Scalar Euler" begin

        x0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x

        target_exact! = function (rng, xt, t0, tf, x0, f, yt)
            ntgt = size(xt, 1)
            dt = (tf - t0) / (ntgt - 1)
            xt[1] = x0
            integral = 0.0
            for n in 2:ntgt
                integral += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(integral)
            end
        end
        
        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)
    end

    @testset "Scalar/Vector Euler" begin

        x0law = Normal()
        y0 = 0.0
        noise = ProductProcess(
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        )
        f = (t, x, y) -> (y[1] + 3y[2])/4 * x

        target_exact! = function (rng, xt, t0, tf, x0, f, yt)
            ntgt = size(xt, 1)
            dt = (tf - t0) / (ntgt - 1)
            xt[1] = x0
            integral = 0.0
            for n in 2:ntgt
                integral += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(integral)
            end
        end

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)
    end

    @testset "Vector/scalar Euler" begin
        # two independent normals
        # but careful not to use `MvNormal(I(2))` because that uses a FillArray Zeros(2) for the means
        # which allocates dut to trouble with broadcasting
        # see https://github.com/JuliaArrays/FillArrays.jl/issues/208
        x0law = MvNormal(zeros(2), I(2))
        # alternative implementation:
        # X0law = product_distribution(Normal(), Normal())
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f! = (dx, t, x, y) -> (dx .= y * x)

        target_exact! = function (rng, xt, t0, tf, x0, f!, yt)
            ntgt = size(xt, 1)
            dt = (tf - t0) / (ntgt - 1)
            xt[1, :] .= x0
            integral = 0.0
            for n in 2:ntgt
                integral += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n, :] .= exp(integral) * x0
            end
        end

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)
    end

    @testset "Vector/Vector Euler" begin
        # two independent normals
        # two independent normals
        # but careful not to use `MvNormal(I(2))` because that uses a FillArray Zeros(2) for the means
        # which allocates dut to trouble with broadcasting
        # see https://github.com/JuliaArrays/FillArrays.jl/issues/208
        x0law = MvNormal(zeros(2), I(2))
        # alternative implementation:
        # X0law = product_distribution(Normal(), Normal())
        y0 = 0.0
        noise = ProductProcess(
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        )
        f! = (dx, t, x, y) -> (dx .= (y[1] + 3y[2]) * x / 4)

        target_exact! = function (rng, xt, t0, tf, x0, f!, yt)
            ntgt = size(xt, 1)
            dt = (tf - t0) / (ntgt - 1)
            xt[1, :] .= x0
            integral = 0.0
            for n in 2:ntgt
                integral += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
                xt[n, :] .= exp(integral) * x0
            end
        end

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)
    end

    t0 = 0.0
    tf = 2.0
    ntgt = 2^16
    ns = 2 .^ (5:8)
    m = 500

    @testset "Scalar/Scalar Euler 2" begin

        x0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x / 10

        target_exact! = function (rng, xt, t0, tf, x0, f, yt)
            ntgt = size(xt, 1)
            dt = (tf - t0) / (ntgt - 1)
            xt[1] = x0
            integral = 0.0
            for n in 2:ntgt
                integral += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(integral / 10)
            end
        end

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target_exact!, solve_euler!, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test results.deltas ≈ (tf - t0) ./ (ns .- 1)
        @test results.p ≈ 1.0 (atol = 0.1)
    end
end