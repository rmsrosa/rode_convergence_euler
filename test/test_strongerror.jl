@testset "Strong error" begin
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 1.0
    Ntgt = 2^14
    Ns = 2 .^ (4:7)
    M = 500

    @testset "Scalar/Scalar Euler" begin

        X0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x

        target_exact! = function (rng, xt, t0, tf, x0, f, yt)
            Ntgt = size(xt, 1)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1] = x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(It)
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target_exact!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target_approx!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end

    @testset "Scalar/Vector Euler" begin

        X0law = Normal()
        y0 = 0.0
        noise = [
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        ]
        f = (t, x, y) -> (y[1] + 3y[2])/4 * x

        target_exact! = function (rng, xt, t0, tf, x0, f, yt)
            Ntgt = size(xt, 1)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1] = x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(It)
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target_exact!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target_approx!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end

    @testset "Vector/scalar Euler" begin
        X0law = product_distribution(Normal(), Normal())
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f! = (dx, t, x, y) -> (dx .= y * x)

        target_exact! = function (rng, xt, t0, tf, x0, f!, yt)
            Ntgt = size(xt, 1)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1, :] .= x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n, :] .= exp(It) * x0
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f!, noise, target_exact!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f!, noise, target_approx!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end

    @testset "Vector/Vector Euler" begin
        X0law = product_distribution(Normal(), Normal())
        y0 = 0.0
        noise = [
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        ]
        f! = (dx, t, x, y) -> (dx .= (y[1] + 3y[2]) * x / 4)

        target_exact! = function (rng, xt, t0, tf, x0, f!, yt)
            Ntgt = size(xt, 1)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1, :] .= x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
                xt[n, :] .= exp(It) * x0
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f!, noise, target_exact!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)

        target_approx! = solve_euler!

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f!, noise, target_approx!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end

    t0 = 0.0
    tf = 2.0
    Ntgt = 2^16
    Ns = 2 .^ (5:8)
    M = 500

    @testset "Scalar/Scalar Euler 2" begin

        X0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x / 10

        target_exact! = function (rng, xt, t0, tf, x0, f, yt)
            Ntgt = size(xt, 1)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1] = x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(It / 10)
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target_exact!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end
end