@testset "Strong error" begin
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 1.0
    Ntgt = 2^16
    Ns = 2 .^ (4:8)
    M = 1_000

    @testset "Scalar/Scalar Euler case" begin

        X0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x

        target! = function (rng, xt, t0, tf, x0, f, yt)
            Ntgt = length(yt)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1] = x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(It)
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end

    @testset "Scalar/Vector Euler case" begin

        X0law = Normal()
        y0 = 0.0
        noise = [
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        ]
        f = (t, x, y) -> (y[1] + 3y[2])/4 * x

        target! = function (rng, xt, t0, tf, x0, f, yt)
            Ntgt = length(xt)
            dt = (tf - t0) / (Ntgt - 1)
            xt[1] = x0
            It = 0.0
            for n in 2:Ntgt
                It += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
                xt[n] = x0 * exp(It)
            end
        end

        deltas, errors, trajerrors, lc, p = @test_nowarn calculate_errors(rng, t0, tf, X0law, f, noise, target!, Ntgt, Ns, M)

        @test deltas ≈ (tf - t0) ./ (Ns .- 1)
        @test p ≈ 1.0 (atol = 0.1)
    end

    @testset "Vector/scalar Euler case" begin
    end

    @testset "Vector/Vector Euler case" begin
    end
end