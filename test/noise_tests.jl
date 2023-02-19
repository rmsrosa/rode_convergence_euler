
@testset "Noise tests" begin
    t0 = 0.0
    tf = 2.0
    N = 2^8
    M = 5_000
    Ythf = Vector{Float64}(undef, M)
    Ytf = Vector{Float64}(undef, M)
    Yt = Vector{Float64}(undef, N)
    @testset "Wiener process" begin
        rng = Xoshiro(123)
        y0 = 0.0
        noise! = Wiener_noise(t0, tf, y0)
        for m in 1:M
            noise!(rng, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ 0.0 (atol = 0.05)
        @test var(Ythf) ≈ tf/2 (atol = 0.05)
        @test mean(Ytf) ≈ 0.0 (atol = 0.05)
        @test var(Ytf) ≈ tf (atol = 0.05)
    end

    @testset "gBM process" begin
        rng = Xoshiro(123)
        y0 = 0.5
        μ = 0.4
        σ = 0.2
        noise! = GBM_noise(t0, tf, μ, σ, y0)
        for m in 1:M
            noise!(rng, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ y0 * exp(μ * (tf / 2)) (atol = 0.05)
        @test var(Ythf) ≈ y0^2 * exp(2μ * (tf / 2)) * (exp(σ^2 * (tf / 2)) - 1) (atol = 0.05)
        @test mean(Ytf) ≈ y0 * exp(μ * tf) (atol = 0.1)
        @test var(Ytf) ≈ y0^2 * exp(2μ * tf) * (exp(σ^2 * tf) - 1) (atol = 0.05)
    end

    @test RODEConvergence.fG_daviesharte(Xoshiro(123), 1.0, 2^10, 0.2) ≈ RODEConvergence.fG_daviesharte_naive(Xoshiro(123), 1.0, 2^10, 0.2)
end