@testset "Test noises" begin
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 2.0
    N = 2^8
    M = 5_000
    Ythf = Vector{Float64}(undef, M)
    Ytf = Vector{Float64}(undef, M)
    Yt = Vector{Float64}(undef, N)
    @testset "Wiener process" begin
        y0 = 0.0
        noise! = Wiener_noise(t0, tf, y0)
        @test_nowarn noise!(rng, Yt)
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

    @testset "GBM process" begin
        y0 = 0.4
        μ = 0.3
        σ = 0.2
        noise! = GBM_noise(t0, tf, μ, σ, y0)
        @test_nowarn noise!(rng, Yt)
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

    @testset "Compound Poisson" begin
        λ = 25.0
        μ = 0.5
        σ = 0.2
        dN = Normal(μ, σ)
        noise! = CompoundPoisson_noise(t0, tf, λ, dN)
        @test_nowarn noise!(rng, Yt)
        for m in 1:M
            noise!(rng, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ μ * λ * tf / 2 (atol = 0.1)
        @test var(Ythf) ≈ λ * (tf/2) * ( μ^2 + σ^2 ) (rtol = 0.1)
        @test mean(Ytf) ≈ μ * λ * tf (atol = 0.1)
        @test var(Ytf) ≈ λ * tf * ( μ^2 + σ^2 ) (rtol = 0.1)
    end

    @testset "Step Poisson" begin
        λ = 25.0
        α = 2.0
        β = 15.0
        S = Beta(α, β)
        noise! = StepPoisson_noise(t0, tf, λ, S)
        @test_nowarn noise!(rng, Yt)
        for m in 1:M
            noise!(rng, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ α/(α + β) (atol = 0.05)
        @test var(Ythf) ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.05)
        @test mean(Ytf) ≈ α/(α + β) (atol = 0.1)
        @test var(Ytf) ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.05)
    end

    @testset "Transport process" begin
        nr = 5
        f = (t, r) -> sum(sin(r[i] * t) for i in 1:nr)
        α = 2.0
        β = 15.0
        RV = Beta(α, β)
        noise! = Transport_noise(t0, tf, f, RV, nr)
        @test_nowarn noise!(rng, Yt)
        for m in 1:M
            noise!(rng, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ mean(sum(sin(r * tf / 2) for r in rand(rng, RV, nr)) for _ in 1:M) (atol = 0.01)
        @test var(Ythf) ≈ var(sum(sin(r * tf / 2) for r in rand(rng, RV, nr)) for _ in 1:M) (atol = 0.01)
        @test mean(Ytf) ≈ mean(sum(sin(r * tf) for r in rand(rng, RV, nr)) for _ in 1:M) (atol = 0.01)
        @test var(Ytf) ≈ var(sum(sin(r * tf) for r in rand(rng, RV, nr)) for _ in 1:M) (atol = 0.01)
    end

    @testset "fBm process" begin
        y0 = 0.0
        H = 0.25
        noise! = fBm_noise(t0, tf, y0, H, N)
        @test_nowarn noise!(rng, Yt)
        for m in 1:M
            noise!(rng, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ 0.0 (atol = 0.05)
        @test var(Ythf) ≈ (tf/2)^(2H) (atol = 0.05)
        @test mean(Ytf) ≈ 0.0 (atol = 0.05)
        @test var(Ytf) ≈ tf^(2H) (atol = 0.05)
        rngcp = copy(rng)
        @test RODEConvergence.fG_daviesharte(rng, tf, N, H) ≈ RODEConvergence.fG_daviesharte_naive(rngcp, tf, N, H)
    end
end