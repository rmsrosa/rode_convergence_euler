@testset "Test noises" begin
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
        Y = WienerProcess(t0, tf, y0)
        
        @test_nowarn rand!(rng, Y, Yt)
        @test (@ballocated rand!($rng, $Y, $Yt)) == 0
        @test_nowarn (@inferred rand!(rng, Y, Yt))

        for m in 1:M
            rand!(rng, Y, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ y0 (atol = 0.1)
        @test var(Ythf) ≈ tf/2 (atol = 0.1)
        @test mean(Ytf) ≈ y0 (atol = 0.1)
        @test var(Ytf) ≈ tf (atol = 0.1)
    end

    @testset "gBm process" begin
        rng = Xoshiro(123)
        y0 = 0.4
        μ = 0.3
        σ = 0.2
        Y = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)
        
        @test_nowarn rand!(rng, Y, Yt)
        @test (@ballocated $rand!($rng, $Y, $Yt)) == 0
        @test_nowarn (@inferred rand!(rng, Y, Yt))

        for m in 1:M
            rand!(rng, Y, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ y0 * exp(μ * (tf / 2)) (atol = 0.1)
        @test var(Ythf) ≈ y0^2 * exp(2μ * (tf / 2)) * (exp(σ^2 * (tf / 2)) - 1) (atol = 0.1)
        @test mean(Ytf) ≈ y0 * exp(μ * tf) (atol = 0.1)
        @test var(Ytf) ≈ y0^2 * exp(2μ * tf) * (exp(σ^2 * tf) - 1) (atol = 0.1)
    end

    @testset "Compound Poisson" begin
        rng = Xoshiro(123)
        λ = 10.0
        μ = 0.5
        σ = 0.2
        dYlaw = Normal(μ, σ)
        Y = CompoundPoissonProcess(t0, tf, λ, dYlaw)
        
        @test_nowarn rand!(rng, Y, Yt)
        @test (@ballocated $rand!($rng, $Y, $Yt)) == 0
        @test_nowarn (@inferred rand!(rng, Y, Yt))
        
        for m in 1:M
            rand!(rng, Y, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ μ * λ * tf / 2 (atol = 0.1)
        @test var(Ythf) ≈ λ * (tf/2) * ( μ^2 + σ^2 ) (rtol = 0.1)
        @test mean(Ytf) ≈ μ * λ * tf (atol = 0.1)
        @test var(Ytf) ≈ λ * tf * ( μ^2 + σ^2 ) (rtol = 0.1)
    end

    @testset "Step Poisson" begin
        rng = Xoshiro(123)
        λ = 10.0
        α = 2.0
        β = 15.0
        Slaw = Beta(α, β)
        Y = PoissonStepProcess(t0, tf, λ, Slaw)
        
        @test_nowarn rand!(rng, Y, Yt)
        @test (@ballocated $rand!($rng, $Y, $Yt)) == 0
        @test_nowarn (@inferred rand!(rng, Y, Yt))
        
        for m in 1:M
            rand!(rng, Y, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ α/(α + β) (atol = 0.1)
        @test var(Ythf) ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.1)
        @test mean(Ytf) ≈ α/(α + β) (atol = 0.1)
        @test var(Ytf) ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.1)
    end

    @testset "Transport process" begin
        rng = Xoshiro(123)
        α = 2.0
        β = 15.0
        Ylaw = Beta(α, β)
        f = (t, r) -> mapreduce(ri -> sin(ri*t), +, r)
        ny = 5
        Y = TransportProcess(t0, tf, Ylaw, f, ny)
        
        @test_nowarn rand!(rng, Y, Yt)
        @test (@ballocated $rand!($rng, $Y, $Yt)) == 0
        @test_nowarn (@inferred rand!(rng, Y, Yt))

        for m in 1:M
            rand!(rng, Y, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ mean(sum(sin(r * tf / 2) for r in rand(rng, Ylaw, ny)) for _ in 1:M) (atol = 0.1)
        @test var(Ythf) ≈ var(sum(sin(r * tf / 2) for r in rand(rng, Ylaw, ny)) for _ in 1:M) (atol = 0.1)
        @test mean(Ytf) ≈ mean(sum(sin(r * tf) for r in rand(rng, Ylaw, ny)) for _ in 1:M) (atol = 0.1)
        @test var(Ytf) ≈ var(sum(sin(r * tf) for r in rand(rng, Ylaw, ny)) for _ in 1:M) (atol = 0.1)
    end

    @testset "fBm process" begin
        rng = Xoshiro(123)
        y0 = 0.0
        H = 0.25
        Y = FractionalBrownianMotionProcess(t0, tf, y0, H, N)
        
        @test_nowarn rand!(rng, Y, Yt)
        @test (@ballocated $rand!($rng, $Y, $Yt)) == 0
        @test_nowarn (@inferred rand!(rng, Y, Yt))
        
        for m in 1:M
            rand!(rng, Y, Yt)
            Ythf[m] = Yt[div(N, 2)]
            Ytf[m] = last(Yt)
        end
        @test mean(Ythf) ≈ y0 (atol = 0.1)
        @test var(Ythf) ≈ (tf/2)^(2H) (atol = 0.1)
        @test mean(Ytf) ≈ y0 (atol = 0.1)
        @test var(Ytf) ≈ tf^(2H) (atol = 0.1)
    end
    
    @testset "Multi noise" begin
        rng = Xoshiro(123)
        y0 = 0.2
        μ = 0.3
        σ = 0.2
        λ = 10.0
        α = 2.0
        β = 15.0
        dYlaw = Normal(μ, σ)
        Slaw = Beta(α, β)
        nr = 5
        f = (t, r) -> mapreduce(ri -> sin(ri*t), +, r)
        Ylaw = Beta(α, β)
        H = 0.25
        Y = [
            WienerProcess(t0, tf, y0),
            GeometricBrownianMotionProcess(t0, tf, y0, μ, σ),
            CompoundPoissonProcess(t0, tf, λ, dYlaw),
            PoissonStepProcess(t0, tf, λ, Slaw),
            TransportProcess(t0, tf, Ylaw, f, nr),
            FractionalBrownianMotionProcess(t0, tf, y0, H, N)
        ]

        YMt = Matrix{Float64}(undef, N, length(Y))
        YMtf = Matrix{Float64}(undef, M, length(Y))

        @test_nowarn rand!(rng, Y, YMt)

        # `MultiProcess_noise`` is allocating a little but it is not affecting performance and might just be due to closure behaving finicky sometimes
        # (per Jerry Ling (Moelf) https://github.com/Moelf on Slack)
        @test_broken (@ballocated rand!($rng, $Y, $YMt)) == 0
        
        @test_nowarn (@inferred rand!(rng, Y,YMt))

        for m in 1:M
            rand!(rng, Y, YMt)
            for j in 1:length(Y)
                YMtf[m, j] = YMt[end, j]
            end
        end
        means = mean(YMtf, dims=1)
        vars = var(YMtf, dims=1)
        
        @test means[1] ≈ y0 (atol = 0.1)
        @test vars[1] ≈ tf (atol = 0.1)

        @test means[2] ≈ y0 * exp(μ * tf) (atol = 0.1)
        @test vars[2] ≈ y0^2 * exp(2μ * tf) * (exp(σ^2 * tf) - 1) (atol = 0.1)

        @test means[3] ≈ μ * λ * tf (atol = 0.1)
        @test vars[3] ≈ λ * tf * ( μ^2 + σ^2 ) (rtol = 0.1)

        @test means[4] ≈ α/(α + β) (atol = 0.1)
        @test vars[4] ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.1)

        @test means[5] ≈ mean(sum(sin(r * tf) for r in rand(rng, Ylaw, nr)) for _ in 1:M) (atol = 0.1)
        @test vars[5] ≈ var(sum(sin(r * tf) for r in rand(rng, Ylaw, nr)) for _ in 1:M) (atol = 0.1)

        @test means[6] ≈ y0 (atol = 0.1)
        @test vars[6] ≈ tf^(2H) (atol = 0.1)
    end
end