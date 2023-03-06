@testset "Test noises" begin
    t0 = 0.0
    tf = 2.0
    n = 2^8
    m = 5_000
    ythf = Vector{Float64}(undef, m)
    ytf = Vector{Float64}(undef, m)
    yt = Vector{Float64}(undef, n)
    @testset "Wiener process" begin
        rng = Xoshiro(123)
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        
        @test eltype(noise) == Float64
        @test_nowarn rand!(rng, noise, yt)
        @test (@ballocated rand!($rng, $noise, $yt)) == 0
        @test_nowarn (@inferred rand!(rng, noise, yt))

        for mi in 1:m
            rand!(rng, noise, yt)
            ythf[mi] = yt[div(n, 2)]
            ytf[mi] = last(yt)
        end
        @test mean(ythf) ≈ y0 (atol = 0.1)
        @test var(ythf) ≈ tf/2 (atol = 0.1)
        @test mean(ytf) ≈ y0 (atol = 0.1)
        @test var(ytf) ≈ tf (atol = 0.1)
    end

    @testset "gBm process" begin
        rng = Xoshiro(123)
        y0 = 0.4
        μ = 0.3
        σ = 0.2
        noise = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)
        
        @test eltype(noise) == Float64
        @test_nowarn rand!(rng, noise, yt)
        @test (@ballocated $rand!($rng, $noise, $yt)) == 0
        @test_nowarn (@inferred rand!(rng, noise, yt))

        for mi in 1:m
            rand!(rng, noise, yt)
            ythf[mi] = yt[div(n, 2)]
            ytf[mi] = last(yt)
        end
        @test mean(ythf) ≈ y0 * exp(μ * (tf / 2)) (atol = 0.1)
        @test var(ythf) ≈ y0^2 * exp(2μ * (tf / 2)) * (exp(σ^2 * (tf / 2)) - 1) (atol = 0.1)
        @test mean(ytf) ≈ y0 * exp(μ * tf) (atol = 0.1)
        @test var(ytf) ≈ y0^2 * exp(2μ * tf) * (exp(σ^2 * tf) - 1) (atol = 0.1)
    end

    @testset "Compound Poisson" begin
        rng = Xoshiro(123)
        λ = 10.0
        μ = 0.5
        σ = 0.2
        dylaw = Normal(μ, σ)
        noise = CompoundPoissonProcess(t0, tf, λ, dylaw)
        
        @test eltype(noise) == Float64
        @test_nowarn rand!(rng, noise, yt)
        @test (@ballocated $rand!($rng, $noise, $yt)) == 0
        @test_nowarn (@inferred rand!(rng, noise, yt))
        
        for mi in 1:m
            rand!(rng, noise, yt)
            ythf[mi] = yt[div(n, 2)]
            ytf[mi] = last(yt)
        end
        @test mean(ythf) ≈ μ * λ * tf / 2 (atol = 0.1)
        @test var(ythf) ≈ λ * (tf/2) * ( μ^2 + σ^2 ) (rtol = 0.1)
        @test mean(ytf) ≈ μ * λ * tf (atol = 0.1)
        @test var(ytf) ≈ λ * tf * ( μ^2 + σ^2 ) (rtol = 0.1)
    end

    @testset "Step Poisson" begin
        rng = Xoshiro(123)
        λ = 10.0
        α = 2.0
        β = 15.0
        steplaw = Beta(α, β)
        noise = PoissonStepProcess(t0, tf, λ, steplaw)
        
        @test eltype(noise) == Float64
        @test_nowarn rand!(rng, noise, yt)
        @test (@ballocated $rand!($rng, $noise, $yt)) == 0
        @test_nowarn (@inferred rand!(rng, noise, yt))
        
        for mi in 1:m
            rand!(rng, noise, yt)
            ythf[mi] = yt[div(n, 2)]
            ytf[mi] = last(yt)
        end
        @test mean(ythf) ≈ α/(α + β) (atol = 0.1)
        @test var(ythf) ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.1)
        @test mean(ytf) ≈ α/(α + β) (atol = 0.1)
        @test var(ytf) ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.1)
    end

    @testset "Transport process" begin
        rng = Xoshiro(123)
        α = 2.0
        β = 15.0
        ylaw = Beta(α, β)
        nr = 50
        f = (t, r) -> mapreduce(ri -> sin(t/ri), +, r) / length(r)
        noise = TransportProcess(t0, tf, ylaw, f, nr)
        
        @test eltype(noise) == Float64
        @test_nowarn rand!(rng, noise, yt)
        @test (@ballocated $rand!($rng, $noise, $yt)) == 0
        @test_nowarn (@inferred rand!(rng, noise, yt))

        for mi in 1:m
            rand!(rng, noise, yt)
            ythf[mi] = yt[div(n, 2)]
            ytf[mi] = last(yt)
        end
        @test mean(ythf) ≈ mean(mean(sin(tf / 2r) for r in rand(rng, ylaw, nr)) for _ in 1:m) (atol = 0.1)
        @test var(ythf) ≈ var(mean(sin(tf / 2r) for r in rand(rng, ylaw, nr)) for _ in 1:m) (atol = 0.1)
        @test mean(ytf) ≈ mean(mean(sin(tf / r) for r in rand(rng, ylaw, nr)) for _ in 1:m) (atol = 0.1)
        @test var(ytf) ≈ var(mean(sin(tf / r) for r in rand(rng, ylaw, nr)) for _ in 1:m) (atol = 0.1)
    end

    @testset "fBm process" begin
        rng = Xoshiro(123)
        y0 = 0.0
        H = 0.25
        noise = FractionalBrownianMotionProcess(t0, tf, y0, H, n)
        
        @test eltype(noise) == Float64
        @test_nowarn rand!(rng, noise, yt)
        @test (@ballocated $rand!($rng, $noise, $yt)) == 0
        @test_nowarn (@inferred rand!(rng, noise, yt))
        
        for mi in 1:m
            rand!(rng, noise, yt)
            ythf[mi] = yt[div(n, 2)]
            ytf[mi] = last(yt)
        end
        @test mean(ythf) ≈ y0 (atol = 0.1)
        @test var(ythf) ≈ (tf/2)^(2H) (atol = 0.1)
        @test mean(ytf) ≈ y0 (atol = 0.1)
        @test var(ytf) ≈ tf^(2H) (atol = 0.1)
    end
    
    @testset "Product process I" begin
        rng = Xoshiro(123)
        y0 = 0.2
        noise = ProductProcess(
            WienerProcess(t0, tf, y0)
        )

        @test eltype(noise) == Float64
        
        ymt = Matrix{Float64}(undef, n, length(noise))
        ymtf = Matrix{Float64}(undef, m, length(noise))

        @test_nowarn rand!(rng, noise, ymt)

        @test (@ballocated rand!($rng, $noise, $ymt)) == 0
        
        @test_nowarn (@inferred rand!(rng, noise,ymt))

        for mi in 1:m
            rand!(rng, noise, ymt)
            for j in 1:length(noise)
                ymtf[mi, j] = ymt[end, j]
            end
        end
        means = mean(ymtf, dims=1)
        vars = var(ymtf, dims=1)
        
        @test means[1] ≈ y0 (atol = 0.1)
        @test vars[1] ≈ tf (atol = 0.1)
    end

    @testset "Product process II" begin
        rng = Xoshiro(123)
        y0 = 0.2
        μ = 0.3
        σ = 0.2
        λ = 10.0
        α = 2.0
        β = 15.0
        dylaw = Normal(μ, σ)
        steplaw = Beta(α, β)
        nr = 50
        f = (t, r) -> mapreduce(ri -> sin(t/ri), +, r) / length(r)
        ylaw = Beta(α, β)
        H = 0.25

        noise = ProductProcess(
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        )

        @test eltype(noise) == Float64

        ymt = Matrix{Float64}(undef, n, length(noise))
        ymtf = Matrix{Float64}(undef, m, length(noise))

        @test_nowarn rand!(rng, noise, ymt)

        @test (@ballocated rand!($rng, $noise, $ymt)) == 0
        
        @test_nowarn (@inferred rand!(rng, noise,ymt))

        noise = ProductProcess(
            WienerProcess(t0, tf, y0),
            GeometricBrownianMotionProcess(t0, tf, y0, μ, σ),
            CompoundPoissonProcess(t0, tf, λ, dylaw),
            PoissonStepProcess(t0, tf, λ, steplaw),
            TransportProcess(t0, tf, ylaw, f, nr),
            FractionalBrownianMotionProcess(t0, tf, y0, H, n)
        )

        @test eltype(noise) == Float64

        ymt = Matrix{Float64}(undef, n, length(noise))
        ymtf = Matrix{Float64}(undef, m, length(noise))

        @test_nowarn rand!(rng, noise, ymt)

        # `ProductProcess` was allocating a little when there were different types of processes in `ProductProcess`, but just an overhead, not affecting performance.
        # It could be due to failed inference and/or type instability.
        # Anyway, I changed it to a generated function with specialized rolled out loop and it is fine, now!
        @test (@ballocated rand!($rng, $noise, $ymt)) == 0
        
        @test_nowarn (@inferred rand!(rng, noise,ymt))

        for mi in 1:m
            rand!(rng, noise, ymt)
            for j in 1:length(noise)
                ymtf[mi, j] = ymt[end, j]
            end
        end
        means = mean(ymtf, dims=1)
        vars = var(ymtf, dims=1)
        
        @test means[1] ≈ y0 (atol = 0.1)
        @test vars[1] ≈ tf (atol = 0.1)

        @test means[2] ≈ y0 * exp(μ * tf) (atol = 0.1)
        @test vars[2] ≈ y0^2 * exp(2μ * tf) * (exp(σ^2 * tf) - 1) (atol = 0.1)

        @test means[3] ≈ μ * λ * tf (atol = 0.1)
        @test vars[3] ≈ λ * tf * ( μ^2 + σ^2 ) (rtol = 0.1)

        @test means[4] ≈ α/(α + β) (atol = 0.1)
        @test vars[4] ≈ α*β/(α + β)^2/(α + β + 1) (atol = 0.1)

        @test means[5] ≈ mean(mean(sin(tf / r) for r in rand(rng, ylaw, nr)) for _ in 1:m) (atol = 0.1)
        @test vars[5] ≈ var(mean(sin(tf / r) for r in rand(rng, ylaw, nr)) for _ in 1:m) (atol = 0.1)

        @test means[6] ≈ y0 (atol = 0.1)
        @test vars[6] ≈ tf^(2H) (atol = 0.1)
    end
end