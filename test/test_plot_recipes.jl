custom_solver = function(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, rng::AbstractRNG) where {T, F}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )

    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    integral = zero(T)
    for i in Iterators.drop(eachindex(xt, yt), 1)
        integral += (yt[i] + yt[i1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
        xt[i] = x0 * exp(integral)
        i1 = i
    end
end

@testset "Plot recipes" begin
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 1.0
    ntgt = 2^12
    ns = 2 .^ (4:8)
    m = 1

    @testset "Scalar/Scalar Euler" begin

        x0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x

        target = CustomUnivariateMethod(custom_solver, rng)
        method = RandomEuler()

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test_nowarn plot(suite)
        @test_nowarn plot(suite, shownoise=true)
        @test_nowarn plot(suite, shownoise=true, showtarget=true, showapprox=false)
        @test_nowarn plot(suite, ns=[64, 128, 160, 192])
    end

    @testset "Scalar/Vector Euler" begin

        x0law = Normal()
        y0 = 0.0
        noise = ProductProcess(
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        )
        f = (t, x, y) -> (y[1] + 3y[2])/4 * x

        target = RandomEuler()
        method = RandomEuler()

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        
        @test_nowarn plot(suite)
        @test_nowarn plot(suite, shownoise=true)
        @test_nowarn plot(suite, shownoise=true, showtarget=true, showapprox=false)
        @test_nowarn plot(suite, ns=[64, 128, 160, 192])
    end

    @testset "Vector/scalar Euler" begin
        # two independent normals
        # but careful not to use `MvNormal(I(2))` because that uses a FillArray Zeros(2) for the means
        # which allocates due to trouble with broadcasting
        # see https://github.com/JuliaArrays/FillArrays.jl/issues/208
        x0law = MvNormal(zeros(2), I(2))
        # alternative implementation:
        # X0law = product_distribution(Normal(), Normal())
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f! = (dx, t, x, y) -> (dx .= y .* x)

        target = RandomEuler(length(x0law))
        method = RandomEuler(length(x0law))

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test_nowarn plot(suite)
        @test_nowarn plot(suite, shownoise=true)
        @test_nowarn plot(suite, shownoise=true, showtarget=true, showapprox=false)
        @test_nowarn plot(suite, ns=[64, 128, 160, 192])
        @test_nowarn plot(suite, ns=[64, 128, 160, 192])
        @test_nowarn plot(suite, ns=[64, 128, 160, 192], idxs=2)
        @test_nowarn plot(suite, ns=[64, 128, 160, 192], idxs=1:2)
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
        f! = (dx, t, x, y) -> (dx .= (y[1] + 3y[2]) .* x ./ 4)

        target = RandomEuler(length(x0law))
        method = RandomEuler(length(x0law))

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
        results = @test_nowarn solve(rng, suite)
        @test_nowarn plot(suite)
        @test_nowarn plot(suite, shownoise=true)
        @test_nowarn plot(suite, shownoise=true, showtarget=true, showapprox=false)
        @test_nowarn plot(suite, ns=[64, 128, 160, 192])
        @test_nowarn plot(suite, ns=[64, 128, 160, 192], idxs=2)
        @test_nowarn plot(suite, ns=[64, 128, 160, 192], idxs=1:2)
    end
end
