target_exact1! = function (xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractVector{T}, rng::AbstractRNG) where {T, F}
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

target_exact2! = function (xt::AbstractVector{T}, t0::T, tf::T, x0::T, f::F, yt::AbstractMatrix{T}, rng::AbstractRNG) where {T, F}
    ntgt = size(xt, 1)
    dt = (tf - t0) / (ntgt - 1)
    xt[1] = x0
    integral = 0.0
    for n in 2:ntgt
        integral += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
        xt[n] = x0 * exp(integral)
    end
end

target_exact3! = function (xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractVector{T}, rng::AbstractRNG) where {T, F}
    ntgt = size(xt, 1)
    dt = (tf - t0) / (ntgt - 1)
    for j in eachindex(axes(xt, 2), x0)
        xt[1, j] = x0[j]
    end
    integral = 0.0
    for n in 2:ntgt
        integral += (yt[n] + yt[n-1]) * dt / 2 + sqrt(dt^3 / 12) * randn(rng)
        xt[n, :] .= exp(integral) .* x0
    end
end

target_exact4! = function (xt::AbstractMatrix{T}, t0::T, tf::T, x0::AbstractVector{T}, f::F, yt::AbstractMatrix{T}, rng::AbstractRNG) where {T, F}
    ntgt = size(xt, 1)
    dt = (tf - t0) / (ntgt - 1)
    for j in eachindex(axes(xt, 2), x0)
        xt[1, j] = x0[j]
    end
    integral = 0.0
    for n in 2:ntgt
        integral += (yt[n, 1] + yt[n-1, 1] + 3yt[n, 2] + 3yt[n-1, 2]) * dt / 8 + sqrt(dt^3 / 12) * randn(rng)
        xt[n, :] .= exp(integral) .* x0
    end
end

@testset "No alloc conv" begin
    rng = Xoshiro(123)
    t0 = 0.0
    tf = 1.0
    ntgt = 2^4
    ns = 2 .^ (2:3)
    m = 100
    trajerrors = zeros(last(ns), length(ns))

    @testset "Scalar/Scalar Euler" begin

        x0law = Normal()
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f = (t, x, y) -> y * x

        target = CustomUnivariateMethod(target_exact1!, rng)
        method = RandomEuler()

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
        @test (@ballocated RODEConvergence.calculate_trajerrors!($rng, $trajerrors, $suite)) == 0
    end

    @testset "Scalar/Vector Euler" begin

        x0law = Normal()
        y0 = 0.0
        noise = ProductProcess(
            WienerProcess(t0, tf, y0),
            WienerProcess(t0, tf, y0)
        )
        f = (t, x, y) -> (y[1] + 3y[2])/4 * x

        target = CustomUnivariateMethod(target_exact2!, rng)
        method = RandomEuler()

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f, noise, target, method, ntgt, ns, m)
        @test (@ballocated RODEConvergence.calculate_trajerrors!($rng, $trajerrors, $suite)) == 0
    end

    @testset "Vector/scalar Euler" begin
        # two independent normals
        # but careful not to use `MvNormal(I(2))` because that uses a FillArray Zeros(2) for the means
        # which allocates due to a bug with broadcasting
        # see https://github.com/JuliaArrays/FillArrays.jl/issues/208
        x0law = MvNormal(zeros(2), I(2))
        # alternative implementation:
        # X0law = product_distribution(Normal(), Normal())
        y0 = 0.0
        noise = WienerProcess(t0, tf, y0)
        f! = (dx, t, x, y) -> (dx .= y .* x)

        target = CustomMultivariateMethod(target_exact3!, rng)
        method = RandomEuler(length(x0law))

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
        @test (@ballocated RODEConvergence.calculate_trajerrors!($rng, $trajerrors, $suite)) == 0
    end

    @testset "Vector/Vector Euler" begin
        # two independent normals
        # but careful not to use `MvNormal(I(2))` because that uses a FillArray Zeros(2) for the means
        # which allocates due to a bug with broadcasting
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

        target = CustomMultivariateMethod(target_exact4!, rng)
        method = RandomEuler(length(x0law))

        suite = @test_nowarn ConvergenceSuite(t0, tf, x0law, f!, noise, target, method, ntgt, ns, m)
        @test (@ballocated RODEConvergence.calculate_trajerrors!($rng, $trajerrors, $suite)) == 0
    end
end
