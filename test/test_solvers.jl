# We test the solvers with the equation $x' = f(t, x, y)$
# and initial condition $x(0) = x_0$, where
# $f(t, x, y) = (y + cos(t))x$ and "noise" $y(t) = cos(t)$.
# The solution is $x(t) = x0 e^{2\sin(t)}$.
# For vectorial unknown and vectorial noises, we simply repeat the equations and/or noise

# This is for testing the CustomMethod at the end of the test.
# There is some scoping problem that gives me extra allocations if I define this inside the test
custom_solver = function(xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, params::Q, solver_params::P) where {T, F, P, Q}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )

    n = length(xt) - 1
    dt = (tf - t0) / n
    i1 = firstindex(xt)
    xt[i1] = x0
    integral = 0.0
    λ = params[1]
    for i in Iterators.drop(eachindex(xt, yt), 1)
        integral += (yt[i] + yt[i1]) * dt / 2
        if solver_params isa AbstractRNG
            integral += sqrt(dt^3 / 12) * randn(solver_params)
        end
        xt[i] = x0 * exp(λ * integral)
        i1 = i
    end
end

@testset "Solvers" begin
    t0 = 0.0
    tf = 1.5
    n = 2^8
    tt = range(t0, tf, length=n+1)
    @testset "scalar/scalar" begin
        x0 = 0.5
        k = 2
        f = (t, x, y, p) -> ( y + (p[1] - 1) * cos(t) ) * x
        yt = cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        sol = x0 * exp.( k * sin.(tt))
        params = (k,)

        for method in (RandomEuler(), RandomHeun(), CustomUnivariateMethod(custom_solver, nothing))
            @testset "$(nameof(typeof(method)))" begin
                @test_nowarn solve!(xt, t0, tf, x0, f, yt, params, method)
                @test maximum(abs, xt .- sol) < 0.05
                @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $params, $method)) == 0
            end
        end
    end

    @testset "scalar/vector" begin
        x0 = 0.5
        f = (t, x, y, p) -> ( sum(y) + cos(t) ) * x
        yt = [0.3 0.7] .* cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        sol = x0 * exp.( 2 * sin.(tt))
        params = nothing

        for method in (RandomEuler(), RandomHeun())
            @testset "$(nameof(typeof(method)))" begin
                @test_nowarn solve!(xt, t0, tf, x0, f, yt, params, method)
                @test maximum(abs, xt .- sol) < 0.05
                @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $params, $method)) == 0
            end
        end
    end

    @testset "vector/scalar" begin
        x0 = [0.2, 0.3]
        f! = (dx, t, x, y, p) -> (dx .= ( y + cos(t) ) .* x)
        yt = cos.(tt)
        xt = Matrix{Float64}(undef, n + 1, length(x0))
        sol = [x0[1] x0[2]] .* exp.( 2 * sin.(tt))
        params = nothing

        for method in (RandomEuler(2), RandomHeun(2))
            @testset "$(nameof(typeof(method)))" begin
                @test_nowarn solve!(xt, t0, tf, x0, f!, yt, params, method)
                @test maximum(abs, xt .- sol) < 0.05
                @test (@ballocated solve!($xt, $t0, $tf, $x0, $f!, $yt, $params, $method)) == 0
            end
        end
    end

    @testset "vector/vector" begin
        x0 = [0.2, 0.3]
        f! = (dx, t, x, y, p) -> (dx .= ( sum(y) + cos(t) ) .* x)
        yt = [0.2 0.2 0.6] .* cos.(tt)
        xt = Matrix{Float64}(undef, n + 1, length(x0))
        sol = [x0[1] x0[2]] .* exp.( 2 * sin.(tt))
        params = nothing
        
        for method in (RandomEuler(2), RandomHeun(2))
            @testset "$(nameof(typeof(method)))" begin
                @test_nowarn solve!(xt, t0, tf, x0, f!, yt, params, method)
                @test maximum(abs, xt .- sol) < 0.05
                @test (@ballocated solve!($xt, $t0, $tf, $x0, $f!, $yt, $params, $method)) == 0
            end
        end
    end

    @testset "scalar/scalar 2" begin
        rng = Xoshiro(123)
        x0 = 0.5
        λ = 1.0
        f = (t, x, y, p) -> p[1] * y * x
        yt = cos.(tt)
        xt = Vector{Float64}(undef, n + 1)
        params = (λ,)
        sol = x0 * exp.( λ * sin.(tt))

        for method in (RandomEuler(), RandomHeun(), CustomUnivariateMethod(custom_solver, nothing))
            @testset "$(nameof(typeof(method)))" begin
                @test_nowarn solve!(xt, t0, tf, x0, f, yt, params, method)
                @test maximum(abs, xt .- sol) < 0.05
                @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $params, $method)) == 0
            end
        end
    end

    @testset "scalar/scalar 3" begin
        rng = Xoshiro(123)    
        x0 = 0.5
        λ = 1.0
        f = (t, x, y, p) -> p[1] * y * x
        noise = WienerProcess(t0, tf, 0.0)
        yt = Vector{Float64}(undef, n + 1)
        rand!(rng, noise, yt)
        xt = Vector{Float64}(undef, n + 1)
        params = (λ,)
        dt = (tf - t0) / n
        sol = x0 * exp.( λ * cumsum(yt) * dt)

        for method in (RandomEuler(), RandomHeun(), CustomUnivariateMethod(custom_solver, nothing))
            @testset "$(nameof(typeof(method)))" begin
                @test_nowarn solve!(xt, t0, tf, x0, f, yt, params, method)
                @test maximum(abs, xt .- sol) < 0.05
                @test (@ballocated solve!($xt, $t0, $tf, $x0, $f, $yt, $params, $method)) == 0
            end
        end
    end
end
