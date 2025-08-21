# # Testing the confidence intervals with a homogenous linear RODE with a Wiener process noise coefficient
#
# We consider a homogeneous linear equation in which the coefficient is a Wiener process. In this case, it is already know, by other means, that the Euler method converges strongly of order 1, because it can be regarded as system of stochastic differential equations with additive noise. Nevertheless, we use it here for testing purposes.

# ## The equation

# We consider the RODE
# ```math
#   \begin{cases}
#     \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = W_t X_t, \qquad 0 \leq t \leq T, \\
#   \left. X_t \right|_{t = 0} = X_0,
#   \end{cases}
# ```
# where $\{W_t\}_{t\geq 0}$ is a standard Wiener process.
# The explicit solution is
# ```math
#   X_t = e^{\int_0^t W_s \;\mathrm{d}s} X_0.
# ```

# ## Computing a solution with the exact distribution
#
# As seen in the first example, once an Euler approximation is computed, along with realizations $\{W_{t_i}\}_{i=0}^n$ of a sample path of the noise, we consider an exact sample solution given by
# ```math
#     X_{t_j} = X_0 e^{\sum_{i = 0}^{j-1}\left(\frac{1}{2}\left(W_{t_i} + W_{t_{i+1}}\right)(t_{i+1} - t_i) + Z_i\right)},
# ```
# for realizations $Z_i$ drawn from a normal distribution and scaled by the standard deviation $\sqrt{(t_{i+1} - t_i)^3/12}$. This is implemented by computing the integral recursively, via
# ```math
#     \begin{cases}
#         I_j = I_{j-1} + \frac{1}{2}\left(W_{t_{j-1}} + W_{t_j}\right)(t_{j} - t_{j-1}) + Z_j, \\
#         Z_j = \sqrt{\frac{(t_{j} - t_{j-1})^3}{12}} R_j, \\
#         R_j \sim \mathcal{N}(0, 1), \\
#     \end{cases}
# ```
# with $I_0 = 0$, and setting
# ```math
#   X_{t_j} = X_0 e^{I_j}.
# ```
# 
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages

using Plots
using Random
using Distributions
using RODEConvergence

# Then we set up some variables, starting by choosing the `Xoshiro256++` pseudo-random number generator, and setting its seed for the sake of reproducibility:

rng = Xoshiro(123)
nothing # hide

# We set the right hand side of the equation:

f(t, x, y, p) = y * x
nothing # hide

# Next we set up the time interval and the initial distribution law for the initial value problem, which we take it to be a [Distributions.Normal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Normal) random variable:

t0, tf = 0.0, 1.0
x0law = Normal()

# The noise is a [`WienerProcess`](@ref) starting at ``y_0 = 0``:

y0 = 0.0
noise = WienerProcess(t0, tf, y0)

# There is no parameter in the equation, so we just set `params` to `nothing`.

params = nothing

# The number of mesh points for the target solution and the approximations

ntgt = 2^10
ns = 2 .^ (4:3:8)

# and for a visualization of one of the sample approximations

nsample = ns[[1, 2]]

# and add some information about the simulation, for the caption of the convergence figure.

info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = W_t X_t\$",
    noise = "a standard Wiener process noise \$\\{W_t\\}_t\$",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)
nothing # hide

# The *target* solution as described above is implemented as

target_solver! = function (xt::Vector{T}, t0::T, tf::T, x0::T, f::F, yt::Vector{T}, params::Q, rng::AbstractRNG) where {T, F, Q}
    axes(xt) == axes(yt) || throw(
        DimensionMismatch("The vectors `xt` and `yt` must match indices")
    )

    n = size(xt, 1)
    dt = (tf - t0) / (n - 1)
    i1 = firstindex(xt)
    xt[i1] = x0
    integral = zero(T)
    zscale = sqrt(dt^3 / 12)
    for i in Iterators.drop(eachindex(xt, yt), 1)
        integral += (yt[i] + yt[i1]) * dt / 2 + zscale * randn(rng)
        xt[i] = x0 * exp(integral)
        i1 = i
    end
end
nothing # hide

# and with that we construct the [`CustomMethod`](@ref) that solves the problem with this `target_solver!`:

target = CustomUnivariateMethod(target_solver!, rng)
nothing # hide

# The method for which we want to estimate the rate of convergence is, naturally, the Euler method, denoted [`RandomEuler`](@ref):

method = RandomEuler()

# ### Investigation of the statistics of the approximations

# We first write a function to grab the statistics

function getstatistics(rng, suite, ns, nk, m)
    ps = zeros(nk)
    allerrors = zeros(nk, length(ns))
    allstderrs = zeros(nk, length(ns))
    @time for k in 1:nk
        resultk = solve(rng, suite)
        ps[k] = resultk.p
        allerrors[k, :] .= resultk.errors
        allstderrs[k, :] .= resultk.stderrs
    end
    meanerror = mean(allerrors, dims=1)

    percent_e1_in = 100 * count(( meanerror[1] .> allerrors[:, 1] .- 1.96allstderrs[:, 1] ) .& ( meanerror[1] .< allerrors[:, 1] .+ 1.96allstderrs[:, 1] )) / nk

    percent_e2_in = 100 * count(( meanerror[2] .> allerrors[:, 2] .- 1.96allstderrs[:, 2] ) .& ( meanerror[2] .< allerrors[:, 2] .+ 1.96allstderrs[:, 2] )) / nk

    percent_e_in = 100 * count(
        ( meanerror[1] .> allerrors[:, 1] .- 1.96allstderrs[:, 1] ) .& ( meanerror[1] .< allerrors[:, 1] .+ 1.96allstderrs[:, 1] ) .&
        ( meanerror[2] .> allerrors[:, 2] .- 1.96allstderrs[:, 2] ) .& ( meanerror[2] .< allerrors[:, 2] .+ 1.96allstderrs[:, 2] ) 
        ) / nk

    percent_ehalf_in = 100 * count(
        ( meanerror[1] .> allerrors[1:div(nk,2), 1] .- 1.96allstderrs[1:div(nk,2), 1] ) .& ( meanerror[1] .< allerrors[1:div(nk,2), 1] .+ 1.96allstderrs[1:div(nk,2), 1] ) .&
        ( meanerror[2] .> allerrors[div(nk,2)+1:nk, 2] .- 1.96allstderrs[div(nk,2)+1:nk, 2] ) .& ( meanerror[2] .< allerrors[div(nk,2)+1:nk, 2] .+ 1.96allstderrs[div(nk,2)+1:nk, 2] ) 
        ) / nk * 2

    percent_ehalf_in2536 = 100 * count(
        ( meanerror[1] .> allerrors[1:div(nk,2), 1] .- 2.536allstderrs[1:div(nk,2), 1] ) .& ( meanerror[1] .< allerrors[1:div(nk,2), 1] .+ 2.536allstderrs[1:div(nk,2), 1] ) .&
        ( meanerror[2] .> allerrors[div(nk,2)+1:nk, 2] .- 2.536allstderrs[div(nk,2)+1:nk, 2] ) .& ( meanerror[2] .< allerrors[div(nk,2)+1:nk, 2] .+ 2.536allstderrs[div(nk,2)+1:nk, 2] ) 
        ) / nk * 2

    return ps, allerrors, allstderrs, meanerror, percent_p_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_ehalf_in2536
end

function printpercents(percent_p_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_ehalf_in2536)
    println("percent p in: $percent_p_in%")
    println("percent E1 in: $percent_e1_in%")
    println("percent E2 in: $percent_e2_in%")
    println("percent E in: $percent_e_in%")
    println("percent E in independent: $percent_ehalf_in%")
    println("percent E in independent larger: $percent_ehalf_in2536%")
end

function showplots(allerrors, result, m, nk, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_ehalf_in2536)
    rect = Shape(
        [
            (result.errors[1] - 2result.stderrs[1], result.errors[2] - 2result.stderrs[2]),
            (result.errors[1] - 2result.stderrs[1], result.errors[2] + 2result.stderrs[2]),
            (result.errors[1] + 2result.stderrs[1], result.errors[2] + 2result.stderrs[2]),
            (result.errors[1] + 2result.stderrs[1], result.errors[2] - 2result.stderrs[2])
        ]
    )

    plt = plot(title="Errors all (m=$m, nk=$nk, $percent_e_in%)", titlefont=10, xlabel="E1", ylabel="E2")
    begin
        scatter!(plt, allerrors[:, 1], allerrors[:, 2], label="errors")
        plot!(plt, rect, alpha = 0.2, label="CI")
    end
    display(plt)

    plt = plot(title="Errors split (m=$m, nk=$nk, $percent_ehalf_in%)", titlefont=10, xlabel="E1", ylabel="E2")
    begin
        scatter!(plt, allerrors[1:div(nk,2), 1], allerrors[div(nk,2)+1:nk, 2], label="errors")
        plot!(plt, rect, alpha = 0.2, label="CI")
    end
    display(plt)

    plt = plot(title="Histogram of E_1 (m=$m, nk=$nk, $percent_e1_in%)", titlefont=10, xlabel="E_1")
    begin
        histogram!(plt, allerrors[:, 1], label="error 1")
        vline!(plt, [mean(allerrors[:, 1])], color=:steelblue, linewidth=4, label="mean")
        vline!(plt, [result.errors[1]], label="sample")
        vline!(plt, [result.errors[1] - 2result.stderrs[1], result.errors[1] + 2result.stderrs[1]], label="CI from sample")
    end
    display(plt)

    plt = plot(title="Histogram of E_2 (m=$m, nk=$nk, $percent_e2_in%)", titlefont=10, label="error 2")
    begin
        histogram!(plt, allerrors[:, 2], label="error 2")
        vline!(plt, [mean(allerrors[:, 2])], color=:steelblue, linewidth=4, label="mean")
        vline!(plt, [result.errors[2]], label="sample")
        vline!(plt, [result.errors[2] - 2result.stderrs[2], result.errors[2] + 2result.stderrs[2]], label="CI from sample")
    end
    display(plt)
end

# We loop varying the number of samples in each run and the number of test runs.

ms = (10, 20, 100, 200, 500)
nks = (200, 200, 500, 500, 1000)

@assert all(iseven, nks)

for (nrun, m, nk) in zip(eachindex(ms), ms, nks)
    
    @info "==="
    @info "Run $nrun with m=$m and nk=$nk"
    suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)

    ps, allerrors, allstderrs, meanerror, percent_p_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_ehalf_in2536 = getstatistics(rng, suite, ns, nk, m)

    @show cor(allerrors) # strongly correlated!

    @show cor([allerrors[1:div(nk,2), 1] allerrors[div(nk,2)+1:nk, 2]]) # weakly correlated

    printpercents(percent_p_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_ehalf_in2536)

    @time result = solve(rng, suite)

    percent_p_in = 100 * count(( ps .> result.pmin ) .& ( ps .< result.pmax )) / nk
    println("Percent p in $percent_p_in")

    showplots(allerrors, result, m, nk, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_ehalf_in2536)
end
