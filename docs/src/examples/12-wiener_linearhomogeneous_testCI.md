```@meta
EditURL = "../../literate/examples/12-wiener_linearhomogeneous_testCI.jl"
```

# Testing the confidence regions and intervals 1

We consider a simple and quick-to-solve Random ODE to test the confidence regions and intervals. With a simple model, we can easily run a million simulations to test the statistics.

The Random ODE is a simple homogeneous linear equation in which the coefficient is a Wiener process and for which we know the distribution of the exact solution.

In this first example, we consider only two mesh resolutions, for full visualization.

## The equation

We consider the RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = W_t X_t, \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{W_t\}_{t\geq 0}$ is a standard Wiener process.
The explicit solution is
```math
  X_t = e^{\int_0^t W_s \;\mathrm{d}s} X_0.
```

As seen in the first example of this documentation, once an Euler approximation is computed, along with realizations $\{W_{t_i}\}_{i=0}^n$ of a sample path of the noise, we consider an exact sample solution given by
```math
    X_{t_j} = X_0 e^{\sum_{i = 0}^{j-1}\left(\frac{1}{2}\left(W_{t_i} + W_{t_{i+1}}\right)(t_{i+1} - t_i) + Z_i\right)},
```
for realizations $Z_i$ drawn from a normal distribution and scaled by the standard deviation $\sqrt{(t_{i+1} - t_i)^3/12}$. This is implemented by computing the integral recursively, via
```math
    \begin{cases}
        I_j = I_{j-1} + \frac{1}{2}\left(W_{t_{j-1}} + W_{t_j}\right)(t_{j} - t_{j-1}) + Z_j, \\
        Z_j = \sqrt{\frac{(t_{j} - t_{j-1})^3}{12}} R_j, \\
        R_j \sim \mathcal{N}(0, 1), \\
    \end{cases}
```
with $I_0 = 0$, and setting
```math
  X_{t_j} = X_0 e^{I_j}.
```

## Setting up the problem

First we load the necessary packages

````@example 12-wiener_linearhomogeneous_testCI
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables, starting by choosing the `Xoshiro256++` pseudo-random number generator, and setting its seed for the sake of reproducibility:

````@example 12-wiener_linearhomogeneous_testCI
rng = Xoshiro(123)
nothing # hide
````

We set the right hand side of the equation:

````@example 12-wiener_linearhomogeneous_testCI
f(t, x, y, p) = y * x
nothing # hide
````

Next we set up the time interval and the initial distribution law for the initial value problem, which we take it to be a standard [Distributions.Normal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Normal) random variable:

````@example 12-wiener_linearhomogeneous_testCI
t0, tf = 0.0, 1.0
x0law = Normal()
````

The noise is a [`WienerProcess`](@ref) starting at ``y_0 = 0``:

````@example 12-wiener_linearhomogeneous_testCI
y0 = 0.0
noise = WienerProcess(t0, tf, y0)
````

There is no parameter in the equation, so we just set `params` to `nothing`.

````@example 12-wiener_linearhomogeneous_testCI
params = nothing
````

The number of mesh points for the target solution and the approximations

````@example 12-wiener_linearhomogeneous_testCI
ntgt = 2^8
ns = 2 .^ (4:2:6)
````

Notice we just chose two mesh sizes, so we can easily visualize the distributions.

The *target* solution as described above is implemented as

````@example 12-wiener_linearhomogeneous_testCI
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
````

and with that we construct the [`CustomMethod`](@ref) that solves the problem with this `target_solver!`:

````@example 12-wiener_linearhomogeneous_testCI
target = CustomUnivariateMethod(target_solver!, rng)
nothing # hide
````

The method for which we want to estimate the rate of convergence is, naturally, the Euler method, denoted [`RandomEuler`](@ref):

````@example 12-wiener_linearhomogeneous_testCI
method = RandomEuler()
````

## Defining helper functions

We first write some helper functions to grab the statistics, print some information, and build some plots.

````@example 12-wiener_linearhomogeneous_testCI
function getstatistics(rng, suite, ns, nk, m)
    ps = zeros(nk)
    pmins = zeros(nk)
    pmaxs = zeros(nk)
    allerrors = zeros(nk, length(ns))
    allstderrs = zeros(nk, length(ns))
    @time for k in 1:nk
        resultk = solve(rng, suite)
        ps[k] = resultk.p
        allerrors[k, :] .= resultk.errors
        allstderrs[k, :] .= resultk.stderrs
        pmins[k] = resultk.pmin
        pmaxs[k] = resultk.pmax
    end
    meanerror = mean(allerrors, dims=1)
    pmean = mean(ps)

    percent_e1_in = 100 * count(( meanerror[1] .> allerrors[:, 1] .- 1.96allstderrs[:, 1] ) .& ( meanerror[1] .< allerrors[:, 1] .+ 1.96allstderrs[:, 1] )) / nk

    percent_e2_in = 100 * count(( meanerror[2] .> allerrors[:, 2] .- 1.96allstderrs[:, 2] ) .& ( meanerror[2] .< allerrors[:, 2] .+ 1.96allstderrs[:, 2] )) / nk

    percent_e_in = 100 * count(
        ( meanerror[1] .> allerrors[:, 1] .- 2.24allstderrs[:, 1] ) .& ( meanerror[1] .< allerrors[:, 1] .+ 2.24allstderrs[:, 1] ) .&
        ( meanerror[2] .> allerrors[:, 2] .- 2.24allstderrs[:, 2] ) .& ( meanerror[2] .< allerrors[:, 2] .+ 2.24allstderrs[:, 2] )
        ) / nk

    percent_ehalf_in = 100 * count(
        ( meanerror[1] .> allerrors[1:div(nk,2), 1] .- 2.24allstderrs[1:div(nk,2), 1] ) .& ( meanerror[1] .< allerrors[1:div(nk,2), 1] .+ 2.24allstderrs[1:div(nk,2), 1] ) .&
        ( meanerror[2] .> allerrors[div(nk,2)+1:nk, 2] .- 2.24allstderrs[div(nk,2)+1:nk, 2] ) .& ( meanerror[2] .< allerrors[div(nk,2)+1:nk, 2] .+ 2.24allstderrs[div(nk,2)+1:nk, 2] )
        ) / nk * 2

    percent_edealigned_in = 100 * count(
        ( meanerror[1] .> allerrors[begin:end-1, 1] .- 2.24allstderrs[begin:end-1, 1] ) .& ( meanerror[1] .< allerrors[begin:end-1, 1] .+ 2.24allstderrs[begin:end-1, 1] ) .&
        ( meanerror[2] .> allerrors[begin+1:end, 2] .- 2.24allstderrs[begin+1:end, 2] ) .& ( meanerror[2] .< allerrors[begin+1:end, 2] .+ 2.24allstderrs[begin+1:end, 2] )
        ) / ( nk - 1 )

    deltas = (suite.tf - suite.t0) ./ suite.ns
    A = [one.(deltas) log.(deltas)]
    L = inv(A' * A) * A'

    Llnerrors = L * log.(allerrors')

    Llnerrorsdealigned = L * log.([allerrors[:, 1] circshift(allerrors[:, 2], -1)]')

    percent_p_dealigned_in = 100 * count( ( pmean .> Llnerrorsdealigned[2, :] .- (ps .- pmins) ) .& ( pmean .< Llnerrorsdealigned[2, :] .+ (pmaxs .- ps) ) ) / nk

    percent_p_in = 100 * count(( pmean .> pmins ) .& ( pmean .< pmaxs )) / nk

    pstd = std(ps)
    percent_p_alt_in = 100 * count(( pmean .> ps .- 1.96pstd ) .& ( pmean .< ps .+ 1.96pstd )) / nk

    pdlgnstd = std(Llnerrorsdealigned[2, :])
    percent_p_alt_dealigned_in = 100 * count(( pmean .> Llnerrorsdealigned[2, :] .- 1.96pdlgnstd ) .& ( pmean .< Llnerrorsdealigned[2, :] .+ 1.96pdlgnstd )) / nk

    return ps, allerrors, allstderrs, meanerror, pmean, Llnerrors, Llnerrorsdealigned, percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_edealigned_in, L
end

function printpercents(
    percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_edealigned_in
)
    println("percent p in: $percent_p_in%")
    println("percent p dealigned in: $percent_p_dealigned_in%")
    println("percent p alt dealigned in: $percent_p_alt_dealigned_in%")
    println("percent E1 in: $percent_e1_in%")
    println("percent E2 in: $percent_e2_in%")
    println("percent E in: $percent_e_in%")
    println("percent E in half-half: $percent_ehalf_in%")
    println("percent E in dealigned larger: $percent_edealigned_in%")
end

function showplots(
    ps, allerrors, Llnerrors, Llnerrorsdealigned, pmean, result, m, nk, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_p_dealigned_in, percent_edealigned_in, L
)
    rect = Shape(
        [
            (result.errors[1] - 2.24result.stderrs[1], result.errors[2] - 2.24result.stderrs[2]),
            (result.errors[1] - 2.24result.stderrs[1], result.errors[2] + 2.24result.stderrs[2]),
            (result.errors[1] + 2.24result.stderrs[1], result.errors[2] + 2.24result.stderrs[2]),
            (result.errors[1] + 2.24result.stderrs[1], result.errors[2] - 2.24result.stderrs[2])
        ]
    )

    plt_errors = plot(title="Errors all (m=$m, nk=$nk)", titlefont=10, xlabel="ϵ₁", ylabel="ϵ₂")

    scatter!(plt_errors, allerrors[:, 1], allerrors[:, 2], alpha=0.2, label="errors ($(round(percent_e_in, digits=2))% in CI)")
    scatter!(plt_errors, allerrors[begin:end-1, 1], allerrors[begin+1:end, 2], alpha=0.2, label="errors dealigned ($(round(percent_edealigned_in, digits=2))% in CI)")
    scatter!(plt_errors, Tuple(mean(allerrors, dims=1)), markersize=4, label="error mean")
    plot!(plt_errors, rect, alpha=0.2, label="CI")

    plt_errors_split = plot(title="Errors split (m=$m, nk=$nk) \n ($(round(percent_ehalf_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₁", ylabel="ϵ₂")
    begin
        scatter!(plt_errors_split, allerrors[1:div(nk,2), 1], allerrors[div(nk,2)+1:nk, 2], alpha=0.2, label="errors")
        scatter!(plt_errors_split, Tuple(mean(allerrors, dims=1)), markersize=4, label="error mean")
        plot!(plt_errors_split, rect, alpha=0.2, label="CI")
    end

    plt_errors_dealigned = plot(title="Errors dealigned (m=$m, nk=$nk) \n ($(round(percent_edealigned_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₁", ylabel="ϵ₂")
    begin
        scatter!(plt_errors_dealigned, allerrors[begin:end-1, 1], allerrors[begin+1:end, 2], alpha=0.2, label="errors")
        scatter!(plt_errors_dealigned, Tuple(mean(allerrors, dims=1)), markersize=4, label="error mean")
        plot!(plt_errors_dealigned, rect, alpha=0.2, label="CI")
    end

    plt_hist_e1 = plot(title="Histogram of ϵ₁ (m=$m, nk=$nk) \n ($(round(percent_e1_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₁")
    begin
        histogram!(plt_hist_e1, allerrors[:, 1], label="error ϵ₁")
        vline!(plt_hist_e1, [mean(allerrors[:, 1])], color=:steelblue, linewidth=4, label="mean")
        vline!(plt_hist_e1, [result.errors[1]], label="sample")
        vline!(plt_hist_e1, [result.errors[1] - 2result.stderrs[1], result.errors[1] + 2result.stderrs[1]], label="CI from sample")
    end

    plt_hist_e2 = plot(title="Histogram of ϵ₂ (m=$m, nk=$nk) \n ($(round(percent_e2_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₂")
    begin
        histogram!(plt_hist_e2, allerrors[:, 2], label="error ϵ₂")
        vline!(plt_hist_e2, [mean(allerrors[:, 2])], color=:steelblue, linewidth=4, label="mean")
        vline!(plt_hist_e2, [result.errors[2]], label="sample")
        vline!(plt_hist_e2, [result.errors[2] - 2result.stderrs[2], result.errors[2] + 2result.stderrs[2]], label="CI from sample")
    end

    sn = 50
    s1 = L * log.(max.(0.0, [result.errors[1] .+ 2.24result.stderrs[1] * range(-1, 1, length=sn) ( result.errors[2] - 2.24result.stderrs[2] ) .* ones(sn)]'))
    s2 = L * log.(max.(0.0, [( result.errors[1] + 2.24result.stderrs[1] ) .* ones(sn) result.errors[2] .+ 2.24result.stderrs[2] * range(-1, 1, length=sn)]'))
    s3 = L * log.(max.(0.0, [result.errors[1] .+ 2.24result.stderrs[1] * reverse(range(-1, 1, length=sn)) ( result.errors[2] + 2.24result.stderrs[2] ) .* ones(sn)]'))
    s4 = L * log.(max.(0.0, [( result.errors[1] - 2.24result.stderrs[1] ) .* ones(sn) result.errors[2] .+ 2.24result.stderrs[2] * range(-1, 1, length=sn)]'))
    sides = hcat(s1, s2, s3, s4)

    temean = L * log.(mean(allerrors, dims=1)')

    plt_Cp = plot(title="(C, p) sample from (ϵ₁, ϵ₂) (m=$m, nk=$nk)", titlefont=10, xlabel="C", ylabel="p")
    begin
        scatter!(plt_Cp, Llnerrors[1, :], Llnerrors[2, :], alpha=0.2, label="correlated")
        scatter!(plt_Cp, Llnerrorsdealigned[1, :], Llnerrorsdealigned[2, :], alpha=0.2, label="dealigned")
        plot!(plt_Cp, sides[1, :], sides[2, :], label="transformed errors CI")
        scatter!(plt_Cp, Tuple(temean), markersize=4, color=:orange, label="transformed error mean")
        hline!(plt_Cp, [pmean], label="p mean")
        hline!(plt_Cp, [result.pmin, result.pmax], label="sample p CI ($(round(percent_p_dealigned_in, digits=2))% in CI)")
        hline!(plt_Cp, [result.p], label="sample p")
    end

    plt_hist_p = plot(title="Histogram of p (m=$m, nk=$nk) \n ($(round(percent_p_dealigned_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₁")
    begin
        histogram!(plt_hist_p, Llnerrorsdealigned[2, :], label="p dealigned")
        histogram!(plt_hist_p, ps, label="p")
        vline!(plt_hist_p, [pmean], linewidth=4, label="p mean")
        vline!(plt_hist_p, [result.pmin, result.pmax], label="sample p CI ($(round(percent_p_dealigned_in, digits=2))% in CI)")
        vline!(plt_hist_p, [result.p], label="sample p")
    end

    plts = (
        errors = plt_errors,
        split = plt_errors_split,
        dealigned = plt_errors_dealigned,
        hist1 = plt_hist_e1,
        hist2 = plt_hist_e2,
        cp = plt_Cp,
        histp = plt_hist_p
    )

    return plts
end
````

## Statistics

Now, with the helper functions, we run a loop varying the number $m$ of samples in each run and the number $nk$ of test runs, showing some relevant statistics.

````@example 12-wiener_linearhomogeneous_testCI
ms = (200, 500, 1000, 2000)
nks = (2000, 2000, 2000, 2000)

@assert all(iseven, nks)

allplts = Any[]

for (nrun, m, nk) in zip(eachindex(ms), ms, nks)

    @info "==="
    @info "Run $nrun with m=$m and nk=$nk"
    suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)

    ps, allerrors, allstderrs, meanerror, pmean, Llnerrors, Llnerrorsdealigned, percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_edealigned_in, L = getstatistics(rng, suite, ns, nk, m)

    @show cor(allerrors) # strongly correlated!

    @show cor([allerrors[:, 1] circshift(allerrors[:, 2], -1)]) # weakly correlated

    @show cor(Llnerrors') # somehow correlated

    @show cor(Llnerrorsdealigned') # weakly correlated

    printpercents(percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_edealigned_in)

    result = solve(rng, suite)

    plts = showplots(ps, allerrors, Llnerrors, Llnerrorsdealigned, pmean, result, m, nk, percent_e1_in, percent_e2_in, percent_e_in, percent_ehalf_in, percent_p_dealigned_in, percent_edealigned_in, L)

    append!(allplts, [plts])
end
````

## Visualizations

We now visualize some statistics, including histograms and sample distribution. In those plots, we report the percentage of confidence intervals and regions that include the mean.

### Histograms of the marginal strong errors

We start with the histograms of each of the strong errors at each mesh resolution, with $\epsilon_1$ corresponding to $\Delta t = 2^4$ and with $\epsilon_2$ corresponding to $\Delta t = 2^6.$ These are the marginals of the joint distribution $(\epsilon_1, \epsilon_2).$

Notice that in the first two plots, with lower samples, the distribution of the sample means is not quite normal and only a bit more than 90% of the corresponding 95% CIs contain the mean, or rather a better approximation of the mean with orders of magnitude more samples. The last two plots, with more samples, the histogram resembles more a Gaussian distribution and the CI is close to the expected 95% level.

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[1].hist1, allplts[1].hist2)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[2].hist1, allplts[2].hist2)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[3].hist1, allplts[3].hist2)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[4].hist1, allplts[4].hist2)
````

### Density of the joint distribution of strong errors and their transformed distributions

In the following, we plot, on the left panel, the sample points of the joint distribution $(\epsilon_1, \epsilon_2).$ Because the way they were computed (using pathwise samples for the noise process in the finer mesh bridged from the one in the coarset mesh), this joint distribution is highly correlated (about 0.9 correlation, as calculated above). We can see this from the plots. We also plot a decorrelated sample obtained from shifting the indices of $\epsilon_2.$ Another option is to take the first half of $\epsilon_1$ and the second half of $\epsilon_2,$ to make them completely independent, but the result is about the same. We also illustrate one confidence region, from a single random sample, which is supposed to include the mean of the joint distribution, obtained by averaging the strong errors, which themselves are averagings of pathwise erros.

On the right panel, we see the corresponding transformed samples $(C, p) = (A^{\textrm{tr}}A)^{-1}A^{\textrm{tr}}(\epsilon_1, \epsilon_2)$ and transformed confidence region.

The confidence interval for the order of convergence $p$ is the projection, onto the $p$ axis, of the confidence region in the $(C, p)$ plane. It includes not only the samples within the confidence region but all of those in the band $p_{\min} \leq p \leq p_{\max},$ increasing considerably the confidence level.

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[1].errors, allplts[1].cp)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[2].errors, allplts[2].cp)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[3].errors, allplts[3].cp)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(size=(800, 400), allplts[4].errors, allplts[4].cp)
````

### Histrogram of the order of convergence

Finally, we plot the histograms for $p$, obtained both from the correlated and the decorrelated strong errors. Notice that the distributions for $p$ resembles a normal distribution even for low samples, and building a CI from the decorrelated samples works fine, in this example, despite the fact that the theory does not guarantee that. But it works only with the uncorrelad samples! Nevertheless, it requires a lot more samples, being computationally quite expensive, especially with more complicate equations. The CI from the push-forward method underestimates the confidence level, but it is more trustworthy and less demanding.

````@example 12-wiener_linearhomogeneous_testCI
plot(allplts[1].histp)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(allplts[2].histp)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(allplts[3].histp)
````

````@example 12-wiener_linearhomogeneous_testCI
plot(allplts[4].histp)
````

````@example 12-wiener_linearhomogeneous_testCI
nothing # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

