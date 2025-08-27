```@meta
EditURL = "../../literate/examples/13-wiener_linearhomogeneous_testCI_multidim.jl"
```

# Testing the confidence regions and intervals 2

```@meta
Draft = false
```

We consider a simple and quick-to-solve Random ODE to test the confidence regions and intervals. With a simple model, we can easily run a million simulations to test the statistics.

The Random ODE is a simple homogeneous linear equation in which the coefficient is a Wiener process and for which we know the distribution of the exact solution.

Now we consider an arbitrary number of mesh resolutions.

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

````@example 13-wiener_linearhomogeneous_testCI_multidim
using Plots
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables, starting by choosing the `Xoshiro256++` pseudo-random number generator, and setting its seed for the sake of reproducibility:

````@example 13-wiener_linearhomogeneous_testCI_multidim
rng = Xoshiro(123)
nothing # hide
````

We set the right hand side of the equation:

````@example 13-wiener_linearhomogeneous_testCI_multidim
f(t, x, y, p) = y * x
nothing # hide
````

Next we set up the time interval and the initial distribution law for the initial value problem, which we take it to be a standard [Distributions.Normal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Normal) random variable:

````@example 13-wiener_linearhomogeneous_testCI_multidim
t0, tf = 0.0, 1.0
x0law = Normal()
````

The noise is a [`WienerProcess`](@ref) starting at ``y_0 = 0``:

````@example 13-wiener_linearhomogeneous_testCI_multidim
y0 = 0.0
noise = WienerProcess(t0, tf, y0)
````

There is no parameter in the equation, so we just set `params` to `nothing`.

````@example 13-wiener_linearhomogeneous_testCI_multidim
params = nothing
````

The number of mesh points for the target solution and the approximations

````@example 13-wiener_linearhomogeneous_testCI_multidim
ntgt = 2^12
ns = 2 .^ (4:2:10)
````

Notice we just chose two mesh sizes, so we can easily visualize the distributions.

The *target* solution as described above is implemented as

````@example 13-wiener_linearhomogeneous_testCI_multidim
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

````@example 13-wiener_linearhomogeneous_testCI_multidim
target = CustomUnivariateMethod(target_solver!, rng)
nothing # hide
````

The method for which we want to estimate the rate of convergence is, naturally, the Euler method, denoted [`RandomEuler`](@ref):

````@example 13-wiener_linearhomogeneous_testCI_multidim
method = RandomEuler()
````

## Defining helper functions

We first write some helper functions to grab the statistics, print some information, and build some plots.

````@example 13-wiener_linearhomogeneous_testCI_multidim
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

    percent_ei_in = 100 * [ count(( meanerror[i] .> allerrors[:, i] .- 1.96allstderrs[:, i] ) .& ( meanerror[i] .< allerrors[:, i] .+ 1.96allstderrs[:, i] )) for i in eachindex(axes(meanerror, 2), axes(allstderrs, 2)) ] / nk

    clevel = 0.95 # 95% confidence level
    elevel = clevel^(1/length(suite.ns)) # assuming independence
    elevel = 1.0 - ( 1.0 - clevel ) / length(suite.ns) # not assuming independence, using Bonfaroni inequality
    escore = quantile(Normal(), (1 + elevel) / 2)

    percent_e_in = 100 * count(
        reduce(
            &,
            ae - escore * ast .< meanerror' .< ae + escore * ast
        ) for (ae, ast) in zip(eachrow(allerrors), eachrow(allstderrs))
    ) / nk

    nkji = div(size(allerrors,1), size(allerrors,2))
    allerrorssplit = hcat((allerrors[nkji*(i-1)+1:nkji*i, i] for i in axes(allerrors, 2))...)
    allstderrssplit = hcat((allstderrs[nkji*(i-1)+1:nkji*i, i] for i in axes(allstderrs, 2))...)

    percent_e_split_in = 100 * count(
        reduce(
            &,
            ae - escore * ast .< meanerror' .< ae + escore * ast
        ) for (ae, ast) in zip(eachrow(allerrorssplit), eachrow(allstderrssplit))
    ) / nkji

    allerrorsdealigned = hcat([circshift(allerrors[:, i], i-1) for i in 1:4]...)
    allstderrsdealigned = hcat([circshift(allstderrs[:, i], i-1) for i in 1:4]...)

    percent_e_dealigned_in = 100 * count(
        reduce(
            &,
            ae - escore * ast .< meanerror' .< ae + escore * ast
        ) for (ae, ast) in zip(eachrow(allerrorsdealigned), eachrow(allstderrsdealigned))
    ) / nk

    deltas = (suite.tf - suite.t0) ./ suite.ns
    A = [one.(deltas) log.(deltas)]
    L = inv(A' * A) * A'

    Llnerrors = L * log.(allerrors')

    Llnerrorsdealigned = L * log.(allerrorsdealigned')

    percent_p_dealigned_in = 100 * count( ( pmean .> Llnerrorsdealigned[2, :] .- (ps .- pmins) ) .& ( pmean .< Llnerrorsdealigned[2, :] .+ (pmaxs .- ps) ) ) / nk

    percent_p_in = 100 * count(( pmean .> pmins ) .& ( pmean .< pmaxs )) / nk

    pstd = std(ps)
    percent_p_alt_in = 100 * count(( pmean .> ps .- 1.96pstd ) .& ( pmean .< ps .+ 1.96pstd )) / nk

    pdlgnstd = std(Llnerrorsdealigned[2, :])
    percent_p_alt_dealigned_in = 100 * count(( pmean .> Llnerrorsdealigned[2, :] .- 1.96pdlgnstd ) .& ( pmean .< Llnerrorsdealigned[2, :] .+ 1.96pdlgnstd )) / nk

    return ps, escore, allerrors, allstderrs, allerrorssplit, allerrorsdealigned, meanerror, pmean, Llnerrors, Llnerrorsdealigned, percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_ei_in, percent_e_in, percent_e_split_in, percent_e_dealigned_in, L
end

function printpercents(
    percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_ei_in, percent_e_in, percent_e_split_in, percent_e_dealigned_in
)
    println("percent p in: $percent_p_in%")
    println("percent p dealigned in: $percent_p_dealigned_in%")
    println("percent p alt dealigned in: $percent_p_alt_dealigned_in%")
    for i in eachindex(percent_ei_in)
        println("percent E$i in: $(percent_ei_in[i])%")
    end
    println("percent E in: $percent_e_in%")
    println("percent E in split: $percent_e_split_in%")
    println("percent E in dealigned: $percent_e_dealigned_in%")
end

function showplots(
    ps, allerrors, Llnerrors, Llnerrorsdealigned, pmean, result, m, nk, percent_ei_in, percent_e_in, percent_e_split_in, percent_p_dealigned_in, percent_e_dealigned_in, L
)

    clevel = 0.95 # 95% confidence level
    elevel = clevel^(1/length(suite.ns)) # assuming independence
    elevel = 1.0 - ( 1.0 - clevel ) / length(suite.ns) # not assuming independence, using Bonfaroni inequality
    escore = quantile(Normal(), (1 + elevel) / 2)
    rect = Shape(
        [
            (result.errors[1] - escore * result.stderrs[1], result.errors[2] - escore * result.stderrs[2]),
            (result.errors[1] - escore * result.stderrs[1], result.errors[2] + escore * result.stderrs[2]),
            (result.errors[1] + escore * result.stderrs[1], result.errors[2] + escore * result.stderrs[2]),
            (result.errors[1] + escore * result.stderrs[1], result.errors[2] - escore * result.stderrs[2])
        ]
    )

    plt_errors = plot(title="Errors all (m=$m, nk=$nk)", titlefont=10, xlabel="ϵ₁", ylabel="ϵ₂")
    begin
        scatter!(plt_errors, allerrors[:, 1], allerrors[:, 2], alpha=0.2, label="errors ($(round(percent_e_in, digits=2))% in CI)")
        scatter!(plt_errors, allerrors[begin:end-1, 1], allerrors[begin+1:end, 2], alpha=0.2, label="errors dealigned ($(round(percent_e_dealigned_in, digits=2))% in CI)")
        scatter!(plt_errors, Tuple(mean(view(allerrors, :, 1:2), dims=1)), markersize=4, label="error mean")
        plot!(plt_errors, rect, alpha=0.2, label="CI")
    end

    plt_errors_split = plot(title="Errors split (m=$m, nk=$nk) \n ($(round(percent_e_split_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₁", ylabel="ϵ₂")
    begin
        scatter!(plt_errors_split, allerrors[1:div(nk,2), 1], allerrors[div(nk,2)+1:nk, 2], alpha=0.2, label="errors")
        scatter!(plt_errors_split, Tuple(mean(view(allerrors, :, 1:2), dims=1)), markersize=4, label="error mean")
        plot!(plt_errors_split, rect, alpha=0.2, label="CI")
    end

    plt_errors_dealigned = plot(title="Errors dealigned (m=$m, nk=$nk) \n ($(round(percent_e_dealigned_in, digits=2))% in CI)", titlefont=10, xlabel="ϵ₁", ylabel="ϵ₂")
    begin
        scatter!(plt_errors_dealigned, allerrors[begin:end-1, 1], allerrors[begin+1:end, 2], alpha=0.2, label="errors")
        scatter!(plt_errors_dealigned, Tuple(mean(view(allerrors, :, 1:2), dims=1)), markersize=4, label="error mean")
        plot!(plt_errors_dealigned, rect, alpha=0.2, label="CI")
    end

    plt_hist_e = [
        begin
            plot(title="Histogram of ϵ$i (m=$m, nk=$nk) \n ($(round(percent_ei_in[i], digits=2))% in CI)", titlefont=10, xlabel="ϵ$i")
            histogram!(allerrors[:, i], label="error ϵ$i")
            vline!([mean(allerrors[:, i])], color=:steelblue, linewidth=4, label="mean")
            vline!([result.errors[i]], label="sample")
            vline!([result.errors[i] - 2result.stderrs[i], result.errors[i] + 2result.stderrs[i]], label="CI from sample")
        end
        for i in eachindex(axes(allerrors, 2))
    ]

    #sn = 50
    #s1 = L * log.(max.(0.0, [result.errors[1] .+ escore * result.stderrs[1] * range(-1, 1, length=sn) ( result.errors[2] - escore * result.stderrs[2] ) .* ones(sn)]'))
    #s2 = L * log.(max.(0.0, [( result.errors[1] + escore * result.stderrs[1] ) .* ones(sn) result.errors[2] .+ escore * result.stderrs[2] * range(-1, 1, length=sn)]'))
    #s3 = L * log.(max.(0.0, [result.errors[1] .+ escore * result.stderrs[1] * reverse(range(-1, 1, length=sn)) ( result.errors[2] + escore * result.stderrs[2] ) .* ones(sn)]'))
    #s4 = L * log.(max.(0.0, [( result.errors[1] - escore * result.stderrs[1] ) .* ones(sn) result.errors[2] .+ escore * result.stderrs[2] * range(-1, 1, length=sn)]'))
    #sides = hcat(s1, s2, s3, s4)

    temean = L * log.(mean(allerrors, dims=1)')

    plt_Cp = plot(title="(C, p) sample from ϵ (m=$m, nk=$nk)", titlefont=10, xlabel="C", ylabel="p")
    begin
        scatter!(plt_Cp, Llnerrors[1, :], Llnerrors[2, :], alpha=0.2, label="correlated")
        scatter!(plt_Cp, Llnerrorsdealigned[1, :], Llnerrorsdealigned[2, :], alpha=0.2, label="dealigned")
        #plot!(plt_Cp, sides[1, :], sides[2, :], label="transformed errors CI")
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
        hist = plt_hist_e,
        cp = plt_Cp,
        histp = plt_hist_p
    )

    return plts
end
````

## Statistics

Now, with the helper functions, we run a loop varying the number $m$ of samples in each run and the number $nk$ of test runs, showing some relevant statistics.

````@example 13-wiener_linearhomogeneous_testCI_multidim
ms = (500, 1000, 2000)
nks = (1000, 1000, 2000)

@assert all(iseven, nks)

allplts = Any[]

for (nrun, m, nk) in zip(eachindex(ms), ms, nks)

    @info "==="
    @info "Run $nrun with m=$m and nk=$nk"
    suite = ConvergenceSuite(t0, tf, x0law, f, noise, params, target, method, ntgt, ns, m)

    ps, escore, allerrors, allstderrs, allerrorssplit, allerrorsdealigned, meanerror, pmean, Llnerrors, Llnerrorsdealigned, percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_ei_in, percent_e_in, percent_e_split_in, percent_e_dealigned_in, L = getstatistics(rng, suite, ns, nk, m)

    @show cor(allerrors) # strongly correlated!

    @show cor(allerrorssplit) # weakly correlated

    @show cor(allerrorsdealigned) # weakly correlated

    @show cor(Llnerrors') # somehow correlated

    @show cor(Llnerrorsdealigned') # weakly correlated

    printpercents(percent_p_in, percent_p_dealigned_in, percent_p_alt_dealigned_in, percent_ei_in, percent_e_in, percent_e_split_in, percent_e_dealigned_in)

    result = solve(rng, suite)

    plts = showplots(ps, allerrors, Llnerrors, Llnerrorsdealigned, pmean, result, m, nk, percent_ei_in, percent_e_in, percent_e_split_in, percent_p_dealigned_in, percent_e_dealigned_in, L)

    append!(allplts, [plts])
end
````

## Visualizations

We now visualize some statistics, including histograms and sample distribution. In those plots, we report the percentage of confidence intervals and regions that include the mean.

### Histograms of the marginal strong errors

We start with the histograms of each of the strong errors at each mesh resolution. These are the marginals of the joint distribution $(\epsilon_1, \ldots, \epsilon_{i_{\max}).$

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400*div(length(ns), 2)), allplts[1].hist...)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400*div(length(ns), 2)), allplts[2].hist...)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400*div(length(ns), 2)), allplts[3].hist...)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400*div(length(ns), 2)), allplts[4].hist...)
````

### Density of the joint distribution of strong errors of the transformed distributions

In the following, we plot, on the left panel, the sample points of the joint distribution $(\epsilon_1, \epsilon_2)$ of the strong errors of the first two mesh resolutions.

On the right panel, we see the corresponding transformed samples $(C, p) = (A^{\textrm{tr}}A)^{-1}A^{\textrm{tr}}(\epsilon_1, \ldots, \epsilon_{i_{\max}})$.

The confidence interval for the order of convergence $p$ is the projection, onto the $p$ axis, of the confidence region in the $(C, p)$ plane. It includes not only the samples within the confidence region but all of those in the band $p_{\min} \leq p \leq p_{\max},$ increasing considerably the confidence level.

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400), allplts[1].errors, allplts[1].cp)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400), allplts[2].errors, allplts[2].cp)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(size=(800, 400), allplts[3].errors, allplts[3].cp)
````

### Histrogram of the order of convergence

Finally, we plot the histograms for $p$, obtained both from the correlated and the decorrelated strong errors. Notice that the distributions for $p$ resembles a normal distribution even for low samples, and building a CI from the decorrelated samples works fine, in this example, despite the fact that the theory does not guarantee that. But it works only with the uncorrelad samples! Nevertheless, it requires a lot more samples, being computationally quite expensive, especially with more complicate equations. The CI from the push-forward method underestimates the confidence level, but it is more trustworthy and less demanding.

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(allplts[1].histp)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(allplts[2].histp)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
plot(allplts[3].histp)
````

````@example 13-wiener_linearhomogeneous_testCI_multidim
nothing # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

