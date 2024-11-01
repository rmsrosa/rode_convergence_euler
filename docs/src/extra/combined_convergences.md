# Combined plot

```@meta
    Draft = false
```

Here we just read the results of some examples to build a combined plot for the companion article.

```@example combined
using JLD2
using Plots
using Measures
using Distributions
using RODEConvergence
```

```@example combined
println(@__DIR__())
```

```@example combined
println(pwd())
```

```@example combined
println(joinpath(@__DIR__(), "/../../literate/examples/results/06-popdyn_result.jld2"))
```


```@example combined
#results_popdyn = load(joinpath(@__DIR__() * "/../../literate/examples/results/06-popdyn_result.jld2"), "result")
#results_risk = load(joinpath(@__DIR__() * "/../../literate/examples/results/10-risk_result.jld2"), "result")
nothing
```

```@example combined
#plt_popdyn = plot(results_popdyn)
#plt_risk = plot(results_risk)

#plt_combined = plot(plt_popdyn, plt_risk, legendfont=6, size=(800, 240), title=["(a)" "(b)"], titlefont=10, bottom_margin=5mm, left_margin=5mm)
nothing
```

```@example combined
#savefig(plt_combined, joinpath(@__DIR__() * "../../../../latex/img/", "combined_popdyn_risk.pdf"))
nothing # hide
```