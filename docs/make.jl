#!/usr/bin/env julia

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using RODEConvergence

if "liveserver" in ARGS
    using Revise
    Revise.revise()
end

ENV["GKSwstype"] = "100"

# Generate markdown pages from Literate scripts and get the list of generated pages as `generate_examples`:
include(joinpath(@__DIR__(), "literate.jl"))

@time makedocs(
    sitename = "Euler method for RODEs",
    repo = "https://github.com/rmsrosa/rode_convergence_euler",
    pages = [
        "Overview" => "index.md",
        "Theory" => [
            "theory/results.md",
            "theory/idea.md",
            "theory/extras.md"
        ],
        "Examples" => [
            "Basic Linear RODEs" => [
                "examples/01-wiener_linearhomogeneous.md",
                "examples/02-wiener_linearnonhomogeneous.md",
                "examples/03-sin_gBm_linearhomogeneous.md",
            ],
            "examples/04-allnoises.md",
            "examples/05-fBm_linear.md",
            "examples/06-popdyn.md",
            "examples/07-toggle_switch.md",
            "examples/08-earthquake.md",
            "examples/09-risk.md",
            "examples/10-fisherkpp.md",
            "examples/11-combined_convergences.md"
        ],
        "Noises" => [
            "noises/noiseintro.md",
            "noises/homlin.md",
            "noises/fBm.md"
        ],
        "CI testing" => [
            "examples/12-wiener_linearhomogeneous_testCI.md",
            "examples/13-wiener_linearhomogeneous_testCI_multidim.md",
            "examples/14-wiener_linearhomogeneous_exploreCI.md"
        ],
        "api.md",
    ],
    authors = "Peter Kloeden and Ricardo Rosa",
    draft = "draft" in ARGS,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/rode_convergence_euler",
        edit_link = "main",
        repolink = "https://github.com/rmsrosa/rode_convergence_euler",
        example_size_threshold = 10^4, # default is 8Kb, increase if necessary
        size_threshold_warn = 10^4, # default is 100Kb; increase if necessary
        size_threshold = 10^6, # default is 200Kb; increase if necessary
    ),
    modules = [RODEConvergence],
)

if get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
    deploydocs(
        repo = "github.com/rmsrosa/rode_convergence_euler.git",
        devbranch = "main",
        forcepush = true,
    )
end
