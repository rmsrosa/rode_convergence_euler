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
    repo = "https://github.com/rmsrosa/rode_conv_em",
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
            "examples/09-fisherkpp.md",
            "examples/10-risk.md"
        ],
        "Noises" => [
            "noises/noiseintro.md",
            "noises/homlin.md",
            "noises/fBm.md"
        ],
        "api.md",
    ],
    authors = "Peter Kloeden and Ricardo Rosa",
    draft = "draft" in ARGS,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/rode_conv_em",
        edit_link = "main",
    ),
    modules = [RODEConvergence],
)

if get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
    deploydocs(
        repo = "github.com/rmsrosa/rode_conv_em.git",
        devbranch = "main",
        forcepush = true,
    )
end
