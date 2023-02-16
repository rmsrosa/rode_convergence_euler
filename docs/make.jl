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

makedocs(
    sitename = "Euler method for RODEs",
    repo = "https://github.com/rmsrosa/rode_conv_em",
    pages = [
        "Overview" => "index.md",
        "Theory" => [
            "theory/results.md",
            "theory/idea.md",
        ],
        "Examples" => generated_examples,
        #= "DifferentialEquations.jl" => [
            "Nonhomogenous Wiener noise" => "sciml/wiener_nonhomogeneous.md",
            "Homogenous Wiener noise" => "sciml/wiener_homogeneous.md",
        ], =#
        "Noises" => [
            "noises/fBm.md",
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

#= deploydocs(
    repo      = "https://github.com/rmsrosa/rode_conv_em",
    devbranch = "main",
) =#