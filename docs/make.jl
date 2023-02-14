using Documenter
using RODEConvergence

ENV["GKSwstype"] = "100"

makedocs(
    sitename = "Euler method for RODEs",
    pages = [
        "Overview" => "index.md",
        "Theory" => [
            "Main results" => "theory/results.md",
            "Main idea" => "theory/idea.md",
        ],
        "Examples" => [
            "Homogenous linear equation with Wiener noise coefficient" => "examples/wiener_homogeneous.md",
            "Homogenous linear equation with sine of Wiener noise coefficient" => "examples/sin_wiener_homogeneous.md",
        ],
        "DifferentialEquations.jl" => [
            "Nonhomogenous Wiener noise" => "sciml/wiener_nonhomogeneous.md",
            "Homogenous Wiener noise" => "sciml/wiener_homogeneous.md",
        ],
        "Noises" => [
            "Fractional Brownian motion" => "noises/fBm.md",
        ],
        "API" => "api.md",
    ],
    authors = "Peter Kloeden and Ricardo Rosa",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/rode_conv_em",
        edit_link = "main",
    ),
    modules = [RODEConvergence],
)