using Documenter

makedocs(
    sitename = "Euler method for RODEs",
    pages = [
        "Overview" => "index.md",
        "Theory" => [
            "Main results" => "results.md",
            "Main idea" => "idea.md"
        ],
        "Examples" => [
            "Nonhomogenous Wiener noise" => "wiener_nonhomogeneous.md",
            "Homogenous Wiener noise" => "wiener_homogeneous.md",
            "Homogenous sine Wiener noise" => "sin_wiener_homogeneous.md"
        ],
        "Appendix" => [
            "Fractional Brownian motion" => "fBm.md"
        ]
    ],
    authors = "Ricardo Rosa and Peter Kloeden",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/rode_conv_em",
        edit_link = "main",
    ),
)