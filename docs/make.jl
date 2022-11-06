using Documenter

makedocs(
    sitename = "Euler method for RODEs",
    pages = [
        "Overview" => "index.md",
        "Theory" => [
            "Main results" => "results.md",
            "Idea of the proof" => "idea.md"
        ],
        "Examples" => [
            "Nonhomogenous Wiener term" => "wiener_nonhomogeneous.md"
        ]
    ],
    authors = "Ricardo Rosa and Peter Kloeden",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/rode_conv_em",
        edit_link = "main",
    ),
)