using Documenter
using Literate
using RODEConvergence

ENV["GKSwstype"] = "100"

const repo_root = dirname(@__DIR__)

literate_dir = joinpath(repo_root, "docs", "literate")
generated_relative_dir = "examples"
generated_dir = joinpath(repo_root, "docs", "src", generated_relative_dir)
common_script = "common_end.jl"

mkpath(generated_dir)

append_common_script(content) = replace(content, """include(@__DIR__() * "/common_end.jl")""" => read(joinpath(literate_dir, common_script), String))

generated_examples = String[]

for fn in filter(!=(common_script), readdir("docs/literate"))
    Literate.markdown(joinpath(literate_dir, fn), generated_dir, documenter=true, execute=false, preprocess = append_common_script)
    push!(generated_examples, replace(joinpath(generated_relative_dir, fn), ".jl" => ".md"))
end

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
        "API" => [
            "api/api_overview.md",
            "api/api_noises.md",
            "api/api_solvers.md",
            "api/api_errors.md",
            "api/api_output.md",
            "api/api_extra.md"
        ]
    ],
    authors = "Peter Kloeden and Ricardo Rosa",
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