#!/usr/bin/env julia

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Literate

const REPO_ROOT = dirname(@__DIR__)

const LITERATE_DIR = joinpath(REPO_ROOT, "docs", "literate", "examples")
const GENERATED_RELATIVE_DIR = "examples"
const GENERATED_DIR = joinpath(REPO_ROOT, "docs", "src", GENERATED_RELATIVE_DIR)
COMMON_SCRIPT = "common_end.jl"

mkpath(GENERATED_DIR)

generated_examples = String[]

for fn in filter(f -> match(r"^\d\d\-(.*)\.jl", f) !== nothing, readdir(LITERATE_DIR))
    Literate.markdown(
        joinpath(LITERATE_DIR, fn),
        GENERATED_DIR,
        documenter=true,
        execute=false,
        repo_root_url = "https://github.com/rmsrosa/rode_conv_em"
    )
    push!(generated_examples, replace(joinpath(GENERATED_RELATIVE_DIR, fn), ".jl" => ".md"))
end
