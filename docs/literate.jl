#!/usr/bin/env julia

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Literate

const REPO_ROOT = dirname(@__DIR__)

const LITERATE_DIR = joinpath(REPO_ROOT, "docs", "literate")
const GENERATED_RELATIVE_DIR = "examples"
const GENERATED_DIR = joinpath(REPO_ROOT, "docs", "src", GENERATED_RELATIVE_DIR)
COMMON_SCRIPT = "common_end.jl"

mkpath(GENERATED_DIR)

append_common_script(content) = replace(content, """include(@__DIR__() * "/common_end.jl")""" => read(joinpath(LITERATE_DIR, COMMON_SCRIPT), String))

generated_examples = String[]

for fn in filter(f -> match(r"^\d\d\-(.*)\.jl", f) !== nothing, readdir("docs/literate"))
    Literate.markdown(
        joinpath(LITERATE_DIR, fn),
        GENERATED_DIR,
        documenter=true,
        execute=false,
        preprocess = append_common_script,
        repo_root_url = "https://github.com/rmsrosa/rode_conv_em"
    )
    push!(generated_examples, replace(joinpath(GENERATED_RELATIVE_DIR, fn), ".jl" => ".md"))
end
