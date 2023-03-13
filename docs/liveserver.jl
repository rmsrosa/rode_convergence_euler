#!/usr/bin/env julia
# From https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/liveserver.jl

# Root of the repository
const REPO_ROOT = dirname(@__DIR__)

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Communicate with docs/make.jl that we are running in live mode
"liveserver" in ARGS || push!(ARGS, "liveserver")

# Prompt for draft mode or not
println("Draft mode [n]? (y/n)")
readline() == "y" ? ("draft" in ARGS || push!(ARGS, "draft")) : filter!(!=("draft"), ARGS)

# Run LiveServer.servedocs(...)
import LiveServer
LiveServer.servedocs(;
    # Documentation root where make.jl and src/ are located
    foldername = joinpath(REPO_ROOT, "docs"),
    # Extra source folder to watch for changes
    include_dirs = [
        # Watch the src folder so docstrings can be Revise'd
        joinpath(REPO_ROOT, "src"),
        # Watch the `docs/literate` so pages from the scripts can be updated
        joinpath(REPO_ROOT, "docs", "literate", "examples"),
    ],
    skip_dirs = [
        # Skip the folder where Literate.jl output is written. This is needed
        # to avoid infinite loops where running make.jl updates watched files,
        # which then triggers a new run of make.jl etc.
        joinpath(REPO_ROOT, "docs/src/examples"),
    ],
    literate=joinpath(REPO_ROOT, "docs/literate/examples")
)