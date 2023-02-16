using Literate
# using RODEConvergence

Literate.markdown(@__DIR__() * "/scripts/wiener_linearhomogeneous.md", @__DIR__() * "/src/examples/", documenter=true, execute=false)