## Overview

These are the companion notes for the paper "Improved error estimate for the strong order of convergence of the Euler method for random ordinary differential equations", by Peter E. Kloeden and Ricardo M. S. Rosa.

We briefly review the main results of the paper and reveal the numerical codes used in the examples presented in the paper.

The code is written in the [Julia programming language](https://julialang.org), using a few standard libraries ([Random](https://docs.julialang.org/en/v1/stdlib/Random/), [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/), [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/), [Test](https://docs.julialang.org/en/v1/stdlib/Test/)) and a few packages ([JuliaStats/Distributions.jl](https://juliastats.github.io/Distributions.jl/stable/), [JuliaMath/FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/), [JuliaPlots/Plots.jl](https://docs.juliaplots.org/stable)).

This documentation makes use of [Documenter.jl](https://documenter.juliadocs.org/stable/) and [Literate.jl](https://fredrikekre.github.io/Literate.jl/stable/), with the help of [LiveServer.jl](https://tlienart.github.io/LiveServer.jl/stable/) and [Revise.jl](https://timholy.github.io/Revise.jl/stable/), during development.

Some extra material uses [JuliaCI/BenchmarkToolsjl](https://juliaci.github.io/BenchmarkTools.jl/stable) and the [SciML ecosystem](https://sciml.ai).
