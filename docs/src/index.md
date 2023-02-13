## Overview

These are the companion notes for the paper "Improved error estimate for the strong order of convergence of the Euler method for random ordinary differential equations", by Peter E. Kloeden and Ricardo M. S. Rosa.

We briefly review the main results of the paper and reveal the numerical codes used in the examples presented in the paper.

The code is written in the [Julia programming language](https://julialang.org), using a few standard libraries ([Random](https://docs.julialang.org/en/v1/stdlib/Random/), [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/), [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/), [Test](https://docs.julialang.org/en/v1/stdlib/Test/)) and a few packages ([JuliaStats/Distributions.jl](https://github.com/JuliaStats/Distributions.jl), [JuliaMath/FFTW.jl](https://github.com/JuliaMath/FFTW.jl), [JuliaPlots/Plots.jl](https://github.com/JuliaPlots/Plots.jl)). Some extra material uses [JuliaCI/BenchmarkToolsjl](https://github.com/JuliaCI/BenchmarkTools.jl) and the [SciML ecosystem](https://sciml.ai).
