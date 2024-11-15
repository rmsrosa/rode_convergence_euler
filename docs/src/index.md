## Overview

These are the companion notes for the paper "Strong order-one convergence of the Euler method for random ordinary differential equations driven by semi-martingale noises", by Peter E. Kloeden and Ricardo M. S. Rosa.

We briefly review the main results of the paper and reveal the numerical codes used for the examples presented in the paper.

The codes are written in the [Julia programming language](https://julialang.org). The examples are based on the local package `RODEConvergence.jl`, residing on the folder `src/` of the github repository. It contains the implementation of the Euler method for scalar equations and systems of equations and all the helper functions needed to defined the noises, setup the problem, check the convergence of the method, and plot the desired figures. The methods defined in this local package can be seen in the section [API](api.md).

The local package `RODEConvergence.jl` used here is *not* a registered package in Julia, as it is only used here as a companion code for the paper, with the bare minimum needed for it. For a much more complete package for solving Random ODEs and other types of differential equations, check the [SciML: Open Source Software for Scientific Machine Learning](https://sciml.ai) ecosystem.

For the code used here, it is illustrative to see the first example [Homogenous linear RODE with a Wiener process noise coefficient](examples/01-wiener_linearhomogeneous.md), in which all the steps are explained in more details.

We use a few standard libraries ([Random](https://docs.julialang.org/en/v1/stdlib/Random/), [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/), [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/), [Test](https://docs.julialang.org/en/v1/stdlib/Test/)) and a few packages ([JuliaStats/Distributions.jl](https://juliastats.github.io/Distributions.jl/stable/), [JuliaMath/FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/), [JuliaPlots/Plots.jl](https://docs.juliaplots.org/stable)).

This documentation makes use of [Documenter.jl](https://documenter.juliadocs.org/stable/) and [Literate.jl](https://fredrikekre.github.io/Literate.jl/stable/), with the help of [LiveServer.jl](https://tlienart.github.io/LiveServer.jl/stable/) and [Revise.jl](https://timholy.github.io/Revise.jl/stable/), during development.

Some extra material uses [JuliaCI/BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable).
