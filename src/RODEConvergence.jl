module RODEConvergence

using Random
using Distributions
using LinearAlgebra
using FFTW
using Plots

import Random: rand

# rode
export RODEIVP

# noises
export AbstractProcess, UnivariateProcess, MultivariateProcess
export WienerProcess, GeometricBrownianMotionProcess
export CompoundPoissonProcess, PoissonStepProcess
export TransportProcess
export FractionalBrownianMotionProcess
export ProductProcess

# solvers
export solve_euler!, solve_heun!
export RandomEuler, RandomHeun, solve!

# convergence calculation
export ConvergenceSuite, ConvergenceResults, solve

# output
export plot_sample_approximations, generate_error_table, plot_dt_vs_error, plot_t_vs_errors

include("rode.jl")
include("noises.jl")
include("solvers.jl")
include("solvers_old.jl")
include("convergence.jl")
include("output.jl")

end # module RODEConvergence
