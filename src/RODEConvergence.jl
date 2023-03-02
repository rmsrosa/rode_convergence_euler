module RODEConvergence

using Random
using Distributions
using LinearAlgebra
using FFTW
using Plots

import Random: rand

# noises
export WienerProcess, GeometricBrownianMotionProcess
export CompoundPoissonProcess, PoissonStepProcess
export TransportProcess
export FractionalBrownianMotionProcess
export ProductProcess

# solvers
export solve_euler!, solve_heun!

# convergence calculation
export ConvergenceSuite, ConvergenceResults, solve!

export prepare_variables, calculate_errors!, calculate_errors
# output
export plot_sample_approximations, generate_error_table, plot_dt_vs_error, plot_t_vs_errors

include("noises.jl")
include("solvers.jl")
include("convergence.jl")
include("errors.jl")
include("output.jl")

end # module RODEConvergence
