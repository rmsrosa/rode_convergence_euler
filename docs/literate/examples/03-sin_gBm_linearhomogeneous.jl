# # Homogenous linear RODE with the sine of a Geometric Brownian motion coefficient

# Now we consider a homogeneous linear equation in which the coefficient depends on the sine of a Geometric Brownian motion process.

# ## The equation

# More precisely, we consider the RODE
# ```math
#   \begin{cases}
#     \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = \sin(Y_t) X_t, \qquad 0 \leq t \leq T, \\
#   \left. X_t \right|_{t = 0} = X_0,
#   \end{cases}
# ```
# where $\{Y_t\}_{t\geq 0}$ is a geometric Brownian motion process.

# The explicit solution is
# ```math
#   X_t = e^{\int_0^t \sin(Y_s) \;\mathrm{d}s} X_0.
# ```

# ## Computing a higher order approximation of the solution

# The integral $\int_0^{t_j} \sin(Y_s)\;\mathrm{d}s$ and, hence, the exact solution, is not uniquely defined from the values $W_{t_j}$ of the noise on the mesh points, no matter how fine it is, and an exact distribution for the collection of exact solutions conditioned on the mesh points is not available in closed form. Hence, we consider an approximation of an exact solution by solving the equation numerically, with the Euler method itself, but in a much higher resolution.

# Indeed, the convergence will be estimated from a set of discretizations with mesh points with time step $\Delta t_N = N$, for $N = N_1 < N_2 < \ldots N_n$, for some $n\in \mathbb{N}$, by comparing the error of such solutions to an approximated solutions computed in a finer mesh with $\Delta t_{\textrm{fine}} = \Delta t_{N_n}^2$, hence with $N_\textrm{fine} = N_n^2$.
#
# ## Numerical approximation
# 
# ### Setting up the problem
# 
# First we load the necessary packages

using Plots
using Random
using Distributions
using RODEConvergence

# Then we set up some variables

rng = Xoshiro(123)

f(t, x, y) = sin(y) * x

t0 = 0.0
tf = 1.0
x0law = Normal()

μ = 1.0
σ = 0.2
y0 = 1.0
noise = GeometricBrownianMotionProcess(t0, tf, y0, μ, σ)

ntgt = 2^18
ns = 2 .^ (4:9)
nsample = ns[[1, 2, 3, 4]]
m = 1_000

# And add some information about the simulation:

info = (
    equation = "\$\\mathrm{d}X_t/\\mathrm{d}t = \\sin(Y_t) X_t\$",
    noise = "a geometric Brownian motion process noise \$\\{Y_t\\}_t\$ (drift=$μ; diffusion=$σ)",
    ic = "\$X_0 \\sim \\mathcal{N}(0, 1)\$"
)

# We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

target = RandomEuler()
method = RandomEuler()

include(@__DIR__() * "/common_end.jl")
