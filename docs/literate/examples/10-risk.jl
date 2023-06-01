# # An actuarial risk model
#
# ```@meta
# Draft = false
# ```
#
# A classical model for the surplus $U_t$ at time $t$ of an insurance company is the Cramér–Lundberg model (see [Delbaen & Haezendonck](https://doi.org/10.1016/0167-6687(87)90019-9)) given by
# ```math
#   U_t = U_0 + \gamma t - \sum_{i=1}^{N_t} C_i
# ```
# where $U_0$ is the initial capital, $\gamma$ is a constant premium rate received from the insurees, $C_i$ is a random variable representing the value of the $i$-th claim paid to a given insuree, and $N_t$ is the number of claims up to time $t$. The process $\{N_t\}_t$ is modeled as a Poisson counter, so that the accumulated claims form a compound Poisson process. It is also common to use inhomogeneous Poisson processes and Hawkes self-exciting process, or combinations of such processes, but the classical model uses a compound Poisson process.
#
# The model above, however, does not take into account the variability of the premium rate received by the company, nor the investiment of the accumulated reserves, among other things. Several diffusion type models have been proposed to account for these and other factors. We will consider a simple model, with a randomly perturbed premium and with variable rentability.
#
# More precisely, we start by rewriting the above expression as the following jump differential equation
# ```math
#   \mathrm{d}U_t = \gamma\;\mathrm{d}t - \mathrm{d}C_t,
# ```
# where
# ```math
#   C_t = \sum_{i=1}^{N_t} C_i.
# ```
#
# The addition of an interest rate leads to
# ```math
#   \mathrm{d}U_t = \mu U_t \mathrm{d}t + \gamma\;\mathrm{d}t - \mathrm{d}C_t,
# ```
#
# Assuming a premium rate perturbed by a white noise and assuming the interest rate as a process $\{M_t\}_t$, we find
# ```math
#   \mathrm{d}U_t = M_t U_t\;\mathrm{d}t + \gamma\;\mathrm{d}t + \varepsilon\;\mathrm{d}W_t - \mathrm{d}C_t,
# ```
# so the equation becomes
# ```math
#   \mathrm{d}U_t = (\gamma + M_t U_t)\;\mathrm{d}t + \varepsilon\;\mathrm{d}W_t - \mathrm{d}C_t.
# ```
#
# Since we can compute exactly the accumulated claims $C_t$, we subtract it from $U_t$ to get rid of the jump term. We also subtract an Ornstein-Uhlenbeck process, in the classical way to transform an SDE into a RODE. So, defining
# ```math
#   X_t = U_t - C_t - O_t
# ```
# where $\{O_t\}_t$ is defined by
# ```math
#   \mathrm{d}O_t = \gamma\;\mathrm{d}t + \varepsilon\;\mathrm{d}W_t,
# ```
# we find
# ```math
#   \mathrm{d}X_t = M_t U_t\;\mathrm{d}t = M_t (X_t + C_t + O_t)\;\mathrm{d}t.
# ```
#
# This leads us to the linear random ordinary differential equation
# ```math
#   \frac{\mathrm{d}X_t}{\mathrm{d}t} = M_t X_t + M_t (C_t + O_t).
# ```
#