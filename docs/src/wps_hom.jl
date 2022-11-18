using StochasticDiffEq, Plots, DiffEqDevTools, Random

f(u, p, t, W) = W * u

function f_analytic!(sol)
    empty!(sol.u_analytic)

    u0 = sol.prob.u0
    push!(sol.u_analytic, u0)

    ti1, Wi1 = sol.W.t[1], sol.W.W[1]
    integral = 0.0
    for i in 2:length(sol)
        ti, Wi = sol.W.t[i], sol.W.W[i]
        integral += (Wi + Wi1) * (ti - ti1) / 2 + 0 * (ti - ti1)^3 / 24 + sqrt((ti - ti1)^3 / 12) * randn()
        push!(sol.u_analytic, u0 * exp(integral))
        ti1, Wi1 = ti, Wi
    end
end

ff = RODEFunction(
    f,
    analytic = f_analytic!,
    analytic_full=true
)

X0 = 1.0
tspan = (0.0, 1.0)

prob = RODEProblem(ff, X0, tspan)

prob_func = (prob,i,repeat) -> (remake(prob, u0 = 1.0 + 0.2 * randn()))
ensprob = EnsembleProblem(prob; prob_func)
#= 
enssol = solve(ensprob, RandomEM(), dt = 1/100, trajectories=1000)

summsol = EnsembleSummary(enssol; quantiles=[0.05,0.95])

plot(summsol, ylims=(-1.0, 2.0)) =#

#

reltols = 1.0 ./ 10.0 .^ (1:5)
abstols = reltols
dts = 1.0./2.0.^((1:length(reltols)) .+ 4)
setups = [
    Dict(:alg=>RandomEM(), :dts => dts),
    Dict(:alg=>RandomHeun(), :dts => dts)
]
N = 100_000
wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:l∞)

begin
    display(plot(wp))
    display(plot(wp, view=:dt_convergence))
end

#

# wp = WorkPrecisionSet(ensprob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:L∞)

#

using Statistics
m = 10_000
n = 1
dt = 1/n

am = exp(sum((dt)^3 / 24 for _ in 1:n))
am == exp(dt^2/24)

b = [exp(sum(sqrt((dt)^3 / 12) * randn() for _ in 1:n)) for _ in 1:m]
bm = mean(b)
bv = var(b)

ci = √(bv/m)

bm - 2ci < am < bm + 2ci
bm - ci < am < bm + ci