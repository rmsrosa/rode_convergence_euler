using DiffEqNoiseProcess, StochasticDiffEq, Plots, DiffEqDevTools, Random

f(u, p, t, W) = p * u + W
p = -1.0

function f_analytic!(sol)
    empty!(sol.u_analytic)

    u0 = sol.prob.u0
    p = sol.prob.p
    push!(sol.u_analytic, u0)

    t(i) = sol.W.t[i]
    W(i) = sol.W.W[i]

    ti1, Wi1 = sol.W.t[1], sol.W.W[1]
    expintegral1 = 1.0
    integral2 = 0.0
    for i in 2:length(sol)
        ti, Wi = sol.W.t[i], sol.W.W[i]
        expaux = exp(p * (ti - ti1))
        expintegral1 *= expaux
        integral2 = expaux * (integral2 + (Wi + Wi1) * (ti - ti1) / 2)
        push!(sol.u_analytic, u0 * expintegral1 + integral2)
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

prob = RODEProblem(ff, X0, tspan, p)

ensprob = EnsembleProblem(prob)

enssol = solve(ensprob, RandomEM(), dt = 1/100, trajectories=1000)

reltols = 1.0 ./ 10.0 .^ (1:5)
abstols = reltols
dts = 1.0./5.0.^((1:length(reltols)) .+ 1)
setups = [
    Dict(:alg=>RandomEM(), :dts => dts)
]
N = 5_000
wp = WorkPrecisionSet(prob,abstols,reltols,setups;numruns=N,maxiters=1e7,error_estimate=:L∞)