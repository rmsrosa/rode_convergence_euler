"""
    plot(suite::ConvergenceSuite; ns = suite.ns, shownoise=false, showtarget=true, showapprox=true, idxs=1, noiseidxs=1, noisealpha=((showtarget || showapprox) ? 0.4 : 1.0))

Plot the target solution in `suite.xt`, the noise in `suite.yt`, and a few sample paths in the interval `t0` to `tf`, with different time steps as defined by the number of mesh points in `suite.ns` or by an alternate keyword `ns` with a vector of mesh point numbers.

The noise, the target solution, and the approximations can be displayed or not, according to the keywords `shownoise=false`, `showtarget=true`, `showapprox=true` being true or not.

The `linealpha` for plotting the noise can be changed via keyword `noiselpha`.

If `suite` refers to a system of equations (i.e. with `x0law` as a `ContinuousMultivariateDistribution` instead of a `ContinuousUnivariateDistribution`, one can choose to display one or more specific coordinates by specifying the keyword `idxs`, e.g. `idxs=2`, or `idxs=1:3`, or even the sum of all the coordinates, with `idxs=:sum`).

If `noise` is a `ProductProcess`, onde can choose to display one or more specific noise contituents by specifying the keyword `noiseidxs`, e.g. `noiseidxs=1`, or `noiseidexs=2:3`.
"""
plot_suite(suite::ConvergenceSuite, kwargs...) = plot(suite, kwargs...)

@recipe function plot(suite::ConvergenceSuite; ns = suite.ns, shownoise=false, showtarget=true, showapprox=true, idxs=1, noiseidxs=:, xapply=idxs, yapply=noiseidxs, noisealpha=((showtarget || showapprox) ? 0.4 : 1.0))

    # assume suite has been solved and contains the noise in `suite.yt` and the target solution in `suite.xt` and go from there to build a sequence of approximate solutions using the cached vector `suite.xnt`.

    xlabel := "\$t\$"
    ylabel := "\$x\$"
    t0 = suite.t0
    tf = suite.tf
    x0law = suite.x0law
    f = suite.f
    noise = suite.noise
    method = suite.method
    ntgt = suite.ntgt
    yt = suite.yt
    xt = suite.xt
    xnt = suite.xnt

    xapply == (:) && ( xapply = eachindex(eachcol(xt)) )
    xapply isa Int && ( xapply = xapply:xapply )
    yapply == (:) && ( yapply = eachindex(eachcol(yt)) )
    yapply isa Int && ( yapply = yapply:yapply )

    # draw noise
    if shownoise == true
        @series begin
            linecolor --> 2
            linewidth --> 1
            linealpha --> noisealpha
            if noiseidxs == (:)
                noiseidxs = 1:length(noise)
            end
            label --> noise isa UnivariateProcess ?
            "noise" : reshape(["noise $i" for i in idxs], 1, :)
            y = yapply isa Function ?
                map(yapply, eachrow(yt)) :
                yapply == :sum ?
                sum(yt, dims=2) :
                view(yt, :, yapply)
            range(t0, tf, length=ntgt), y
        end
    end

    # solve and draw approximate solutions at selected resolutions
    if showapprox == true
        for (i, nsi) in enumerate(ns)

            nstep = div(ntgt, nsi)

            if x0law isa ContinuousUnivariateDistribution && noise isa UnivariateProcess
                solve!(view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1)), method)
            elseif x0law isa ContinuousUnivariateDistribution
                solve!(view(xnt, 1:nsi), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*(nsi-1), :), method)
            elseif noise isa UnivariateProcess
                solve!(view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1)), method)
            else
                solve!(view(xnt, 1:nsi, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*(nsi-1), :), method)
            end

            @series begin
                linecolor --> i + 2
                linewidth --> 2
                markers --> 3
                markershape --> :auto
                markercolor --> i + 2
                label --> "n=$nsi"
                x = xapply isa Function ?
                    map(xapply, eachrow(view(xnt, 1:nsi, :))) :
                    xapply == :sum ?
                    sum(view(xnt, 1:nsi, :), dims=2) :
                    view(xnt, 1:nsi, xapply)
                range(t0, tf, length=nsi), x
            end
        end
    end

    # draw target solution
    if showtarget == true
        @series begin
            linecolor --> 1
            linewidth --> 2
            label --> "target"
            x = xapply isa Function ?
                map(xapply, eachrow(xt)) :
                xapply == :sum ?
                sum(xt, dims=2) :
                view(xt, :, xapply)
            range(t0, tf, length=ntgt), x
        end
    end
end

"""
    plot(results::ConvergenceResult)

Plot the convergence estimate in a log-log scale (time step vs strong error) based on the values provided in `results`, as computed by [`solve(::ConvergenceSuite)`](@ref).

A scatter plot for the `results.errors` and a line plot from the fitted `errors ≈ C Δtᵖ`, where `C = exp(lc)`, with `Δt` in `results.deltas`, `lc = results.lc`, and `p = results.p`.
"""
plot_convergence(results::ConvergenceResult) = plot(results)

@recipe function plot(results::ConvergenceResult)
    deltas = results.deltas
    lc = results.lc
    p = results.p
    errors = results.errors
    fit = exp(lc) * deltas .^ p

    xlabel := "\$\\Delta t\$"
    ylabel := "\$\\textrm{error}\$"
    xscale := :log10
    yscale := :log10

    @series begin
        linestyle --> :solid
        label --> "\$C\\Delta t^p\$ fit with p = $(round(p, digits=2))"
        deltas, fit
    end

    @series begin
        seriestype --> :scatter
        label --> "strong errors"
        deltas, errors
    end
end