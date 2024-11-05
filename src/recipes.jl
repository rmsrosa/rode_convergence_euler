"""
    plot(suite::ConvergenceSuite; ns = suite.ns, xshow=true, yshow=false, noisealpha=nothing, resolution=2^9)

Plot the target solution, the noise and a few sample paths.

Plot the target solution in `suite.xt`, the noise in `suite.yt`, and a few sample paths in the interval `t0` to `tf`, with different time steps as defined by the number of mesh points in `suite.ns` or as given by the keyword `ns` as a vector of integers with the desired numbers of mesh points.

The noise, the target solution, and the approximations can be displayed or not, according to the keywords `xshow`, `yshow` and `ns`. If any of them is set to `false` or `nothing`, then the corresponding series is not showed.

The `linealpha` for plotting the noise can be changed via keyword `noiselpha`.

If `suite` refers to a system of equations (i.e. with `x0law` as a `MultivariateDistribution` instead of a `UnivariateDistribution`, one can choose to display one or more specific coordinates by specifying the keyword `xshow` in several possible ways, e.g. `xshow=2` (for the second coordinate), or `xshow=1:3` (for the first to third coordinates as separate series), or even the sum of all the coordinates, with either `xshow=:sum`, or the Euclidian norm, if `xshow=:norm`, or in any other way if `xshow` is a `Function` acting on each element of `x`, when `x` is a scalar, or on each column of `x`, when `x` is vector-valued, such as `xshow=sqrt`, `xshow=sum` or `xshow=x->2x[1] + x[2]/2`, etc.).

Similary, if `noise` is a `ProductProcess`, onde can choose to display one or more specific noise contituents, or combinations of them, by specifying the keyword `yshow` in the same way as for `xshow` just described.

The `resolution` keyword can be used to set the maximum resolution for the display of paths, which can be originally about a million points, so reducing it to say the default value of `2^9=512` is plenty enough for the visualization and reduces considerably the size of the generated SVG images. When combining multiple plots into one, it helps reducing even more this value, say `resolution=2^7` or `2^8` (128 or 256 points). The resolution must be a factor of `suite.ntgt` and of all the values of `suite.ns`.
"""
plot_suite(suite::ConvergenceSuite, kwargs...) = plot(suite, kwargs...)

@recipe function plot(suite::ConvergenceSuite; ns = suite.ns, xshow=1, yshow=false, noisealpha=nothing, resolution=2^9)

    # assume suite has been solved and contains the noise in `suite.yt` and the target solution in `suite.xt` (from the last sample path of the Monte-Carlo simulation) and go from there to build a sequence of approximate solutions using the cached vector `suite.xnt`.

    color_palette := :tab10
    xlabel := "\$t\$"
    ylabel := ( xshow !== nothing && xshow !== false ) ? "\$x\$" : "\$y\$"
    t0 = suite.t0
    tf = suite.tf
    x0law = suite.x0law
    f = suite.f
    noise = suite.noise
    params = suite.params
    method = suite.method
    ntgt = suite.ntgt
    yt = suite.yt
    xt = suite.xt
    xnt = suite.xnt
    
    xshow = (xshow == (:) || xshow === true ) ? eachindex(eachcol(xt)) : xshow
    yshow = (yshow == (:) || yshow === true ) ? eachindex(eachcol(yt)) : yshow

    (rem(ntgt, resolution) == 0 || ntgt < resolution ) || error(
        "`resolution` keyword must be either a factor of `suite.ntgt` or larger than `suite.ntgt`"
    )

    ( ns === nothing || ns === false || all( n < resolution || rem(n, resolution) == 0  for n in ns ) ) || error(
        "`resolution` keyword must be either a factor of `n` or larger than `n`, for every `n` in `suite.ns`"
    )

    # draw noise
    if ( yshow !== nothing && yshow !== false )
        if noisealpha === nothing
            noisealpha = ( xshow === nothing || xshow === false ) ? 1.0 : 0.4
        end
        @series begin
            linestyle --> :auto
            linecolor --> 2
            linewidth --> 2
            linealpha --> noisealpha
            noiselabel = (yshow isa Function || noise isa UnivariateProcess ) ?
                "noise" :
                yshow == :sum ?
                "sum of noises" :
                yshow == :norm ?
                "l2-norm noises" :
                reshape([string(nameof(typeof(pr))) for pr in noise.processes], 1, :)
            label --> noiselabel
            inds = first(axes(yt, 1)):max(1, div(size(yt, 1) - 1, resolution)):last(axes(yt, 1))
            y = yshow isa Function && noise isa UnivariateProcess ?
                map(yshow, view(yt, inds)) :
                yshow isa Function ?
                map(yshow, eachrow(view(yt, inds, :))) :
                yshow == :sum ?
                sum(view(yt, inds, :), dims=2) :
                yshow == :norm ?
                map(norm, eachrow(view(yt, inds, :))) :
                view(yt, inds, yshow)
            range(t0, tf, length=length(inds)), y
        end
    end

    # solve and draw approximate solutions at selected resolutions
    if (ns !== nothing && ns !== false && !isempty(ns) && xshow !== nothing && xshow !== false)
        for (i, nsi) in enumerate(ns)

            nstep = div(ntgt, nsi)

            if x0law isa UnivariateDistribution && noise isa UnivariateProcess
                solve!(view(xnt, 1:nsi+1), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*nsi), params, method)
            elseif x0law isa UnivariateDistribution
                solve!(view(xnt, 1:nsi+1), t0, tf, xt[1], f, view(yt, 1:nstep:1+nstep*nsi, :), params, method)
            elseif noise isa UnivariateProcess
                solve!(view(xnt, 1:nsi+1, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*nsi), params, method)
            else
                solve!(view(xnt, 1:nsi+1, :), t0, tf, view(xt, 1, :), f, view(yt, 1:nstep:1+nstep*nsi, :), params, method)
            end

            inds = first(axes(view(xnt, 1:nsi+1), 1)):max(1, div(size(view(xnt, 1:nsi+1), 1) - 1, resolution)):last(axes(view(xnt, 1:nsi+1), 1))

            @series begin
                linecolor --> i + 2
                linewidth --> 2
                markers --> 3
                markershape --> :auto
                markercolor --> i + 2
                label --> "n=$nsi"
                x = xshow isa Function && x0law isa UnivariateDistribution ?
                    map(xshow, view(xnt, inds)) :
                    xshow isa Function ?
                    map(xshow, eachrow(view(xnt, inds, :))) :
                    xshow == :sum ?
                    sum(view(xnt, inds, :), dims=2) :
                    xshow == :norm ?
                    map(norm, eachrow(view(xnt, inds, :))) :
                    view(xnt, inds, xshow)
                range(t0, tf, length=length(inds)), copy(x)
            end
        end
    end

    # draw target solution
    if ( xshow !== nothing && xshow !== false )
        @series begin
            linecolor --> 1
            linewidth --> 2
            sollabel = (xshow isa Function || x0law isa UnivariateDistribution ) ?
                "target" :
                xshow == :sum ?
                "sum of target coordinates" :
                xshow == :norm ?
                "l2-norm of target" :
                reshape(["target $i" for i in 1:length(x0law)], 1, :)
            label --> sollabel
            inds = first(axes(xt, 1)):max(1, div(size(xt, 1) - 1, resolution)):last(axes(xt, 1))
            x = xshow isa Function && x0law isa         UnivariateDistribution ?
                map(xshow, view(xt, inds)) :
                xshow isa Function ?
                map(xshow, eachrow(view(xt, inds, :))) :
                xshow == :sum ?
                sum(view(xt, inds, :), dims=2) :
                xshow == :norm ?
                map(norm, eachrow(view(xt, inds, :))) :
                view(xt, inds, xshow)
            range(t0, tf, length=length(inds)), copy(x)
        end
    end
end

"""
    plot(results::ConvergenceResult)

Plot the convergence estimate in a log-log scale (time step vs strong error).

It is based on the values provided in `results`, as computed by [`solve(::ConvergenceSuite)`](@ref).

The plot consists of a scatter plot for the `results.errors` and a line plot from the fitted `errors ≈ C Δtᵖ`, where `C = exp(lc)`, with `Δt` in `results.deltas`, `lc = results.lc`, and `p = results.p`, with appropriate legends.
"""
plot_convergence(results::ConvergenceResult) = plot(results)

@recipe function plot(results::ConvergenceResult)
    deltas = results.deltas
    lc = results.lc
    p = results.p
    pmin = results.pmin
    pmax = results.pmax
    errors = results.errors
    stderrs = results.stderrs
    fit = exp(lc) * deltas .^ p

    xlabel := "\$\\Delta t\$"
    ylabel := "\$\\textrm{error}\$"
    xscale := :log10
    yscale := :log10
    color_palette := :tab10

    @series begin
        linestyle --> :solid
        label --> "\$C\\Delta t^p\$ fit with p = $(round(p, digits=2)); 95% CI [$(round(pmin, digits=2)), $(round(pmax, digits=2))]"
        deltas, fit
    end

    @series begin
        seriestype --> :scatter
        label --> "strong errors"
        yerror --> 2stderrs
        deltas, errors
    end
end