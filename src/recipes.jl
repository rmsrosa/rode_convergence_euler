@recipe function plot(suite::ConvergenceSuite; ns = suite.ns, shownoise=false, showtarget=true, showapprox=true, idxs=1, noisealpha=((showtarget || showapprox) ? 0.4 : 1.0))

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

    # draw noise
    if shownoise
        @series begin
            linecolor --> 2
            linewidth --> 1
            linealpha --> noisealpha
            label --> "noise"
            range(t0, tf, length=ntgt), yt
        end
    end

    # solve and draw approximate solutions at selected resolutions
    if showapprox
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
                x = x0law isa ContinuousUnivariateDistribution ?
                    view(xnt, 1:nsi) : view(xnt, 1:nsi, idxs)
                range(t0, tf, length=nsi), x
            end
        end
    end

    # draw target solution
    if showtarget
        @series begin
            linecolor --> 1
            linewidth --> 2
            label --> "target"
            x = x0law isa ContinuousUnivariateDistribution ?
                xt : view(xt, :, idxs)
            range(t0, tf, length=ntgt), x
        end
    end
end
