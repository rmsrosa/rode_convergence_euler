@recipe function plot(suite::ConvergenceSuite)

    # assume suite has been solved and contains the noise in `suite.yt` and the target solution in `suite.xt` and go from there to build a sequence of approximate solutions using the cached vector `suite.xnt`.

    t0 = suite.t0
    tf = suite.tf
    x0law = suite.x0law
    f = suite.f
    noise = suite.noise
    method = suite.method
    ntgt = suite.ntgt
    ns = suite.ns
    yt = suite.yt
    xt = suite.xt
    xnt = suite.xnt

    # draw noise
    @series begin
        linecolor --> :red
        linealpha --> 0.2
        label --> "noise"
        range(t0, tf, length=ntgt), yt
    end

    # draw target solution
    @series begin
        linecolor --> :blue
        label --> "target"
        range(t0, tf, length=ntgt), xt
    end

    # solve and draw approximate solutions at selected resolutions
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
            linecolor --> :green
            linestyle --> :auto
            label --> "$nsi"
            range(t0, tf, length=nsi), view(xnt, 1:nsi)
        end
    end
end