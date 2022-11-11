using Plots, FileIO

img = FileIO.load(@__DIR__() * "/grunekloeden_fig4p2.png")

begin
    plt = plot(img)
    points = [
        (1000, 36),
        (1000, 88),
        (552, 422),
        (404, 474),
        (328, 584),
        (284, 552),
        (231, 685),
        (216, 696),
        (204, 652),
        (190, 686),
        (160, 690),
        (106, 710)
    ]
    scatter!(plt, points, markersize=4, markeralpha=0.2, markercolor=:gray, legend=false)
end

i0, j0 = first(points)
il, jl = last(points)

x0, y0 = 0.0, 0.0
xl, yl = 0.5, 1.2

ii = getindex.(points[2:end-1], 1)
jj = getindex.(points[2:end-1], 2)

xx = x0 .+ (ii[2:end-1] .- il) ./ (i0 .- il) * (xl - x0)

yy = yl .- (jj[2:end-1] .- j0) ./ (jl .- j0) * (yl - y0)

lc, p = [one.(xx) log.(xx)] \ log.(yy)
linear_fit = exp(lc) * xx .^ p

plot(xx, yy, scale=:log)
plot!(xx, linear_fit)