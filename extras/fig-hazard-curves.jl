using BNPRegressionGGA2021
using Cairo
using CSV
using DataFrames
using Distributions
using Gadfly
using Random

ygrid = LinRange(0, 6, 50) |> collect;
dy1 = LogNormal(0, 0.3);
dy2 = MixtureModel(LogNormal, [(-1.0, 1.0), (0.3, 1.0)], [0.6, 0.4]);
myplot = plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["x5 = 1"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["x5 = 0"]),
    Guide.xlabel("y"), 
    Guide.ylabel("density function"), 
);
myplot
myplot = plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid) ./ (1.0 .- cdf.(dy2, ygrid)), Geom.line, color=["x5 = 1"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid) ./ (1.0 .- cdf.(dy1, ygrid)), Geom.line, color=["x5 = 0"]),
    Guide.xlabel("y"), 
    Guide.ylabel("hazard function"), 
);
myplot = plot(
    layer(x = ygrid, y = 1.0 .- cdf.(dy2, ygrid), Geom.line, color=["x5 = 1"]),
    layer(x = ygrid, y = 1.0 .- cdf.(dy1, ygrid), Geom.line, color=["x5 = 0"]),
    Guide.xlabel("y"), 
    Guide.ylabel("survival function"), 
);
myplot
draw(PNG("figures/fig-hazard-curves.png", 4inch, 3inch), myplot)
