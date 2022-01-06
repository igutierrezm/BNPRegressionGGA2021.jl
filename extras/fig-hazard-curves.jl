using BNPRegressionGGA2021
using Cairo
using CSV
using DataFrames
using Distributions
using Gadfly
using Random
using LaTeXStrings

ygrid = LinRange(0, 6, 50) |> collect;
dy1 = LogNormal(0, 0.3);
dy2 = MixtureModel(LogNormal, [(-1.0, 1.0), (0.3, 1.0)], [0.6, 0.4]);
myplot = plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["xᵢ₅ = 1"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["xᵢ₅ = 0"]),
    Guide.xlabel("yᵢ"), 
    Guide.ylabel("density function"), 
);
myplot
myplot = plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid) ./ (1.0 .- cdf.(dy2, ygrid)), Geom.line, color=["xᵢ₅ = 1"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid) ./ (1.0 .- cdf.(dy1, ygrid)), Geom.line, color=["xᵢ₅ = 0"]),
    Guide.xlabel("yᵢ"), 
    Guide.ylabel("hazard function"), 
);
myplot = plot(
    layer(x = ygrid, y = 1.0 .- cdf.(dy2, ygrid), Geom.line, color=[L"$x_{i5} = 1$"]),
    layer(x = ygrid, y = 1.0 .- cdf.(dy1, ygrid), Geom.line, color=[L"$x_{i5} = 0$"]),
    Guide.xlabel(String(L"$y_i$")), 
    Guide.ylabel("survival function"), 
    # style(major_label_font = "Arial", minor_label_font = "Arial"),
);
myplot
draw(SVG("figures/fig-hazard-curves.svg", 4inch, 3inch), myplot)
