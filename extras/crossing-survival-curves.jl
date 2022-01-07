using BNPRegressionGGA2021
using Distributions
using Gadfly
using LaTeXStrings

begin
    ygrid = LinRange(0, 6, 50) |> collect;
    dy1 = LogNormal(0.3, 0.7);
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (0.3, 1.2)], [0.6, 0.4]);    
    fy1 = pdf.(dy1, ygrid);
    fy2 = pdf.(dy2, ygrid);
    Sy1 = 1.0 .- cdf.(dy1, ygrid);
    Sy2 = 1.0 .- cdf.(dy2, ygrid);
    hy1 = pdf.(dy1, ygrid) ./ Sy1;
    hy2 = pdf.(dy2, ygrid) ./ Sy2;
    hry = hy2 ./ hy1;
end 

# Densities
plot_fy = plot(
    layer(x = ygrid, y = fy2, Geom.line, color=["xᵢ₅ = 1"]),
    layer(x = ygrid, y = fy1, Geom.line, color=["xᵢ₅ = 0"]),
    Guide.xlabel("zᵢ"), 
    Guide.ylabel("density function"), 
)

# Hazard curves
plot_hy = plot(
    layer(x = ygrid, y = hy2, Geom.line, color=["xᵢ₅ = 1"]),
    layer(x = ygrid, y = hy1, Geom.line, color=["xᵢ₅ = 0"]),
    Guide.xlabel("zᵢ"), 
    Guide.ylabel("hazard function"), 
)

x, y = 0.55*rand(4), 0.55*rand(4)

# Survival curves
# plot_sy = plot(
#     layer(x = ygrid, y = Sy2, Geom.line, Geom.point, shape=[String(L"$x_{i5} = 1$")]),
#     layer(x = ygrid, y = Sy1, Geom.line, Geom.point, shape=[String(L"$x_{i5} = 0$")]),
#     # Theme(point_shapes = [Shape.square, Shape.xcross], default_color = "grey")
#     # layer(x = ygrid, y = Sy1, Geom.line, Geom.point, shape=[L"$x_{i5} = 0$"]),
#     # Guide.xlabel(String(L"$z_i$")), 
#     # Guide.ylabel("survival function"), 
#     # style(key_position = :top)
# )

plot_sy = plot(
    layer(x = ygrid, y = Sy2, Geom.line, linestyle=[String(L"$x_{i5} = 1$")]),
    layer(x = ygrid, y = Sy1, Geom.line, linestyle=[String(L"$x_{i5} = 0$")]),
    Scale.linestyle_discrete(levels = String.([L"$x_{i5} = 1$", L"$x_{i5} = 0$"])),
    Theme(default_color = "grey", line_style = [:dot, :dash])
    # layer(x = ygrid, y = Sy1, Geom.line, Geom.point, shape=[L"$x_{i5} = 0$"]),
    # Guide.xlabel(String(L"$z_i$")), 
    # Guide.ylabel("survival function"), 
    # style(key_position = :top)
)


# Hazard ratio
plot_hr = plot(
    x = ygrid, 
    y = hry, 
    Geom.line,
    Guide.ylabel("hazard ratio"), 
    Guide.xlabel(String(L"$z_i$")), 
)

draw(SVG("figures/crossing-survival-curves.svg", 2.5inch, 3inch), plot_sy)
draw(SVG("figures/non-proportional-hazards.svg", 2.5inch, 3inch), plot_hr)
