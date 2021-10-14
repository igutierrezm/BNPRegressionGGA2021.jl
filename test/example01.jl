using Revise

using BSplines
using BayesNegativeBinomial
using BNPRegressionGGA2021
using Distributions
using Gadfly
using Random
using Statistics
using Test
using LinearAlgebra

# data
rng = MersenneTwister(1);
N0 = 300;
N1 = 50;
dy = MixtureModel(Normal, [(-2.0, 1), (2.0, 1)], [0.5, 0.5]);
y0 = rand(rng, dy, N0);
y1 = 
    LinRange(extrema(y0)..., N1) |>
    x -> collect(x)
x0 = randn(rng, N0)
basis = BSplineBasis(4, quantile(x0))
X0 = basismatrix(basis, x0)
X1 = basismatrix(basis, zeros(N1))

# Fit
data = BNPRegressionGGA2021.Data(; y0, y1, X0, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chain = BNPRegressionGGA2021.sample(rng, smpl);

# Plot
fh = mean(chain);
plot(
    layer(x = y0, Geom.density, Theme(default_color = colorant"green")),
    layer(x = y1, y = fh, Geom.line, Theme(default_color = colorant"red")),
    layer(x = y1, y = pdf(dy, y1), Geom.line, Theme(default_color = colorant"blue")),
    Guide.manual_color_key("Legend", ["kdensity", "de blasi", "true"], ["green", "red", "blue"])
)
