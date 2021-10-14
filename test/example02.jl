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
x0 = rand(rng, [1, 2], N0)
# X0 = [ones(N0) rand(rng, [1, 2], N0, 1)];
X0 = [x0[i] == j for i in 1:N0, j in 1:2]
X1 = [ones(N1) ones(N1); ones(N1) zeros(N1)];
y0 = zeros(N0);
for i in 1:N0
    dy = [
        MixtureModel(Normal, [(-2.0, 1), (2.0, 1)], [0.6, 0.4]),
        MixtureModel(Normal, [(-2.0, 1), (2.0, 1)], [0.4, 0.6])
    ]
    y0[i] = rand(rng, dy[Int(x0[i])])
end
y1 = 
    LinRange(extrema(y0)..., N1) |>
    x -> collect(x) |>
    x -> repeat(x, 2)

# Fit
data = BNPRegressionGGA2021.Data(; y0, y1, X0, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chain = BNPRegressionGGA2021.sample(rng, smpl);
fh = mean(chain);

# Plot
plot(x = y1, y = fh, color = X1[:, 2], Geom.line)
plot(x = y0, color = X1[:, 2], Geom.histogram)
