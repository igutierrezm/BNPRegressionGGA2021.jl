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

rng = MersenneTwister(1);
basis = BSplineBasis(4, -5:2:5);
N1, N, D = 50, 1000, 1;
dy = MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)], [0.5, 0.5])
x0 = randn(rng, N)
X0 = basismatrix(basis, x0);
y0 = x0 + rand(rng, dy, N);
y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
x1 = [zeros(N1); ones(N1)];
X1 = basismatrix(basis, x1);
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chain = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chain);
plot(x = y1, y = fb, color = x1, Geom.line)
# plot(x = y0, color = X0[:, 2], Geom.histogram)
