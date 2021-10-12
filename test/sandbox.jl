using Revise

using BNPRegressionGGA2021
using Distributions
using Gadfly
using Random
using Statistics
using Test

rng = MersenneTwister(1);
N1, N, D = 50, 1000, 1;
dy = MixtureModel(Normal, [(-2.0, 1.0), (2.0, 1.0)], [0.5, 0.5])
X0 = randn(rng, N, D);
y0 = X0[:, 1] + rand(rng, dy, N);
y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
X1 = [zeros(N1, D); ones(N1, D)];
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chain = BNPRegressionGGA2021.sample(rng, smpl);

fb = mean(chain);
plot(x = y1, y = fb, color = X1[:, 1], Geom.line)