using BNPRegressionGGA2021
using Distributions
using Random
using Statistics
using Test

rng = MersenneTwister(1)
N0, N, D = 50, 1000, 1
dy = MixtureModel(Normal, [(-2.0, 1.0), (2.0, 1.0)], [0.5, 0.5])
y0 = rand(rng, dy, N)
X0 = randn(rng, N, D)
y1 = LinRange(-5, 5, N0) |> collect
X1 = zeros(N0, D)
data = BNPRegressionGGA2021.Data(y0, X0, y1, X1)
pa = BNPRegressionGGA2021.Parameters(; N, D)
hp = BNPRegressionGGA2021.HyperParameters(; D)
gq = BNPRegressionGGA2021.GeneratedQuantities(N = N0)
smpl = BNPRegressionGGA2021.Sampler(y0, X0, y1, X1)
chain = BNPRegressionGGA2021.sample(rng, smpl)
fb = mean(chain)
plot(x = y1, y = fb, Geom.line)