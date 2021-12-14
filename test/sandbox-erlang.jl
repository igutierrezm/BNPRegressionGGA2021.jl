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

#=================================#
# Example 1 - 1 trivial predictor #
#=================================#

function simulate_sample_01(rng, N0, N1)
    dy = MixtureModel(LogNormal, [(2.0, 0.5), (0.0, 1)], [0.6, 0.4])
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(rng, dy, N0)
    y1 = LinRange(0, maximum(y0), N1) |> collect
    return y0, X0, y1, X1
end

N0, N1 = 1000, 50;
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
smpl = BNPRegressionGGA2021.ErlangSampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

f1 = mean(chainf);
plot(
    layer(x = y1, y = f1, Geom.line, color=["bnp"]), 
    # layer(x = y0, Geom.density, color=["kden"]),
    # layer(x = y0, Geom.histogram(density = true), color=["hist"])
)