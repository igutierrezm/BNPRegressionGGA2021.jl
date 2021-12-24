using Revise

# using BSplines
# using BayesNegativeBinomial
using BNPRegressionGGA2021
using Distributions
using Gadfly
using Random
# using Statistics
# using Test
# using LinearAlgebra

#=================================#
# Example 1 - 1 trivial predictor #
#=================================#

function simulate_sample_01(rng, N0, N1)
    dy = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(rng, dy, N0)
    y1 = LinRange(0, 7, N1) |> collect
    return dy, y0, X0, y1, X1
end

N0, N1 = 500, 50;
rng = MersenneTwister(1);
dy, y0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
smpl = BNPRegressionGGA2021.ErlangSampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);
plot(
    layer(x = y1, y = mean(chainf), Geom.line, color=["bnp"]), 
    layer(x = y1, y = pdf.(dy, y1), Geom.line, color=["true"]),
)
smpl.φ
smpl.λ
unique(smpl.d)
maximum(smpl.n)
dy = Erlang(ceil(Int, smpl.φ[3]), 1.0 / smpl.λ[])
mean(dy)
std(dy)
mean(dy)
std(dy)
