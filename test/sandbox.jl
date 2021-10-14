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

function simulate_sample_01(rng, N0)
    N1 = 50
    dy = MixtureModel(Normal, [(-2.0, 0.5), (2.0, 0.5)], [0.5, 0.5])
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(rng, dy, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    return y0, X0, y1, X1
end

N = 500
rng = MersenneTwister(1)
simulate_sample_01(rng, N)
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chain = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 4000, burnin = 2000);



# # Simulate a sample 
# function simulate_sample(rng, N)
#     # Atoms
#     τ = rand(rng, Gamma(1, 1), N)
#     μ = randn(rng, N)
#     @. μ *= √(2 / τ)

#     # Predictors
#     x0 = [-0.5, 0.5]
#     X0 = rand(rng, x0, N)[:, :]
    
#     # Outcomes
#     β0 = [0.5]
#     y0 = zeros(N)
#     for i in 1:N
#         θ0 = 1 / (1 + exp(X0[i, :] ⋅ β0))
#         r0 = 1 + rand(rng, NegativeBinomial(2, 1 - θ0))
#         d0 = rand(rng, 1:r0)
#         y0[i] = rand(rng, Normal(μ[d0], 1 / √τ[d0]))
#     end

#     # Grid
#     N1 = 50
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
#     X1 = [x0[1] * ones(N1); x0[2] * ones(N1)][:, :]

#     return y0, X0, y1, X1
# end

# N = 100
# rng = MersenneTwister(1);
# y0, X0, y1, X1 = simulate_sample(rng, N)
# data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
# smpl = BNPRegressionGGA2021.Sampler(data);
# chain = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 4000, burnin = 2000);

# fb = mean(chain);
# plot(x = y1, y = fb, color = x1, Geom.line)
# # plot(x = y0, color = X0[:, 2], Geom.histogram)
