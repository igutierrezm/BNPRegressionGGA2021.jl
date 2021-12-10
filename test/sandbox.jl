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
    dy = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(rng, dy, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect
    return y0, X0, y1, X1
end

N0, N1 = 1000, 50;
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
smpl = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 4000, burnin = 2000);

f1 = mean(chainf);
plot(
    layer(x = y1, y = f1, Geom.line, color=["bnp"]), 
    layer(x = y0, Geom.density, color=["kden"]),
    layer(x = y0, Geom.histogram(density = true), color=["hist"])
)

#===============================================#
# Example 2 - 1 predictor (simple distribution) #
#===============================================#

# Simulate a sample 
function simulate_sample(rng, N0, N1)
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(rng, x0, N0)]
    y0 = X0 * ones(2) + randn(rng, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [ones(2 * N1) x1]
    return y0, X0, y1, X1, r0
end

N0, N1 = 1000, 50
rng = MersenneTwister(1);
y0, X0, y1, X1, r0 = simulate_sample(rng, N0, N1);
smpl = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
plot(x = y0, color = X0[:, 2], Geom.density)
plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
mean(chainβ)

#================================================#
# Example 3 - 1 predictor (mixture distribution) #
#================================================#

# Simulate a sample 
function simulate_sample(rng, N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(rng, x0, N0)]
    y0 = X0 * ones(2) + rand(rng, dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [ones(2 * N1) x1]
    return y0, X0, y1, X1, r0
end

N0, N1 = 1000, 50
rng = MersenneTwister(1);
y0, X0, y1, X1, r0 = simulate_sample(rng, N0, N1);
smpl = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
plot(x = y0, color = X0[:, 2], Geom.histogram(density = true, bincount = 50))
plot(x = y0, color = X0[:, 2], Geom.density)
plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
mean(chainβ)

#=================================================#
# Example 3 - 2 predictors (mixture distribution) #
#=================================================#

# Simulate a sample 
function simulate_sample(rng, N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(rng, x0, N0, 2)]
    y0 = X0 * [ones(2); 0] + rand(rng, dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 4)
    x1 = kron([x0[1], x0[2], x0[1], x0[2]], ones(N1))
    x2 = kron([x0[1], x0[1], x0[2], x0[2]], ones(N1))
    X1 = [ones(4 * N1) x1 x2]
    return y0, X0, y1, X1, r0
end

N0, N1 = 1000, 50
rng = MersenneTwister(1);
y0, X0, y1, X1, r0 = simulate_sample(rng, N0, N1);
smpl = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
X0concat = @. string(X0[:, 2]) * " " * string(X0[:, 3])
X1concat = @. string(X1[:, 2]) * " " * string(X1[:, 3])
plot(x = y0, color = X0concat, Geom.density)
plot(x = y1, y = fb, color = X1concat, Geom.line)
mean(chainβ)

#=================================================#
# Example 4 - 3 predictors (mixture distribution) #
#=================================================#

# Simulate a sample 
function simulate_sample(rng, N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(rng, x0, N0, 3)]
    y0 = X0 * [1, 0, 1, 0] + rand(rng, dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 8)
    x1 = kron([x0[1], x0[1], x0[1], x0[1], x0[2], x0[2], x0[2], x0[2]], ones(N1))
    x2 = kron([x0[1], x0[1], x0[2], x0[2], x0[1], x0[1], x0[2], x0[2]], ones(N1))
    x3 = kron([x0[1], x0[2], x0[1], x0[2], x0[1], x0[2], x0[1], x0[2]], ones(N1))
    X1 = [ones(8 * N1) x1 x2 x3]
    return y0, X0, y1, X1, r0
end

N0, N1 = 1000, 50
rng = MersenneTwister(1);
y0, X0, y1, X1, r0 = simulate_sample(rng, N0, N1);
smpl = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
X0concat = @. string(X0[:, 2]) * " " * string(X0[:, 3]) * " " * string(X0[:, 4]);
X1concat = @. string(X1[:, 2]) * " " * string(X1[:, 3]) * " " * string(X1[:, 4]);
plot(x = y0, color = X0concat, Geom.density)
plot(x = y1, y = fb, color = X1concat, Geom.line)
mean(chainβ)

#==================================================#
# Example 5 - 10 predictors (mixture distribution) #
#==================================================#

# Simulate a sample 
function simulate_sample(rng, N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(rng, x0, N0, 10)]
    y0 = X0 * [1; zeros(9); 1] + rand(rng, dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = kron([x0[1], x0[2]], ones(N1))
    X1 = [ones(2 * N1) x1 ones(2 * N1, 9)]
    return y0, X0, y1, X1, r0
end

N0, N1 = 1000, 50
rng = MersenneTwister(1);
y0, X0, y1, X1, r0 = simulate_sample(rng, N0, N1);
smpl = BNPRegressionGGA2021.Sampler(; y0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
X0concat = @. string(X0[:, 2]) * " " * string(X0[:, 3]) * " " * string(X0[:, 4]);
X1concat = @. string(X1[:, 2]) * " " * string(X1[:, 3]) * " " * string(X1[:, 4]);
plot(x = y0, color = X0concat, Geom.density)
plot(x = y1, y = fb, color = X1concat, Geom.line)
mean([chainβ[i] .== zeros(11) for i in 1:length(chainβ)])

# #=========================#
# # Example 2 - 1 predictor, now using a mixture #
# #=========================#

# # Simulate a sample 
# function simulate_sample(rng, N)
#     # Atoms
#     τ = rand(rng, Gamma(1, 1), N)
#     μ = randn(rng, N)
#     @. μ *= √(2 / τ)

#     # Predictors
#     x0 = [-1, 1]
#     X0 = [rand(rng, x0, N) randn(rng, N)]
    
#     # Outcomes
#     β0 = [2.0, 0.0]
#     y0 = zeros(N)
#     for i in 1:N
#         θ0 = 1 - 1 / (1 + exp( - X0[i, :] ⋅ β0))
#         r0 = 1 + rand(rng, NegativeBinomial(2, θ0))
#         d0 = rand(rng, 1:r0)
#         y0[i] = rand(rng, Normal(μ[d0], 1 / √τ[d0]))
#     end

#     # Grid
#     N1 = 50
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
#     x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
#     X1 = [x1 zeros(2 * N1)]

#     return y0, X0, y1, X1
# end

# N = 1000
# rng = MersenneTwister(1);
# y0, X0, y1, X1 = simulate_sample(rng, N);
# data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
# smpl = BNPRegressionGGA2021.Sampler(data);
# chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

# fb = mean(chainf);
# plot(x = y0, color = X0[:, 1], Geom.density)
# plot(x = y1, y = fb, color = X1[:, 1], Geom.line)
# mean(chainβ)

# #===================================================#
# # Example 3 - 1 predictor - differences in location #
# #===================================================#

# # Simulate a sample 
# function simulate_sample_03(rng, N0)
#     N1 = 50
#     dy = MixtureModel(Normal, [(-1.0, 0.5), (1.0, 0.5)], [0.5, 0.5])
#     x0 = rand(rng, [-1, 1], N0)
#     x1 = [- ones(N1); ones(N1)]
#     X0 = [ones(N0) x0]
#     X1 = [ones(2 * N1) x1]
#     y0 = x0 + rand(rng, dy, N0)
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
#     return y0, x0, X0, y1, x1, X1
# end

# N0 = 2000
# rng = MersenneTwister(1);
# y0, x0, X0, y1, x1, X1 = simulate_sample_03(rng, N0);
# data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
# smpl = BNPRegressionGGA2021.Sampler(data);
# chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

# f1 = mean(chainf);
# plot(x = y0, color = x0, Geom.density)
# plot(x = y1, y = f1, color = X1[:, 2], Geom.line)