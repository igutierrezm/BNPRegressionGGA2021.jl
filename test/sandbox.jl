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
    dy = MixtureModel(Normal, [(-1.0, 0.5), (1.0, 0.5)], [0.5, 0.5])
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(rng, dy, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect
    return y0, X0, y1, X1
end

N = 2000;
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample_01(rng, N);
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 4000, burnin = 2000);

f1 = mean(chainf);
plot(
    layer(x = y1, y = f1, Geom.line, color=["bnp"]), 
    layer(x = y0, Geom.density, color=["kden"])
)

#=========================#
# Example 2 - 1 predictor #
#=========================#

# Simulate a sample 
function simulate_sample(rng, N)
    # Atoms
    τ = rand(rng, Gamma(2, 2), N)
    μ = randn(rng, N)
    @. μ *= √(2 / τ)

    # Predictors
    x0 = [-0.5, 0.5]
    X0 = [rand(rng, x0, N) randn(rng, N)]
    
    # Outcomes
    β0 = [2.0, 0.0]
    y0 = zeros(N)
    r0 = zeros(Int, N)
    for i in 1:N
        θ0 = 1 - 1 / (1 + exp( - X0[i, :] ⋅ β0))
        ri = 1 + rand(rng, NegativeBinomial(2, θ0))
        d0 = rand(rng, 1:ri)
        y0[i] = rand(rng, Normal(μ[d0], 1 / √τ[d0]))
        r0[i] = ri
    end

    # Grid
    N1 = 50
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [x1 zeros(2 * N1)]

    return y0, X0, y1, X1, r0
end

N = 1000
rng = MersenneTwister(1);
y0, X0, y1, X1, r0 = simulate_sample(rng, N);
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
plot(x = y0, color = X0[:, 1], Geom.density)
plot(x = y1, y = fb, color = X1[:, 1], Geom.line)
mean(chainβ)


#=========================#
# Example 2 - 1 predictor, now using a mixture #
#=========================#

# Simulate a sample 
function simulate_sample(rng, N)
    # Atoms
    τ = rand(rng, Gamma(1, 1), N)
    μ = randn(rng, N)
    @. μ *= √(2 / τ)

    # Predictors
    x0 = [-1, 1]
    X0 = [rand(rng, x0, N) randn(rng, N)]
    
    # Outcomes
    β0 = [2.0, 0.0]
    y0 = zeros(N)
    for i in 1:N
        θ0 = 1 - 1 / (1 + exp( - X0[i, :] ⋅ β0))
        r0 = 1 + rand(rng, NegativeBinomial(2, θ0))
        d0 = rand(rng, 1:r0)
        y0[i] = rand(rng, Normal(μ[d0], 1 / √τ[d0]))
    end

    # Grid
    N1 = 50
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [x1 zeros(2 * N1)]

    return y0, X0, y1, X1
end

N = 1000
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample(rng, N);
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

fb = mean(chainf);
plot(x = y0, color = X0[:, 1], Geom.density)
plot(x = y1, y = fb, color = X1[:, 1], Geom.line)
mean(chainβ)

#===================================================#
# Example 3 - 1 predictor - differences in location #
#===================================================#

# Simulate a sample 
function simulate_sample_03(rng, N0)
    N1 = 50
    dy = MixtureModel(Normal, [(-1.0, 0.5), (1.0, 0.5)], [0.5, 0.5])
    x0 = rand(rng, [-1, 1], N0)
    x1 = [- ones(N1); ones(N1)]
    X0 = [ones(N0) x0]
    X1 = [ones(2 * N1) x1]
    y0 = x0 + rand(rng, dy, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
    return y0, x0, X0, y1, x1, X1
end

N0 = 2000
rng = MersenneTwister(1);
y0, x0, X0, y1, x1, X1 = simulate_sample_03(rng, N0);
data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000, burnin = 5000);

f1 = mean(chainf);
plot(x = y0, color = x0, Geom.density)
plot(x = y1, y = f1, color = X1[:, 2], Geom.line)