using Revise

using BayesNegativeBinomial
using BNPRegressionGGA2021
using Distributions
using Gadfly
using Random
using RCall
using Statistics
using StatsBase
using Test
using LinearAlgebra

#=================================#
# Example 1 - 1 trivial predictor #
#=================================#

function simulate_sample_01(N0, N1)
    dy = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(dy, N0)
    y1 = LinRange(-3, 3, N1) |> collect
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0[:, 1] .= 1 
    X1[:, 1] .= 1 
    return y0, X0, y1, X1
end;

begin
    Random.seed!(1)
    N0, N1 = 1000, 50;
    rng = MersenneTwister(1);
    y0, X0, y1, X1 = simulate_sample_01(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

f1 = mean(chainf);
plot(
    layer(x = y1, y = f1, Geom.line, color=["bnp"]), 
    layer(x = y0, Geom.density, color=["kden"]),
    layer(x = y0, Geom.histogram(density = true), color=["hist"])
)

#========================================================#
# Example 2 - 1 discrete predictor (simple distribution) #
#========================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0)]
    y0 = X0 * ones(2) + randn(N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [ones(2 * N1) x1]
    return y0, X0, y1, X1
end;

begin
    Random.seed!(1)
    N0, N1 = 1000, 50;
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);    
end;

fb = mean(chainf);
plot(x = y0, color = X0[:, 2], Geom.density)
plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
mean(chainβ)

#=========================================================#
# Example 3 - 1 discrete predictor (mixture distribution) #
#=========================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0)]
    y0 = X0 * ones(2) + rand(dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [ones(2 * N1) x1]
    return y0, X0, y1, X1
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

fb = mean(chainf);
plot(x = y0, color = X0[:, 2], Geom.histogram(density = true, bincount = 50))
plot(x = y0, color = X0[:, 2], Geom.density)
plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
mean(chainβ)

#=================================================#
# Example 4 - 2 predictors (mixture distribution) #
#=================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0, 2)]
    y0 = X0 * [ones(2); 0] + rand(dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 4)
    x1 = kron([x0[1], x0[2], x0[1], x0[2]], ones(N1))
    x2 = kron([x0[1], x0[1], x0[2], x0[2]], ones(N1))
    X1 = [ones(4 * N1) x1 x2]
    return y0, X0, y1, X1
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

fb = mean(chainf);
X0concat = @. string(X0[:, 2]) * " " * string(X0[:, 3])
X1concat = @. string(X1[:, 2]) * " " * string(X1[:, 3])
plot(x = y0, color = X0concat, Geom.density)
plot(x = y1, y = fb, color = X1concat, Geom.line)
mean(chainβ)

#==========================================================#
# Example 5 - 1 continuous predictor (simple distribution) #
#==========================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    dϵ = Normal(0.0, 1.0)
    x0 = randn(N0)
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 6, Boundary.knots = c(-4, 4)))")
    y0 = x0 + rand(dϵ, N0)
    x_grid = [-1, 1]
    y_grid = LinRange(-3, 3, N1)
    y1, x1 = Iterators.product(y_grid, x_grid) |> x -> zip(x...) .|> collect
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 6, Boundary.knots = c(-4, 4)), $x1))")
    # Standardization
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0[:, 1] .= 1 
    X1[:, 1] .= 1 
    return y0, x0, X0, y1, x1, X1
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, x0, X0, y1, x1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

fb = mean(chainf);
plot(x = y1, y = fb, color = x1, Geom.line)

#===========================================================#
# Example 6 - 1 continuous predictor (mixture distribution) #
#===========================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(-1.0, 0.5), (1.0, 0.5)], [0.5, 0.5])
    dx = MixtureModel(Normal, [(-1.0, 0.2), (1.0, 0.2)], [0.5, 0.5])
    x0 = rand(dx, N0)
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 6, Boundary.knots = c(-2, 2)))")
    y0 = x0 + rand(dϵ, N0)
    x_grid = [-1, 1]
    y_grid = LinRange(-3, 3, N1)
    y1, x1 = Iterators.product(y_grid, x_grid) |> x -> zip(x...) .|> collect
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 6, Boundary.knots = c(-2, 2)), $x1))")
    # Standardization
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0[:, 1] .= 1 
    X1[:, 1] .= 1 
    return y0, x0, X0, y1, x1, X1
end

begin
    Random.seed!(1)
    N0, N1 = 2000, 50
    y0, x0, X0, y1, x1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 4000);
end;

fb = mean(chainf);
plot(x = y1, y = fb, color = x1, Geom.line)
plot(x = y0, color = x0 .> 0.0, Geom.density)

# #=================================================#
# # Example 4 - 3 predictors (mixture distribution) #
# #=================================================#

# # Simulate a sample 
# function simulate_sample(N0, N1)
#     dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
#     x0 = [-0.5, 0.5]
#     X0 = [ones(N0) rand(x0, N0, 3)]
#     y0 = X0 * [1, 0, 1, 0] + rand(dϵ, N0)
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 8)
#     x1 = kron([x0[1], x0[1], x0[1], x0[1], x0[2], x0[2], x0[2], x0[2]], ones(N1))
#     x2 = kron([x0[1], x0[1], x0[2], x0[2], x0[1], x0[1], x0[2], x0[2]], ones(N1))
#     x3 = kron([x0[1], x0[2], x0[1], x0[2], x0[1], x0[2], x0[1], x0[2]], ones(N1))
#     X1 = [ones(8 * N1) x1 x2 x3]
#     return y0, X0, y1, X1
# end

# N0, N1 = 1000, 50
# rng = MersenneTwister(1);
# y0, X0, y1, X1 = simulate_sample(N0, N1);
# smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
# chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000, burnin = 5000);

# fb = mean(chainf);
# X0concat = @. string(X0[:, 2]) * " " * string(X0[:, 3]) * " " * string(X0[:, 4]);
# X1concat = @. string(X1[:, 2]) * " " * string(X1[:, 3]) * " " * string(X1[:, 4]);
# plot(x = y0, color = X0concat, Geom.density)
# plot(x = y1, y = fb, color = X1concat, Geom.line)
# mean(chainβ)

# #==================================================#
# # Example 5 - 10 predictors (mixture distribution) #
# #==================================================#

# # Simulate a sample 
# function simulate_sample(N0, N1)
#     dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
#     x0 = [-0.5, 0.5]
#     X0 = [ones(N0) rand(x0, N0, 10)]
#     y0 = X0 * [1; zeros(9); 1] + rand(dϵ, N0)
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
#     x1 = kron([x0[1], x0[2]], ones(N1))
#     X1 = [ones(2 * N1) x1 ones(2 * N1, 9)]
#     return y0, X0, y1, X1
# end

# N0, N1 = 1000, 50
# rng = MersenneTwister(1);
# y0, X0, y1, X1 = simulate_sample(N0, N1);
# smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
# chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000, burnin = 5000);
# mean([chainβ[i] .== zeros(11) for i in 1:length(chainβ)])

# #=========================#
# # Example 2 - 1 predictor, now using a mixture #
# #=========================#

# # Simulate a sample 
# function simulate_sample(N)
#     # Atoms
#     τ = rand(Gamma(1, 1), N)
#     μ = randn(N)
#     @. μ *= √(2 / τ)

#     # Predictors
#     x0 = [-1, 1]
#     X0 = [rand(x0, N) randn(N)]
    
#     # Outcomes
#     β0 = [2.0, 0.0]
#     y0 = zeros(N)
#     for i in 1:N
#         θ0 = 1 - 1 / (1 + exp( - X0[i, :] ⋅ β0))
#         r0 = 1 + rand(NegativeBinomial(2, θ0))
#         d0 = rand(1:r0)
#         y0[i] = rand(Normal(μ[d0], 1 / √τ[d0]))
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
# y0, X0, y1, X1 = simulate_sample(N);
# data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
# smpl = BNPRegressionGGA2021.DGSBPNormal(data);
# chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000, burnin = 5000);

# fb = mean(chainf);
# plot(x = y0, color = X0[:, 1], Geom.density)
# plot(x = y1, y = fb, color = X1[:, 1], Geom.line)
# mean(chainβ)

# #===================================================#
# # Example 3 - 1 predictor - differences in location #
# #===================================================#

# # Simulate a sample 
# function simulate_sample_03(N0)
#     N1 = 50
#     dy = MixtureModel(Normal, [(-1.0, 0.5), (1.0, 0.5)], [0.5, 0.5])
#     x0 = rand([-1, 1], N0)
#     x1 = [- ones(N1); ones(N1)]
#     X0 = [ones(N0) x0]
#     X1 = [ones(2 * N1) x1]
#     y0 = x0 + rand(dy, N0)
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2);
#     return y0, x0, X0, y1, x1, X1
# end

# N0 = 2000
# rng = MersenneTwister(1);
# y0, x0, X0, y1, x1, X1 = simulate_sample_03(N0);
# data = BNPRegressionGGA2021.Data(; y0, X0, y1, X1);
# smpl = BNPRegressionGGA2021.DGSBPNormal(data);
# chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000, burnin = 5000);

# f1 = mean(chainf);
# plot(x = y0, color = x0, Geom.density)
# plot(x = y1, y = f1, color = X1[:, 2], Geom.line)


# #===========================================================#
# # Example 4 - 1 continuous predictor (mixture distribution) #
# #===========================================================#

# # Simulate a sample 
# function simulate_sample(N0, N1)
#     dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
#     x0 = [-2, 0, 2]
#     X0 = [ones(N0) randn(N0)]
#     y0 = X0 * [0, 1] + 0.25 * rand(dϵ, N0)
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 3)
#     x1 = kron(x0, ones(N1))
#     X1 = [ones(3 * N1) x1]
#     return y0, X0, y1, X1
# end

# N0, N1 = 1000, 50
# rng = MersenneTwister(1);
# y0, X0, y1, X1 = simulate_sample(N0, N1);
# smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
# chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 20000, burnin = 10000);

# fb = mean(chainf);
# plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
# mean(chainβ)

# #==========================================================#
# # Example 3 - 1 continuous predictor (simple distribution) #
# #==========================================================#

# # Simulate a sample 
# function simulate_sample(N0, N1)
#     x0 = [-2, 0, 2]
#     X0 = [ones(N0) randn(N0)]
#     y0 = X0 * [0, 1] + 0.25 * randn(N0)
#     y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 3)
#     x1 = kron(x0, ones(N1))
#     X1 = [ones(3 * N1) x1]
#     return y0, X0, y1, X1
# end

# N0, N1 = 2000, 50
# rng = MersenneTwister(1);
# y0, X0, y1, X1 = simulate_sample(N0, N1);
# smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
# chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000, burnin = 5000);

# fb = mean(chainf);
# # plot(x = y0, color = X0[:, 2], Geom.density)
# plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
# mean(chainβ)
