begin
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
end

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
dy(x) = Normal(1 + x, 1);
plot(x = y0, color = X0[:, 2], Geom.density)    
plot(
    layer(x = y1, y = fb, color = X1[:, 2], Geom.line),
    layer(x = y1, y = pdf.(dy.(X1[:, 2]), y1), color = X1[:, 2], Geom.line)
)
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

function simulate_sample(N0, N1)
    dϵ = Normal(0.0, 1.0)
    x0 = randn(N0)
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 6, Boundary.knots = c(-4, 4)))")
    y0 = x0 + rand(dϵ, N0)
    x_grid = [-0.5, 0.5]
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
    return y0, x0, X0, y1, x1, X1, mean_y0, std_y0
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, x0, X0, y1, x1, X1, mean_y0, std_y0 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

fb = mean(chainf);
dy1(x) = Normal((x - mean_y0[1]) / std_y0[1], 1 / std_y0[1])
plot(
    layer(x = y1, y = fb, color = x1, Geom.line),
    layer(x = y1, y = pdf.(dy1.(x1), y1), color = x1, Geom.line),
    # layer(x = y0, Geom.histogram)
)

#===========================================================#
# Example 6 - 1 continuous predictor (mixture distribution) #
#===========================================================#

function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(-1.5, 0.5), (1.5, 0.5)], [0.5, 0.5])
    dx = Normal()
    x0 = rand(dx, N0)
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 8, Boundary.knots = c(-5, 5)))")
    y0 = x0 + rand(dϵ, N0)
    x_grid = [-0.5, 0.5]
    y_grid = LinRange(-3, 3, N1)
    y1, x1 = Iterators.product(y_grid, x_grid) |> x -> zip(x...) .|> collect
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 8, Boundary.knots = c(-5, 5)), $x1))")
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
plot(x = y0[-1.5 .< x0 .< -0.5], Geom.density)
plot(x = y0[+0.5 .< x0 .< +1.5], Geom.density)
