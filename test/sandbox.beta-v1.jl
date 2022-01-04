using Revise

using BNPRegressionGGA2021
using Distributions
using Gadfly
using Random

const BNP = BNPRegressionGGA2021

#=================================#
# Example 1 - 1 trivial predictor #
#=================================#

function simulate_sample(N0, N1)
    dy = MixtureModel(Beta, [(1.0, 5.0), (10.0, 2.0)], [0.4, 0.6])
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(dy, N0)
    y1 = LinRange(0.0, 1.0, N1) |> collect
    return dy, y0, X0, y1, X1
end;

Random.seed!(1);
N0, N1 = 500, 50;
dy, y0, X0, y1, X1 = simulate_sample(N0, N1);
m = BNP.DGPMBeta(; y0, X0, y1, X1);
# m.skl.s[1] = 5
chainf, chainβ = BNP.sample!(m; mcmcsize = 10000, burnin = 5000);
plot(
    layer(x = y1, y = mean(chainf), Geom.line, color = ["bnp"]), 
    layer(x = y1, y = pdf.(dy, y1), Geom.line, color = ["true"]),
    layer(x = y0, Geom.histogram(density = true, bincount = 50), color = ["hist"]),
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
end

N0, N1 = 500, 50
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample(N0, N1);
m = BNP.DGPMNormal(; y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);

fb = mean(chainf);
plot(x = y0, color = X0[:, 2], Geom.density)
plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
mean(chainβ)

#=================================================#
# Example 3 - 2 predictors (mixture distribution) #
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

N0, N1 = 500, 50
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample(N0, N1);
m = BNP.DGPMNormal(; y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);

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
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0, 3)]
    y0 = X0 * [1, 0, 1, 0] + rand(dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 8)
    x1 = kron([x0[1], x0[1], x0[1], x0[1], x0[2], x0[2], x0[2], x0[2]], ones(N1))
    x2 = kron([x0[1], x0[1], x0[2], x0[2], x0[1], x0[1], x0[2], x0[2]], ones(N1))
    x3 = kron([x0[1], x0[2], x0[1], x0[2], x0[1], x0[2], x0[1], x0[2]], ones(N1))
    X1 = [ones(8 * N1) x1 x2 x3]
    return y0, X0, y1, X1
end

N0, N1 = 500, 50
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample(N0, N1);
m = BNP.DGPMNormal(; y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);

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
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0, 10)]
    y0 = X0 * [1; zeros(9); 1] + rand(dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = kron([x0[1], x0[2]], ones(N1))
    X1 = [ones(2 * N1) x1 ones(2 * N1, 9)]
    return y0, X0, y1, X1
end

N0, N1 = 500, 50
rng = MersenneTwister(1);
y0, X0, y1, X1 = simulate_sample(N0, N1);
m = BNP.DGPMNormal(; y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);
a = mean([chainβ[i] .== zeros(11) for i in 1:length(chainβ)])

