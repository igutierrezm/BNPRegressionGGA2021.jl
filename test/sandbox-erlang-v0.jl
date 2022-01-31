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
    z0 = rand(rng, dy, N0)
    ỹ0 = deepcopy(z0)
    ỹ1 = rand(rng, Exponential(10), N0)
    ỹ0 = min.(ỹ0, ỹ1)
    event = ỹ0 .< ỹ1
    y1 = LinRange(0, 7, N1) |> collect
    return dy, event, z0, ỹ0, X0, y1, X1
end

N0, N1 = 500, 50;
rng = MersenneTwister(2);
dy, event, z0, ỹ0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
mean(event)
smpl = BNPRegressionGGA2021.ErlangSampler(; rng, ỹ0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 20000, burnin = 10000);
plot(
    layer(x = y1, y = mean(chainf), Geom.line, color=["bnp"]), 
    layer(x = y1, y = pdf.(dy, y1), Geom.line, color=["true"]),
    layer(x = z0, Geom.histogram(density = true), color=["histogram"])
)
# smpl.φ
# smpl.λ
# unique(smpl.d)
# maximum(smpl.n)
# dy = Erlang(ceil(Int, smpl.φ[3]), 1.0 / smpl.λ[])
# mean(dy)
# std(dy)
# mean(dy)
# std(dy)

#=====================================#
# Example 2 - 1 non-trivial predictor #
#=====================================#

function simulate_sample_01(rng, N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    x0 = [ones(N0) rand(rng, [1, 2], N0)
    X1 = kron(ones(N1), [1 1; 1 2])
    z0 = [X0[i, 2] == 2 ? rand(rng, dy2) : rand(rng, dy1) for i in 1:N0]
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, 2)
    ỹ0 = deepcopy(z0)
    ỹ1 = rand(rng, Exponential(30), N0)
    event = ỹ0 .< ỹ1
    return dy1, dy2, event, z0, ỹ0, X0, y1, X1
end

N0, N1 = 1000, 50;
rng = MersenneTwister(1);
dy1, dy2, event, z0, ỹ0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
mean(event)
grid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = grid, y = pdf.(dy2, grid), Geom.line, color=["kdensity2"]),
    layer(x = grid, y = pdf.(dy1, grid), Geom.line, color=["kdensity1"]),
)
smpl = BNPRegressionGGA2021.ErlangSampler(; ỹ0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]), Geom.line)
plot(x = z0, color = string.(X0[:, 2]), Geom.histogram(density=true, bincount = 50))

#===================================================#
# Example 3 - 1 trivial and 1 non-trivial predictor #
#===================================================#

function simulate_sample_01(rng, N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand(rng, [1, 2], N0, 2)]
    X1 = kron(ones(N1), [1 1 1; 1 2 1; 1 1 2; 1 2 2])
    z0 = [X0[i, 2] == 2 ? rand(rng, dy2) : rand(rng, dy1) for i in 1:N0]
    ỹ0 = deepcopy(z0)
    ỹ1 = rand(rng, Exponential(30), N0)
    event = ỹ0 .< ỹ1    
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, 4)
    return dy1, dy2, event, z0, ỹ0, X0, y1, X1
end

N0, N1 = 1000, 50;
rng = MersenneTwister(2);
dy1, dy2, event, z0, ỹ0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid) ./ (1.0 .- cdf.(dy2, ygrid)), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid) ./ (1.0 .- cdf.(dy1, ygrid)), Geom.line, color=["pdf1"]),
)
smpl = BNPRegressionGGA2021.ErlangSampler(; rng, ỹ0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]) .* " + " .* string.(X1[:, 3]), Geom.line)

#===================================================#
# Example 4 - 1 trivial and 4 non-trivial predictor #
#===================================================#

function simulate_sample_01(rng, N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand(rng, [1, 2], N0, 5)]
    X1 = kron(ones(N1), [1 1 1 ones(1, 3); 1 2 1 ones(1, 3); 1 1 2 ones(1, 3); 1 2 2 ones(1, 3)])
    z0 = [X0[i, 2] == 2 ? rand(rng, dy2) : rand(rng, dy1) for i in 1:N0]
    ỹ0 = deepcopy(z0)
    ỹ1 = rand(rng, Exponential(30), N0)
    event = ỹ0 .< ỹ1    
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, size(X1, 1) ÷ N1)
    return dy1, dy2, event, z0, ỹ0, X0, y1, X1
end

N0, N1 = 1000, 50;
rng = MersenneTwister(2);
dy1, dy2, event, z0, ỹ0, X0, y1, X1 = simulate_sample_01(rng, N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid) ./ (1.0 .- cdf.(dy2, ygrid)), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid) ./ (1.0 .- cdf.(dy1, ygrid)), Geom.line, color=["pdf1"]),
)
smpl = BNPRegressionGGA2021.ErlangSampler(; rng, ỹ0, X0, y1, X1);
chainf, chainβ = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]) .* " + " .* string.(X1[:, 3]), Geom.line)
