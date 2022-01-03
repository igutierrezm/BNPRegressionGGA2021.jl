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
    dy = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(dy, N0)
    c0 = rand(Exponential(20), N0)
    y1 = LinRange(0, 7, N1) |> collect
    z0 = deepcopy(y0)
    return dy, z0, c0, y0, X0, y1, X1
end;

Random.seed!(1);
N0, N1 = 500, 50;
dy, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);
plot(
    layer(x = y1, y = mean(chainf), Geom.line, color=["bnp"]), 
    layer(x = y1, y = pdf.(dy, y1), Geom.line, color=["true"]),
)
mean(m.event)

#=====================================#
# Example 2 - 1 non-trivial predictor #
#=====================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand([1, 2], N0)]
    X1 = kron(ones(N1), [1 1; 1 2])
    y0 = [X0[i, 2] == 2 ? rand(dy2) : rand(dy1) for i in 1:N0]
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, 2)
    c0 = rand(Exponential(10), N0)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
grid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = grid, y = pdf.(dy2, grid), Geom.line, color=["kdensity2"]),
    layer(x = grid, y = pdf.(dy1, grid), Geom.line, color=["kdensity1"]),
)
plot(
    layer(x = grid, y = pdf.(dy2, grid) ./ (1.0 .- cdf.(dy2, grid)), Geom.line, color=["pdf2"]),
    layer(x = grid, y = pdf.(dy1, grid) ./ (1.0 .- cdf.(dy1, grid)), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]), Geom.line)
mean(m.event)

#===================================================#
# Example 3 - 1 trivial and 1 non-trivial predictor #
#===================================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand([1, 2], N0, 2)]
    X1 = kron(ones(N1), [1 1 1; 1 2 1; 1 1 2; 1 2 2])
    y0 = [X0[i, 2] == 2 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    z0 = deepcopy(y0)
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, 4)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]), Geom.line)
mean(m.event)

#===================================================#
# Example 4 - 1 trivial and 4 non-trivial predictor #
#===================================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = kron(ones(N1), [ones(2, 5) [1, 0]])
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, size(X1, 1) ÷ N1)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 40000, burnin = 20000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, end]), Geom.line)
mean(m.event)

#===================================================#
# Example 5 - 1 trivial and 4 non-trivial predictor #
#===================================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.3)
    dy2 = MixtureModel(LogNormal, [(-1.0, 1.0), (0.3, 1.0)], [0.6, 0.4]);
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = kron(ones(N1), [ones(2, 5) [1, 0]])
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    y1 = LinRange(0, 6, N1) |> collect |> x -> repeat(x, size(X1, 1) ÷ N1)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid) ./ (1.0 .- cdf.(dy2, ygrid)), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid) ./ (1.0 .- cdf.(dy1, ygrid)), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 40000, burnin = 20000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, end]), Geom.line)
plot(x = z0, color = string.(X0[:, end]), Geom.histogram(density = true))
mean([chainβ[i] .== zeros(length(chainβ[1])) for i in 1:length(chainβ)])
mean(m.event)
