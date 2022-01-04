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
    dy = MixtureModel(Beta, [(1.0, 5.0), (10.0, 2.0)], [0.2, 0.8])
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
chainf, chainβ = BNP.sample!(m; mcmcsize = 10000, burnin = 5000);
plot(
    layer(x = y1, y = mean(chainf), Geom.line, color = ["bnp"]), 
    layer(x = y1, y = pdf.(dy, y1), Geom.line, color = ["true"]),
    layer(x = y0, Geom.histogram(density = true, bincount = 50), color = ["hist"]),
)

#===================================================#
# Example 2 - 4 trivial and 1 non-trivial predictor #
#===================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    dy1 = MixtureModel(Beta, [(1.0, 5.0), (10.0, 2.0)], [0.2, 0.8])
    dy2 = MixtureModel(Beta, [(1.0, 5.0), (10.0, 2.0)], [0.8, 0.2])
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = kron(ones(N1), [ones(2, 5) [1, 0]])
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    y1 = LinRange(0, 1, N1) |> collect |> x -> repeat(x, size(X1, 1) ÷ N1)
    return dy1, dy2, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, y0, X0, y1, X1 = simulate_sample(N0, N1);
ygrid = LinRange(0, 1, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMBeta(; y0, X0, y1, X1);
chainf, chainβ, chaing = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, end]), Geom.line)
