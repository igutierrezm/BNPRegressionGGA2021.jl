using Revise

using BNPRegressionGGA2021
using Cairo
using CSV
using DataFrames
using Distributions
using Gadfly
using Random

const BNP = BNPRegressionGGA2021

#======================================#
# Replicate the 2nd simulation example #
#======================================#

function simulate_sample(N0)
    dy1 = LogNormal(0, 0.3)
    dy2 = MixtureModel(LogNormal, [(-1.0, 1.0), (0.3, 1.0)], [0.6, 0.4]);
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = ones(1, 6)
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    y1 = zeros(1)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

function simulate_test_results(N0)
    _, _, _, c0, y0, X0, y1, X1 = simulate_sample(N0)
    m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
    _, _, chaing = BNP.sample!(m; mcmcsize = 4000, burnin = 2000)
    return mean(chaing)[2:end]
end

Random.seed!(1);
niter = 100;
results = zeros(100, 5);
for i in 1:niter
    results[i, :] = simulate_test_results(1000);
end
df = DataFrame(results, :auto)
CSV.write("data/simulation-example-02_bnp.csv", df);