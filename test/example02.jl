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

# data
rng = MersenneTwister(1);
N0 = 1000;
N1 = 50;
x0 = rand(rng, N0)
basis = BSplineBasis(4, quantile(x0))
X0 = basismatrix(basis, x0)
# x1 = [ones(N1); 3 * ones(N1)] / 4
x1 = 1 * ones(N1) / 10
X1 = basismatrix(basis, x1)
y0 = zeros(N0);
for i in 1:N0
    dy = MixtureModel(Normal, [(-2.0, 1), (2.0, 1)], [x0[i], 1 - x0[i]])
    y0[i] = rand(rng, dy)
end
y1 = collect(LinRange(extrema(y0)..., N1))

# Fit
data = BNPRegressionGGA2021.Data(; y0, y1, X0, X1);
smpl = BNPRegressionGGA2021.Sampler(data);
chain = BNPRegressionGGA2021.sample(rng, smpl; mcmcsize = 10000);
fh = mean(chain);

# Plot
plot(x = y1, y = fh, Geom.line)
