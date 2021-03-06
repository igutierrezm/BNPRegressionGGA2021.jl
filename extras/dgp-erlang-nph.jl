using Distributions
using LinearAlgebra
using Random

# Define the DGP
function simulate_sample(N0, N1)
    dy1 = LogNormal(0.3, 0.7);
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (0.3, 1.2)], [0.6, 0.4]);    
    X0 = [ones(N0) rand([0, 1], N0, 4) rand(N0)]
    X1 = kron([ones(2, 4) [1, 0] [0, 0]], ones(N1))
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    y1 = LinRange(0, 6, N1) |> collect |> x -> repeat(x, 2)
    c0 = rand(Exponential(12), N0)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

# Not run
Random.seed!(1);
N0, N1 = 500, 0;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
mean(z0 .> c0)