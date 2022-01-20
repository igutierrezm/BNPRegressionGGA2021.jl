using Distributions
using LinearAlgebra
using Random

# Define the DGP
function simulate_sample(N0, N1)
    dy1 = Normal(1, 1);
    dy2 = Normal(1.5, 1);
    X0 = [ones(N0) rand([0, 1], N0, 4) rand(N0)]
    X1 = kron([ones(2, 4) [1, 0] [0, 0]], ones(N1))
    # y0 = [X0[i, end - 1] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    y0 = 1.0 .+ 0.5 * X0[:, end - 1] .+ 0.5 * X0[:, end] .+ randn(N0)
    y1 = LinRange(-3, 3, N1) |> collect |> x -> repeat(x, 2)
    return dy1, dy2, y0, X0, y1, X1
end

# # Not run
# Random.seed!(1);
# N0, N1 = 500, 2;
# dy1, dy2, y0, X0, y1, X1 = simulate_sample(N0, N1);
