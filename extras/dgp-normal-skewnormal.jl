using Distributions
using LinearAlgebra
using Random

# Define the DGP
function simulate_sample(N0, N1)
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = kron([ones(2, 5) [1, 0]], ones(N1))
    α = 4
    δ = α / √(1 + α^2)
    ω = 1
    ξ = - ω * δ * √(2 / π)
    ϵ = √(1 - δ^2) .* randn(N0) + δ .* abs.(randn(N0)) .+ ξ
    y0 = 0.3 * X0[:, end] + ϵ
    y1 = LinRange(-3, 3, N1) |> collect |> x -> repeat(x, 2)
    return y0, X0, y1, X1, ϵ
end

# # Not run
# Random.seed!(1);
# N0, N1 = 500, 0;
# y0, X0, y1, X1, ϵ = simulate_sample(N0, N1);
# mean(ϵ)