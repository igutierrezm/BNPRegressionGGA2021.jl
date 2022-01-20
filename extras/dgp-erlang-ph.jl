using Distributions
using LinearAlgebra
using Random

# Define the DGP
function simulate_sample(N0, N1)
    dy1 = Weibull(2, 1);    
    dy2 = Weibull(2, 1.3);
    dy = MixtureModel([Weibull(2, 0.5), Weibull(2, 1.5)], [0.5, 0.5])
    X0 = [ones(N0) rand([0, 1], N0, 4) rand(N0)]
    X1 = kron([ones(2, 4) [1, 0] [0, 0]], ones(N1))
    λ0 = @. exp(-1.0 + 1.0 * X0[:, end - 1] + 1.0 * X0[:, end])
    y0 = λ0 .* rand(dy, N0)
    y1 = LinRange(0, 6, N1) |> collect |> x -> repeat(x, 2)
    c0 = rand(Exponential(10), N0)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

# Not run
Random.seed!(1);
N0, N1 = 500, 0;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
println(mean(z0 .> c0))
println(maximum(z0))
println(mean(z0))