using Revise
using CSV
using DataFrames
using Distributions
using Gadfly
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
    y0 = 0.8 * X0[:, end] + ϵ
    y1 = LinRange(-3, 3, N1) |> collect |> x -> repeat(x, size(X1, 1) ÷ N1)
    return y0, X0, y1, X1
end

# Save the 100 simulated samples
N0, N1, Niter = 500, 2, 100;
Random.seed!(1);
df = map(1:Niter) do iter
    y0, X0, _, _ = simulate_sample(N0, N1);
    df = DataFrame(X0, :auto)
    df[!, :y0] = y0
    df
end;
df = reduce(vcat, df);
CSV.write("data/simulated-data-normal-skewnormal.csv", df);
