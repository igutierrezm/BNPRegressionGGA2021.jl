using Revise
using BNPRegressionGGA2021
using CSV
using DataStructures
using DataFrames
using Distributions
using Gadfly
using Random
using Statistics
const BNP = BNPRegressionGGA2021

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

N0, N1, Niter = 100, 2, 1;
df = zeros(Bool, Niter, 6);
Random.seed!(1);
y0, X0, y1, X1 = simulate_sample(N0, N1);
plot(x = y0, Geom.density())
m = BNP.DGPMNormal(; y0, X0, y1, X1);
_, _, chaing = BNP.sample!(m; mcmcsize = 10000, burnin = 5000);
mean(chaing)

# Get the MAP of γ for 100 simulated samples
begin
    N0, N1, Niter = 500, 2, 100;
    df = zeros(Bool, Niter, 6);
    Random.seed!(1);
    for iter in 1:Niter
        y0, X0, y1, X1 = simulate_sample(N0, N1);
        m = BNP.DGPMNormal(; y0, X0, y1, X1);
        _, _, chaing = BNP.sample!(m; mcmcsize = 10000, burnin = 5000);
        df[iter, :] .= begin
            chaing |>
            DataStructures.counter |>
            collect |>
            x -> sort(x, by = x -> x[2], rev = true) |>
            x -> getindex(x, 1) |>
            x -> getindex(x, 1)
        end
        println(iter)
        println(df[iter, :])
    end
    CSV.write("data/simulation-normal-skewnormal-gamma-bnp.csv", DataFrame(df, :auto))
end

# Save the 100 simulated samples
Random.seed!(1);
df = map(1:100) do iter
    y0, X0, _, _ = simulate_sample(N0, N1);
    df = DataFrame(X0, :auto)
    df[!, :y0] = y0
    df
end;
df = reduce(vcat, df);
CSV.write("data/simulation-normal-skewnormal-data.csv", df);
