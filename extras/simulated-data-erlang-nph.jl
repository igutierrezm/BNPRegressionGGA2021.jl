using Revise
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random

# Define the DGP
function simulate_sample(N0, N1)
    dy1 = LogNormal(0.3, 0.7);
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (0.3, 1.2)], [0.6, 0.4]);    
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = kron([ones(2, 5) [1, 0]], ones(N1))
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    y1 = LinRange(0, 6, N1) |> collect |> x -> repeat(x, size(X1, 1) ÷ N1)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

# Save the 100 simulated samples
N0, N1, Niter = 500, 2, 100;
Random.seed!(1);
df = map(1:Niter) do iter
    _, _, _, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
    df = DataFrame(X0, :auto)
    df[!, :iter] .= iter
    df[!, :y0] = y0
    df[!, :c0] = c0
    df
end;
df = reduce(vcat, df);
CSV.write("data/simulated-data-erlang-nph.csv", df);
