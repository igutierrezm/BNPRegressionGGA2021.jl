using CSV
using DataFrames
using Random

# Load simulate_sample(N0, N1):
include("../extras/dgp-erlang-ph.jl");

# Save 100 simulated samples
begin
    N0, N1, Niter = 500, 2, 100;
    Random.seed!(1);
    df = map(1:Niter) do iter
        _, _, _, c0, y0, X0, _, _ = simulate_sample(N0, N1);
        df = DataFrame(X0, :auto)
        df[!, :iter] .= iter
        df[!, :y0] = y0
        df[!, :c0] = c0
        df
    end;
    df = reduce(vcat, df);
    CSV.write("data/simulated-data-erlang-ph.csv", df);
end