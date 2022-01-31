using BNPRegressionGGA2021
using CSV
using DataFrames
using Random
const BNP = BNPRegressionGGA2021

# Simulate the data
begin
    include("dgp-normal-normal.jl");
    Random.seed!(1);
    N0, N1 = 500, 50;
    dy, y0, x0, X0, y1, x1, X1 = simulate_sample(N0, N1);
    y0 = (y0 .- mean(y0)) ./ std(y0)
end;

# Fit the model
begin
    mapping = [[1], collect(2:size(X0, 2))];
    m = BNP.DGSBPNormal(; y0, X0, y1, X1, mapping);
    chainf, chainβ, chaing = BNP.sample!(m; mcmcsize = 25000, burnin = 5000);
end;

# Save the results as a CSV file
begin
    df = DataFrame(
        y = y1, 
        x = x1, 
        fh = mean(chainf), 
        # f0 = pdf.(dy.(x1), y1)
    )
    filename = "data/dgp-normal-normal-example1.csv"
    CSV.write(filename, df)
end
