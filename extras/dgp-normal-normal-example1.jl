using BNPRegressionGGA2021
using CSV
using DataFrames
using Random
const BNP = BNPRegressionGGA2021

# Simulate the data
begin
    include("dgp-normal-normal1.jl");
    Random.seed!(1);
    N0, N1 = 6000, 50;
    dy, y0, x0, X0, y1, x1, X1 = simulate_sample(N0, N1);
    m0, o0 = mean(y0), std(y0)
    y0 = (y0 .- m0) ./ o0
end; 
# did not work with 1000
# did not work with 2000
# did not work with 3000
# did not work with 4000
# worked better with 5000
# worked better with 6000
# worked with 10000

# Fit the model
begin
    update_γ = false;
    mapping = [[1], collect(2:size(X0, 2))];
    m = BNP.DGSBPNormal(; y0, X0, y1, X1, mapping, update_γ);
    chainf, chainβ, chaing = BNP.sample!(m; mcmcsize = 20000, burnin = 10000);
end; 

# Save the results as a CSV file
begin
    df = DataFrame(
        y = y1, 
        x = x1, 
        fh = mean(chainf), 
        f0 = pdf.(dy.(x1), m0 .+ o0 .* y1) .* o0
    )
    filename = "data/dgp-normal-normal-example1.csv"
    CSV.write(filename, df)
end
