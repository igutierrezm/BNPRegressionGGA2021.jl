using BNPRegressionGGA2021
using CSV
using DataStructures
using DataFrames
using Random
const BNP = BNPRegressionGGA2021

# Get the simulated data
data = CSV.read("data/simulated-data-beta-betamix.csv", DataFrame)

# Get the best γ for 100 simulated samples (using the MAP estimator)
begin
    Niter = 100
    Random.seed!(1);
    df = zeros(Bool, Niter, 6)
    gdata = groupby(data, :iter)
    for iter in 1:Niter
        y0 = gdata[iter][!, r"y"] |> Matrix |> vec
        X0 = gdata[iter][!, r"x"] |> Matrix
        y1 = zeros(0)
        X1 = zeros(0, size(X0, 2))
        m = BNP.DGPMBeta(; y0, X0, y1, X1);
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
    end
    filename = "data/simulated-best-gamma-beta-betamix-bnp.csv"
    CSV.write(filename, DataFrame(df, :auto))
end
df