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

#=================================#
# Example 1 - 1 trivial predictor #
#=================================#

function simulate_sample(N0, N1)
    dy = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = ones(N0, 1)
    X1 = ones(N1, 1)
    y0 = rand(dy, N0)
    c0 = rand(Exponential(20), N0)
    y1 = LinRange(0, 7, N1) |> collect
    z0 = deepcopy(y0)
    return dy, z0, c0, y0, X0, y1, X1
end;

Random.seed!(1);
N0, N1 = 500, 50;
dy, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);

m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainS, chainβ, chaing = BNP.sample!(m; mcmcsize = 2000, burnin = 1000);
plot(
    layer(x = y1, y = mean(chainf), Geom.line, color=["bnp"]), 
    layer(x = y1, y = pdf.(dy, y1), Geom.line, color=["true"]),
)
mean(m.event)

#=====================================#
# Example 2 - 1 non-trivial predictor #
#=====================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand([1, 2], N0)]
    X1 = kron(ones(N1), [1 1; 1 2])
    y0 = [X0[i, 2] == 2 ? rand(dy2) : rand(dy1) for i in 1:N0]
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, 2)
    c0 = rand(Exponential(10), N0)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
grid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = grid, y = pdf.(dy2, grid), Geom.line, color=["kdensity2"]),
    layer(x = grid, y = pdf.(dy1, grid), Geom.line, color=["kdensity1"]),
)
plot(
    layer(x = grid, y = pdf.(dy2, grid) ./ (1.0 .- cdf.(dy2, grid)), Geom.line, color=["pdf2"]),
    layer(x = grid, y = pdf.(dy1, grid) ./ (1.0 .- cdf.(dy1, grid)), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]), Geom.line)
mean(m.event)

#===================================================#
# Example 3 - 1 trivial and 1 non-trivial predictor #
#===================================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand([1, 2], N0, 2)]
    X1 = kron(ones(N1), [1 1 1; 1 2 1; 1 1 2; 1 2 2])
    y0 = [X0[i, 2] == 2 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    z0 = deepcopy(y0)
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, 4)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 20000, burnin = 10000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, 2]), Geom.line)
mean(m.event)

#===================================================#
# Example 4 - 1 trivial and 4 non-trivial predictor #
#===================================================#

function simulate_sample(N0, N1)
    dy1 = LogNormal(0, 0.5)
    dy2 = MixtureModel(LogNormal, [(0.0, 1.0), (1.0, 0.25)], [0.6, 0.4]);
    X0 = [ones(N0) rand([0, 1], N0, 5)]
    X1 = kron(ones(N1), [ones(2, 5) [1, 0]])
    y0 = [X0[i, end] == 1 ? rand(dy2) : rand(dy1) for i in 1:N0]
    c0 = rand(Exponential(10), N0)
    y1 = LinRange(0, 6, N1) |> collect  |> x -> repeat(x, size(X1, 1) ÷ N1)
    z0 = deepcopy(y0)
    return dy1, dy2, z0, c0, y0, X0, y1, X1
end

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);

# Save the data for replication purposes
data = DataFrame(X0, :auto)
data[!, :c0] = c0
data[!, :y0] = y0
CSV.write("data/example_survival.csv", data)

# Plot the true density / survival / hazard functions
ygrid = LinRange(0, 6, N1) |> collect;
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["pdf2"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["pdf1"]),
)
m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainβ = BNP.sample!(m; mcmcsize = 40000, burnin = 20000);
plot(x = y1, y = mean(chainf), color = string.(X1[:, end]), Geom.line)
mean(m.event)

#====================================================#
# Example 5 - 1 non-trivial and 4 trivial predictors #
#====================================================#

function simulate_sample(N0, N1)
    # dy1 = LogNormal(0, 0.3)
    # dy2 = MixtureModel(LogNormal, [(-1.0, 1.0), (0.3, 1.0)], [0.6, 0.4]);
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

Random.seed!(1);
N0, N1 = 500, 50;
dy1, dy2, z0, c0, y0, X0, y1, X1 = simulate_sample(N0, N1);
ygrid = LinRange(0, 6, N1) |> collect;

# Densities
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid), Geom.line, color=["x = 1"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid), Geom.line, color=["x = 0"]),
)

# Hazard curves
plot(
    layer(x = ygrid, y = pdf.(dy2, ygrid) ./ (1.0 .- cdf.(dy2, ygrid)), Geom.line, color=["x = 1"]),
    layer(x = ygrid, y = pdf.(dy1, ygrid) ./ (1.0 .- cdf.(dy1, ygrid)), Geom.line, color=["x = 0"]),
)

m = BNP.DGPMErlang(; c0, y0, X0, y1, X1);
chainf, chainS, chainβ, chaing = BNP.sample!(m; mcmcsize = 20000, burnin = 10000);

# Show the marginal "significance" of each variable
mean(chaing)

# Generate a dataframe for plotting the posterior survival curves
chainS_mat = hcat(chainS...) 
chainS_v = [chainS_mat[i, :] for i in 1:size(chainS_mat, 1)]
plot_data = DataFrame(
    z = y1,
    x5 = X1[:, end], 
    St = [1 - cdf(X1[i, end] == 1 ? dy2 : dy1, y1[i]) for i in 1:size(X1, 1)],
    Sh = mean.(chainS_v),
    lb = quantile.(chainS_v, 0.025),
    ub = quantile.(chainS_v, 0.975),

)
CSV.write("data/crossing-survival-curves-fitted.csv", plot_data)


# plot(
#     plot(x = y1, y = mean.(chainS_v), color = Symbol.(X1[:, end]), Geom.line)
#     plot(x = y1, y = quantile.(chainS_v, 0.05), color = Symbol.(X1[:, end]), Geom.line)
#     plot(x = y1, y = quantile.(chainS_v, 0.95), color = Symbol.(X1[:, end]), Geom.line)    
# )
# plot(x = y1, y = mean.(chainS_v), color = Symbol.(X1[:, end]), Geom.line)
# plot(x = y1, y = quantile.(chainS_v, 0.05), color = Symbol.(X1[:, end]), Geom.line)
# plot(x = y1, y = quantile.(chainS_v, 0.95), color = Symbol.(X1[:, end]), Geom.line)
# plot(x = z0, color = string.(X0[:, end]), Geom.density)
# a = mean(chaing)
# mean(m.event)
most_common_γ = 
    chaing |>
    DataStructures.counter |>
    collect |>
    x -> sort(x, by = x -> x[2], rev = true)

