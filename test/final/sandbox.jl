begin
    using CSV
    using Revise
    using BNPRegressionGGA2021
    using DataFrames
    using DataStructures
    using Distributions
    using Random
    using RCall
    using Statistics
    using StatsBase
    using LinearAlgebra
    const BNP = BNPRegressionGGA2021
end;

R"""
library(ISLR)
library(leaps)
data("swiss")
# Inspect the data
sample_n(swiss, 3)
fit <- regsubsets(Fertility~., data = swiss, nvmax = 5)
fitsum <- summary(models, matrix.logical = TRUE)
kstar <- which.max(fitsum$bic)
best_gamma <- fitsum$which[kstar, ]
"""

# Helper functions -------------------------------------------------------------

function run_experiment_bnp(dy; N0, Nrep, Niter, id, filename)
    Random.seed!(1) # seed
    samples = [generate_sample(dy; N0, Nrep) for _ in 1:Niter] # data
    gammas = [get_gamma_bnp(samples[id]) for id in 1:Niter] # map gamma
    reduced_gammas = reduce_gammas(gammas) # reduced df
    reduced_gammas[!, :method] .= "bnp"
    reduced_gammas[!, :id] .= id
    reduced_gammas[!, :N0] .= N0
    CSV.write(filename, reduced_gammas) # csv file
end

function run_experiment_freq(dy; N0, Nrep, Niter, id, filename)
    Random.seed!(1) # seed
    samples = [generate_sample(dy; N0, Nrep) for _ in 1:Niter] # data
    gammas = [get_gamma_freq(samples[id]) for id in 1:Niter] # map gamma
    reduced_gammas = reduce_gammas(gammas) # reduced df
    reduced_gammas[!, :method] .= "freq"
    reduced_gammas[!, :id] .= id
    reduced_gammas[!, :N0] .= N0
    CSV.write(filename, reduced_gammas) # csv file
end

function generate_sample(dy; N0 = 50, Nrep = 10)
    # Simulate the data
    x0 = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    y0 = @. rand(dy(x0))

    # Generate the grid
    N1 = 0
    x1 = LinRange(-1, 1, N1)
    y1 = LinRange(-3, 3, 50) |> collect

    # Expand the grid
    grid = rcopy(R"""expand.grid(y1 = $y1, x1 = $x1)""")
    y1 = grid[!, :y1]
    x1 = grid[!, :x1]

    # Expand x0-x1 using splines
    R"""
    x0min <- min($x0)
    x0max <- max($x0)
    X0 <- splines::bs($x0, df = 6, Boundary.knots = c(x0min, x0max))
    """
    @rget X0

    # Standardise responses and predictors
    mX0, sX0 = mean_and_std(X0, 1)
    for col in 1:size(X0, 2)
        if sX0[col] > 0
            X0[:, col] = (X0[:, col] .- mX0[col]) ./ sX0[col]
        end
    end
    my0, sy0 = mean_and_std(y0, 1)
    y0 = (y0 .- my0) ./ sy0

    # Add a constant term to the design matrices
    X0d = rand([0, 1], size(X0, 1), 4)
    X0d = ((X0d .- mean(X0d, dims = 1)) ./ std(X0d, dims = 1))
    X0 = [ones(size(X0, 1)) X0 X0d]
    X1 = zeros(0, size(X0, 2))

    # Generate the true (standardized) density over the grid
    f1 = pdf.(dy.(x1), my0 .+ y1 .* sy0) .* sy0

    # Set the mapping 
    mapping = [[1], collect(2:7), [8], [9], [10], [11]]

    # Return the preprocessed results
    return y0, x0, y1, x1, X0, X1, f1, mapping
end;

# Get the MAP estimator of gamma using BNP
function get_gamma_bnp(sample)
    y0, x0, y1, x1, X0, X1, f1, mapping = sample
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 4000)
    gamma = begin
        chaing |>
        DataStructures.counter |>
        collect |>
        x -> sort(x, by = x -> x[2], rev = true) |>
        x -> getindex(x, 1) |>
        x -> getindex(x, 1)
    end
    return gamma
end

# Estimate gamma using a frequentist alternative
function get_gamma_freq(sample)
    y0, x0, y1, x1, X0, X1, f1, mapping = sample
    R"""
    library(dplyr)
    data = data.frame(
        y = $y0,
        x2 = $x0, 
        x3 = $X0[, 8],
        x4 = $X0[, 9],
        x5 = $X0[, 10],
        x6 = $X0[, 11]
    )
    fitted_model <- leaps::regsubsets(y ~ ., data = data, nvmax = 5)
    fitsum <- summary(fitted_model, matrix.logical = TRUE)
    kstar <- which.min(fitsum$bic)
    best_gamma <- fitsum$which[kstar, ]
    """
    @rget best_gamma
    return best_gamma
end

# Reduce the MAP estimators after their estimation
function reduce_gammas(gammas)
    names = "g" .* string.(1:6)
    data = reduce(vcat, gammas')
    DataFrame(data, names)
end

# Experiment 1, N0 = 50 (bnp)
begin 
    dy(x) = Normal(x, 1)
    filename = "data/final/gamma-normal-bnp-50.csv"
    run_experiment_bnp(dy; N0 = 50, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 50 (freq)
begin 
    dy(x) = Normal(x, 1)
    filename = "data/final/gamma-normal-freq-50.csv"
    run_experiment_freq(dy; N0 = 50, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 100 (bnp)
begin 
    dy(x) = Normal(x, 1)
    filename = "data/final/gamma-normal-bnp-100.csv"
    run_experiment_bnp(dy; N0 = 100, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 100 (freq)
begin 
    dy(x) = Normal(x, 1)
    filename = "data/final/gamma-normal-freq-100.csv"
    run_experiment_freq(dy; N0 = 100, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 200 (bnp)
begin 
    dy(x) = Normal(x, 1)
    filename = "data/final/gamma-normal-bnp-200.csv"
    run_experiment_bnp(dy; N0 = 200, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 200 (freq)
begin 
    dy(x) = Normal(x, 1)
    filename = "data/final/gamma-normal-freq-200.csv"
    run_experiment_freq(dy; N0 = 200, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 2, N0 = 50 (bnp)
begin 
    dy(x) = SkewNormal(x, 1.5, 4)
    filename = "data/final/gamma-skewnormal-bnp-50.csv"
    run_experiment_bnp(dy; N0 = 50, Nrep = 5, Niter = 100, id = 2, filename)
end 

# Experiment 2, N0 = 50 (freq)
begin 
    dy(x) = SkewNormal(x, 1.5, 4)
    filename = "data/final/gamma-skewnormal-freq-50.csv"
    run_experiment_freq(dy; N0 = 50, Nrep = 5, Niter = 100, id = 2, filename)
end 

# Experiment 2, N0 = 100 (bnp)
begin 
    dy(x) = SkewNormal(x, 1.5, 4)
    filename = "data/final/gamma-skewnormal-bnp-100.csv"
    run_experiment_bnp(dy; N0 = 100, Nrep = 5, Niter = 100, id = 2, filename)
end 

# Experiment 2, N0 = 100 (freq)
begin 
    dy(x) = SkewNormal(x, 1.5, 4)
    filename = "data/final/gamma-skewnormal-freq-100.csv"
    run_experiment_freq(dy; N0 = 100, Nrep = 5, Niter = 100, id = 2, filename)
end 

# Experiment 2, N0 = 200 (bnp)
begin 
    dy(x) = SkewNormal(x, 1.5, 4)
    filename = "data/final/gamma-skewnormal-bnp-200.csv"
    run_experiment_bnp(dy; N0 = 200, Nrep = 5, Niter = 100, id = 2, filename)
end 

# Experiment 2, N0 = 200 (bnp)
begin 
    dy(x) = SkewNormal(x, 1.5, 4)
    filename = "data/final/gamma-skewnormal-freq-200.csv"
    run_experiment_freq(dy; N0 = 200, Nrep = 5, Niter = 100, id = 2, filename)
end 
