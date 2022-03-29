begin
    using Revise
    using BNPRegressionGGA2021
    using DataFrames
    using Distributions
    using FileIO
    using Random
    using RCall
    using Statistics
    using StatsBase
    using LinearAlgebra
    const BNP = BNPRegressionGGA2021
end;

function preprocess(dy, x0, x1)
    # Simulate the responses
    y0 = @. rand(dy(x0))
    y1 = LinRange(-3, 3, 50) |> collect

    # Expand the grid
    grid = rcopy(R"""expand.grid(y1 = $y1, x1 = $x1)""")
    y1 = grid[!, :y1]
    x1 = grid[!, :x1]

    # Expand x0-x1 using splines
    x0min, x0max = extrema(x0)
    R"""
    X0 <- splines::bs($x0, df = 6, Boundary.knots = c($x0min, $x0max))
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
    X0 = [ones(size(X0, 1)) X0 rand([0, 1], size(X0, 1), 4)]
    X1 = zeros(0, size(X0, 2))

    # Generate the true (standardized) density over the grid
    f1 = pdf.(dy.(x1), my0 .+ y1 .* sy0) .* sy0

    # Set the mapping 
    mapping = [[1], collect(2:7), [8], [9], [10], [11]]

    # Return the preprocessed results
    return y0, y1, x1, X0, X1, f1, mapping
end;

# Example 0: Discrete predictor (linear regression, normal distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 0, 20
    x0raw = LinRange(-1, 1, N0)
    x1raw = LinRange(-1, 1, N1)
    dy(x) = Normal(x, 1)
    y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 40000)
    df = DataFrame(j = 1:length(mean(chaing)), value = mean(chaing))
    CSV.write("data/mean_chaing0.csv", df)
end;

# Example 1: Continuous predictor (linear regression, normal distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 50, 20
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    dy(x) = Normal(x, 1)
    y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 40000)
    df = DataFrame(j = 1:length(mean(chaing)), value = mean(chaing))
    CSV.write("data/mean_chaing1.csv", df)
end;

# Example 2: Continuous predictor (linear regression, mixture distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 0, 20
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    dy(x) = MixtureModel(Normal, [(0.2x - 1, 0.5), (0.2x + 1, 0.5)])
    y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 40000)
    df = DataFrame(j = 1:length(mean(chaing)), value = mean(chaing))
    CSV.write("data/mean_chaing2.csv", df)
end;

# Example 3: Continuous predictor (cubic regression, normal distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 0, 20
    x1raw = LinRange(-1.5, 1.5, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    dy(x) = Normal(0.2x^3, 0.5)
    y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 40000)
    df = DataFrame(j = 1:length(mean(chaing)), value = mean(chaing))
    CSV.write("data/mean_chaing3.csv", df)
end;

# Example 4: Continuous predictor (cubic regression, mixture distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 0, 20
    x1raw = LinRange(-2, 2, N1)
    x0raw = repeat(LinRange(-2.5, 2.5, N0 ÷ Nrep), Nrep)
    dy(x) = MixtureModel(Normal, [(0.2x^3 - 1.5, 0.5), (0.2x^3 + 1.5, 0.5)])
    y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 40000)
    df = DataFrame(j = 1:length(mean(chaing)), value = mean(chaing))
    CSV.write("data/mean_chaing4.csv", df)
end;
