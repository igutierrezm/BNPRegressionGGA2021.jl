begin
    using CSV
    using Revise
    using BNPRegressionGGA2021
    using DataFrames
    using DataStructures
    using Distributions
    using FileIO
    using Random
    using RCall
    using Statistics
    using StatsBase
    using LinearAlgebra
    const BNP = BNPRegressionGGA2021
    # import Distributions: pdf, mean, var
    # import Base: rand    
end;

# struct SkewNormal <: ContinuousUnivariateDistribution
#     location::Float64 # ξ
#     scale::Float64 # ω
#     slant::Float64 # α
# end

# function pdf(d::SkewNormal, x::Real)
#     ξ = d.location 
#     ω = d.scale 
#     α = d.slant
#     z = (x - ξ) / ω
#     return (2 / ω) * pdf(Normal(), z) * cdf(Normal(), α * z)
# end

# function rand(rng::AbstractRNG, d::SkewNormal)
#     ξ = d.location
#     ω = d.scale
#     α = d.slant
#     δ = α / √(1 + α^2)
#     z = √(1 - δ^2) * randn(rng) + δ * abs(randn(rng))
#     return ξ + ω * z
# end

# function mean(d::SkewNormal)
#     ξ = d.location 
#     ω = d.scale 
#     α = d.slant
#     b = √(2 / π)
#     δ = α / √(1 + α^2)
#     μz = b * δ
#     return ξ + ω * μz
# end

# function var(d::SkewNormal)
#     ω = d.scale 
#     α = d.slant
#     b = √(2 / π)
#     δ = α / √(1 + α^2)
#     σ2z = 1 - b^2 * δ^2
#     σ2y = ω^2 * σ2z
#     return σ2y
# end

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
    X0d = rand([0, 1], size(X0, 1), 4)
    X0d = ((X0d .- mean(X0d, dims = 1)) ./ std(X0d, dims = 1))
    X0 = [ones(size(X0, 1)) X0 X0d]
    X1 = zeros(0, size(X0, 2))

    # Generate the true (standardized) density over the grid
    f1 = pdf.(dy.(x1), my0 .+ y1 .* sy0) .* sy0

    # Set the mapping 
    mapping = [[1], collect(2:7), [8], [9], [10], [11]]

    # Return the preprocessed results
    return y0, y1, x1, X0, X1, f1, mapping
end;

# Example 1: Continuous predictor (linear regression, normal distribution)
begin
    Random.seed!(1)
    N0, N1, Nrep, Niter = 200, 0, 20, 2
    dy(x) = Normal(x, 1)
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    df = zeros(Bool, Niter, 6)
    for iter in 1:Niter
        y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
        smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
        _, _, chaing = BNP.sample!(smpl; mcmcsize = 10000)
        df[iter, :] .= begin
            chaing |>
            DataStructures.counter |>
            collect |>
            x -> sort(x, by = x -> x[2], rev = true) |>
            x -> getindex(x, 1) |>
            x -> getindex(x, 1)
        end
    end
    filename = "data/simulated-best-normal-bnp-ex-01-2022-03-31.csv"
    CSV.write(filename, DataFrame(df, :auto))    
end;

# Example 2: Continuous predictor (linear regression, skewnormal distribution)
begin
    Random.seed!(1)
    N0, N1, Nrep, Niter = 200, 0, 20, 2
    dy(x) = SkewNormal(2x, 4, 4)
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    df = zeros(Bool, Niter, 6)
    for iter in 1:Niter
        y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
        smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
        _, _, chaing = BNP.sample!(smpl; mcmcsize = 10000)
        df[iter, :] .= begin
            chaing |>
            DataStructures.counter |>
            collect |>
            x -> sort(x, by = x -> x[2], rev = true) |>
            x -> getindex(x, 1) |>
            x -> getindex(x, 1)
        end
    end
    filename = "data/simulated-best-normal-bnp-ex-02-2022-03-31.csv"
    CSV.write(filename, DataFrame(df, :auto))    
end;

# Example 3: Continuous predictor (cubic regression, normal distribution)
begin
    Random.seed!(1)
    N0, N1, Nrep, Niter = 200, 0, 20, 2
    dy(x) = Normal(0.2x^3, 0.5)
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    df = zeros(Bool, Niter, 6)
    for iter in 1:Niter
        y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
        smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
        _, _, chaing = BNP.sample!(smpl; mcmcsize = 10000)
        df[iter, :] .= begin
            chaing |>
            DataStructures.counter |>
            collect |>
            x -> sort(x, by = x -> x[2], rev = true) |>
            x -> getindex(x, 1) |>
            x -> getindex(x, 1)
        end
    end
    filename = "data/simulated-best-normal-bnp-ex-03-2022-03-31.csv"
    CSV.write(filename, DataFrame(df, :auto))    
end;

# Example 4: Continuous predictor (cubic regression, mixture distribution)
begin
    Random.seed!(1)
    N0, N1, Nrep, Niter = 200, 0, 20, 2
    dy(x) = SkewNormal(0.4x^3, 4, 4)
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    df = zeros(Bool, Niter, 6)
    for iter in 1:Niter
        y0, y1, x1, X0, X1, f1, mapping = preprocess(dy, x0raw, x1raw)
        smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, mapping)
        _, _, chaing = BNP.sample!(smpl; mcmcsize = 10000)
        df[iter, :] .= begin
            chaing |>
            DataStructures.counter |>
            collect |>
            x -> sort(x, by = x -> x[2], rev = true) |>
            x -> getindex(x, 1) |>
            x -> getindex(x, 1)
        end
    end
    filename = "data/simulated-best-normal-bnp-ex-04-2022-03-31.csv"
    CSV.write(filename, DataFrame(df, :auto))    
end;
