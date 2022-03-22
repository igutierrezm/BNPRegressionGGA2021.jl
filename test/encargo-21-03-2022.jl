begin
    using Revise
    using BNPRegressionGGA2021
    using DataFrames
    using Distributions
    using Random
    using RCall
    using Statistics
    using StatsBase
    using LinearAlgebra
    const BNP = BNPRegressionGGA2021
    import Distributions: pdf
    import Base: rand
end;

function preprocess1()
    # Simulate the covariates
    N0 = 500
    D0 = 5
    x5 = rand(N0)
    x0min, x0max = extrema(x5)
    X0_1 = rand([0.0, 1.0], N0, D0 - 1)
    X0_2 = rcopy(R"splines::bs($x5, df = 6, Boundary = c($x0min, $x0max))")
    X0 = [X0_1 X0_2]    
    
    # Simulate the responses
    y0 = randn(N0)

    # Simulate the grid
    y1 = zeros(0)
    X1 = zeros(0, size(X0, 2))

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
    X0 = [ones(size(X0, 1)) X0]
    X1 = [ones(size(X1, 1)) X1]

    # Set the mapping 
    mapping = [[1], collect(2:7), [8], [9], [10], [11]]

    # Return the preprocessed results
    return y0, y1, X0, X1, mapping
end;

function fit(y0, y1, x1, X0, X1, mapping; mcmcsize = 10000)
    # Generate some incidental censoring 
    event0 = ones(Bool, length(y0))

    # Fit the model
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, event0, mapping);
    _, _, chaing = BNP.sample!(smpl; mcmcsize);
    return chaing
end;

# Example 0: Discrete predictor (linear regression, normal distribution)
begin
    Random.seed!(1)
    y0, y1, X0, X1, mapping = preprocess1()
    chaing = fit(y0, y1, x1, X0, X1, mapping; mcmcsize = 20000)
end;
chaing