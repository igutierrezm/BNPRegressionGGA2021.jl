begin
    using Revise
    using BNPRegressionGGA2021
    using CSV
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

# function preprocess1()
#     # Simulate the covariates
#     N0 = 500
#     D0 = 5
#     x5 = rand(N0)
#     x0min, x0max = extrema(x5)
#     X0_1 = rand([0.0, 1.0], N0, D0 - 1)
#     # X0_2 = rcopy(R"splines::bs($x5, df = 6, Boundary = c($x0min, $x0max))")
#     # X0 = [X0_1 X0_2]    
#     X0 = X0_1
    
#     # Simulate the responses
#     y0 = randn(N0)

#     # Simulate the grid
#     y1 = zeros(0)
#     X1 = zeros(0, size(X0, 2))

#     # Standardise responses and predictors
#     mX0, sX0 = mean_and_std(X0, 1)
#     for col in 1:size(X0, 2)
#         if sX0[col] > 0
#             X0[:, col] = (X0[:, col] .- mX0[col]) ./ sX0[col]
#         end
#     end
#     my0, sy0 = mean_and_std(y0, 1)
#     y0 = (y0 .- my0) ./ sy0

#     # Add a constant term to the design matrices
#     X0 = [ones(size(X0, 1)) X0]
#     X1 = [ones(size(X1, 1)) X1]

#     # Set the mapping 
#     mapping = [[1], collect(2:7), [8], [9], [10], [11]]
#     mapping = [[1], [2], [3], [4], [5]]

#     # Return the preprocessed results
#     return y0, y1, X0, X1, mapping
# end;

# function fit(y0, y1, X0, X1, mapping; mcmcsize = 10000)
#     # Generate some incidental censoring 
#     event0 = ones(Bool, length(y0))

#     # Fit the model
#     smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1, event0, mapping);
#     _, _, chaing = BNP.sample!(smpl; mcmcsize);
#     return chaing
# end;

# # Example 0: Discrete predictor (linear regression, normal distribution)
# begin
#     Random.seed!(1)
#     y0, y1, X0, X1, mapping = preprocess1()
#     chaing = fit(y0, y1, X0, X1, mapping; mcmcsize = 20000)
# end;
# chaing

# Simulate a sample 
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0, 2)]
    y0 = X0 * [ones(2); 0] + rand(dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 4)
    x1 = kron([x0[1], x0[2], x0[1], x0[2]], ones(N1))
    x2 = kron([x0[1], x0[1], x0[2], x0[2]], ones(N1))
    X1 = [ones(4 * N1) x1 x2]
    return y0, X0, y1, X1
end

begin
    Random.seed!(2)
    N0, N1 = 500, 50
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormalDependent(; y0, X0, y1, X1);
    _, _, chaing = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;
mean(chaing)