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
    X1 <- predict(X0, $x1)  
    """
    @rget X0 X1

    # Standardise responses and predictors
    mX0, sX0 = mean_and_std(X0, 1)
    for col in 1:size(X0, 2)
        if sX0[col] > 0
            X0[:, col] = (X0[:, col] .- mX0[col]) ./ sX0[col]
            X1[:, col] = (X1[:, col] .- mX0[col]) ./ sX0[col]
        end
    end
    my0, sy0 = mean_and_std(y0, 1)
    y0 = (y0 .- my0) ./ sy0

    # Add a constant term to the design matrices
    X0 = [ones(size(X0, 1)) X0]
    X1 = [ones(size(X1, 1)) X1]

    # Generate the true (standardized) density over the grid
    f1 = pdf.(dy.(x1), my0 .+ y1 .* sy0) .* sy0

    # Return the preprocessed results
    return y0, y1, x1, X0, X1, f1
end;

function fit(y0, y1, x1, X0, X1; mcmcsize = 40000)
    # Fit the model
    smpl = BNP.DGSBPNormalDependent(; y0, X0, y1, X1);
    fhchain, _ = BNP.sample!(smpl; mcmcsize);
    fh = mean(fhchain)

    # Organize the results as a DataFrame
    df = DataFrame(; y1, x1, f1, fh)
    return df, smpl
end;

function plot(df; figname)
    R"""
    for (var in c("fh", "f1")) {
        xv <- sort(unique($df[["x1"]]))
        yv <- sort(unique($df[["y1"]]))
        zv <- $df |>
            dplyr::select(x1, y1, !!var) |>
            tidyr::pivot_wider(values_from = !!var, names_from = y1) |>
            dplyr::select(-x1) |>
            as.matrix()
        png(paste0($figname, "-", var, ".png"))
        fig <- persp(
            x = xv, y = yv, z = zv, 
            phi = 90, theta = 45, xlab = "x", ylab = "y", zlab = var
        )  
        dev.off()
    }
    """
end;

# Example 0: Discrete predictor (linear regression, normal distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 2, 20
    x0raw = LinRange(-1, 1, N0)
    x1raw = LinRange(-1, 1, N1)
    dy(x) = Normal(x, 1)
    y0, y1, x1, X0, X1, f1 = preprocess(dy, x0raw, x1raw)
    df, smpl = fit(y0, y1, x1, X0, X1; mcmcsize = 40000)
    plot(df; figname = "figures/encargo-11-03-2020-ex-0")
end;

# Example 1: Continuous predictor (linear regression, normal distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 50, 20
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ?? Nrep), Nrep)
    dy(x) = Normal(x, 1)
    y0, y1, x1, X0, X1, f1 = preprocess(dy, x0raw, x1raw)
    df, smpl = fit(y0, y1, x1, X0, X1; mcmcsize = 40000)
    plot(df; figname = "figures/encargo-11-03-2020-ex-1")
end;

# DGP 2: Continuous predictor (linear regression, mixture distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 50, 20
    x1raw = LinRange(-1, 1, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ?? Nrep), Nrep)
    dy(x) = MixtureModel(Normal, [(0.2x - 1, 0.5), (0.2x + 1, 0.5)])
    y0, y1, x1, X0, X1, f1 = preprocess(dy, x0raw, x1raw)
    df, smpl = fit(y0, y1, x1, X0, X1; mcmcsize = 40000)
    plot(df; figname = "figures/encargo-11-03-2020-ex-2")
end;

# Example 3: Continuous predictor (cubic regression, normal distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 50, 20
    x1raw = LinRange(-1.5, 1.5, N1)
    x0raw = repeat(LinRange(-2, 2, N0 ?? Nrep), Nrep)
    dy(x) = Normal(0.2x^3, 0.5)
    y0, y1, x1, X0, X1, f1 = preprocess(dy, x0raw, x1raw)
    df, smpl = fit(y0, y1, x1, X0, X1; mcmcsize = 40000)
    plot(df; figname = "figures/encargo-11-03-2020-ex-3")
end;

# Example 4: Continuous predictor (cubic regression, mixture distribution)
begin
    Random.seed!(3)
    N0, N1, Nrep = 500, 50, 20
    x1raw = LinRange(-2, 2, N1)
    x0raw = repeat(LinRange(-2.5, 2.5, N0 ?? Nrep), Nrep)
    # dy(x) = MixtureModel(Normal, [(x^3 - 1, 0.6), (x^3 + 1, 0.6)])
    dy(x) = MixtureModel(Normal, [(0.2x^3 - 1.5, 0.5), (0.2x^3 + 1.5, 0.5)])
    y0, y1, x1, X0, X1, f1 = preprocess(dy, x0raw, x1raw)
    df, smpl = fit(y0, y1, x1, X0, X1; mcmcsize = 40000)
    plot(df; figname = "figures/encargo-11-03-2020-ex-4")
end;
