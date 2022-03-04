begin
    using Revise
    using BayesNegativeBinomial
    using BNPRegressionGGA2021
    using DataFrames
    using Distributions
    using FileIO
    using Gadfly
    using Random
    using RCall
    using Statistics
    using StatsBase
    using LinearAlgebra
    const BNP = BNPRegressionGGA2021
end

function foo(dy, x0, x1; filename, discrete = true, mcmcsize = 10000)
    # Simulate the responses
    y0 = @. rand(dy(x0))
    y1 = LinRange(-3, 3, 50) |> collect

    # Expand the grid
    R"""
    grid <- expand.grid(y1 = $y1, x1 = $x1)
    y1 <- grid$y1
    x1 <- grid$x1
    """
    @rget y1 x1

    # Expand x0-x1 using splines
    if discrete == true
        X0 = x0[:, :]
        X1 = x1[:, :]
    else
        x0min, x0max = extrema(x0)
        R"""
        X0 <- splines::bs($x0, df = 6, Boundary.knots = c($x0min, $x0max))
        X1 <- predict(X0, $x1)  
        """
        @rget X0 X1    
    end

    # Standardise responses and predictors
    if discrete == false
        mX0, sX0 = mean_and_std(X0, 1)
        for col in 1:size(X0, 2)
            if sX0[col] > 0
                X0[:, col] = (X0[:, col] .- mX0[col]) ./ sX0[col]
                X1[:, col] = (X1[:, col] .- mX0[col]) ./ sX0[col]
            end
        end
    end
    my0, sy0 = mean_and_std(y0, 1)
    y0 = (y0 .- my0) ./ sy0

    # Add a constant term to the design matrices
    X0 = [ones(size(X0, 1)) X0]
    X1 = [ones(size(X1, 1)) X1]

    # Generate the true (standardized) density over the grid
    f1 = pdf.(dy.(x1), my0 .+ y1 .* sy0) .* sy0

    # Fit the model
    smpl = BNP.DGSBPNormal(; y0, X0, y1, X1);
    fhchain, _ = BNP.sample!(smpl; mcmcsize);
    fh = mean(fhchain)

    # Organize the results as a DataFrame
    df = DataFrame(; y1, x1, f1, fh)
    df = stack(df, [:f1, :fh])

    # Plot the results
    R"""
    p <- 
        $df |>
        ggplot2::ggplot(ggplot2::aes(x = y1, y = value, color = variable)) +
        ggplot2::geom_line() +
        ggplot2::facet_grid(
            cols = ggplot2::vars(x1), 
            labeller = ggplot2::labeller(.cols = ggplot2::label_both)
        ) +
        ggplot2::theme_classic()
    ggplot2::ggsave(filename = $filename)
    """
    return y0, y1, X0, X1
end

# Example 1: No predictor (normal distribution)
begin
    dy(x) = Normal(0, 1)
    Random.seed!(1)
    x1 = [-1.0, 1.0]
    x0 = rand(x1, 500)
    filename = "figures/ex-normal-1.png"
    a = foo(dy, x0, x1; filename, mcmcsize = 4000)
end;

# Example 2: No predictor (mixture distribution)
begin
    dy(x) = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    Random.seed!(1)
    x1 = [-1.0, 1.0]
    x0 = rand(x1, 500)
    filename = "figures/ex-normal-2.png"
    a = foo(dy, x0, x1; filename, mcmcsize = 4000)
end;

# Example 3: One discrete predictor (normal distribution)
begin
    dy(x) = Normal(x, 1)
    Random.seed!(1)
    x1 = [-1.0, 1.0]
    x0 = rand(x1, 500)
    filename = "figures/ex-normal-3.png"
    a = foo(dy, x0, x1; filename, mcmcsize = 4000)
end;

# Example 4: One discrete predictor (mixture distribution)
begin
    dy(x) = MixtureModel(Normal, [(x + 1, 1.0), (x - 1, 0.25)], [0.5, 0.5])
    Random.seed!(1)
    x1 = [-1.0, 1.0]
    x0 = rand(x1, 500)
    filename = "figures/ex-normal-4.png"
    a = foo(dy, x0, x1; filename, mcmcsize = 10000)
end;

# Example 1: No predictor ------------------------------------------------------

function simulate_sample_01(N0, N1)
    dy = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    X00 = ones(N0, 1)
    X10 = ones(N1, 1)
    y00 = rand(dy, N0)
    y10 = LinRange(minimum(y00), maximum(y00), N1) |> collect
    y0, y1, X0, X1, my00, sy00 = standardize_data!(y00, y10, X00, X10)
    return y0, y1, X0, X1
end;

begin
    Random.seed!(1)
    N0, N1 = 1000, 50;
    rng = MersenneTwister(1);
    y0, y1, X0, X1 = simulate_sample_01(N0, N1);
    smpl = BNP.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNP.sample!(smpl; mcmcsize = 10000);
end;

f1 = mean(chainf);
plot(
    layer(x = y1, y = f1, Geom.line, color=["bnp"]), 
    layer(x = y0, Geom.density, color=["kden"]),
    layer(x = y0, Geom.histogram(density = true), color=["hist"])
)

#========================================================#
# Example 2 - 1 discrete predictor (simple distribution) #
#========================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0)]
    y0 = X0 * ones(2) + randn(N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [ones(2 * N1) x1]
    return y0, X0, y1, X1
end;

begin
    Random.seed!(1)
    N0, N1 = 1000, 50;
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);    
end;

fb = mean(chainf);
dy(x) = Normal(1 + x, 1);
plot(x = y0, color = X0[:, 2], Geom.density)    
plot(
    layer(x = y1, y = fb, color = X1[:, 2], Geom.line),
    layer(x = y1, y = pdf.(dy.(X1[:, 2]), y1), color = X1[:, 2], Geom.line)
)
mean(chainβ)

#=========================================================#
# Example 3 - 1 discrete predictor (mixture distribution) #
#=========================================================#

# Simulate a sample 
function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1.0, 1.0), (-1.0, 0.25)], [0.5, 0.5])
    x0 = [-0.5, 0.5]
    X0 = [ones(N0) rand(x0, N0)]
    y0 = X0 * ones(2) + rand(dϵ, N0)
    y1 = LinRange(minimum(y0), maximum(y0), N1) |> collect |> x -> repeat(x, 2)
    x1 = [x0[1] * ones(N1); x0[2] * ones(N1)]
    X1 = [ones(2 * N1) x1]
    return y0, X0, y1, X1
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

fb = mean(chainf);
plot(x = y0, color = X0[:, 2], Geom.histogram(density = true, bincount = 50))
plot(x = y0, color = X0[:, 2], Geom.density)
plot(x = y1, y = fb, color = X1[:, 2], Geom.line)
mean(chainβ)

#=================================================#
# Example 4 - 2 predictors (mixture distribution) #
#=================================================#

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
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, X0, y1, X1 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

fb = mean(chainf);
X0concat = @. string(X0[:, 2]) * " " * string(X0[:, 3])
X1concat = @. string(X1[:, 2]) * " " * string(X1[:, 3])
plot(x = y0, color = X0concat, Geom.density)
plot(x = y1, y = fb, color = X1concat, Geom.line)
mean(chainβ)

#==========================================================#
# Example 5 - 1 continuous predictor (simple distribution) #
#==========================================================#

function simulate_sample(N0, N1)
    dϵ = Normal(0.0, 1.0)
    x0 = LinRange(-2, 2, N0) |> collect
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 6, Boundary.knots = c(-4, 4)))")
    y0 = x0 + rand(dϵ, N0)
    x_grid = LinRange(-2, 2, N1) |> collect 
    y_grid = LinRange(-3, 3, N1) |> collect
    R"""
    grid <- expand.grid(y1 = $y_grid, x1 = $x_grid)
    y1 <- grid$y1
    x1 <- grid$x1
    """
    @rget y1 x1
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 6, Boundary.knots = c(-4, 4)), $x1))")
    # Standardization
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0[:, 1] .= 1 
    X1[:, 1] .= 1 
    return y0, x0, X0, y1, x1, X1, mean_y0, std_y0
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, x0, X0, y1, x1, X1, mean_y0, std_y0 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

dy1(x) = Normal((x - mean_y0[1]) / std_y0[1], 1 / std_y0[1])
f0 = pdf.(dy1.(x1), y1);
fh = mean(chainf);
df = DataFrame(; y1, x1, f0, fh);

R"""
$df |>
    tidyr::pivot_longer(c(f0, fh)) |>
    ggplot2::ggplot(ggplot2::aes(x = x1, y = y1, z = value)) +
    ggplot2::geom_contour() +
    ggplot2::facet_grid(~ name)
ggplot2::ggsave("figures/ex-normal-5.png")
"""

# Save the estimated surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-f0) |>
    tidyr::pivot_wider(values_from = fh, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-5-surface-hat.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "fh"
)    
dev.off()
"""

# Save the true surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-fh) |>
    tidyr::pivot_wider(values_from = f0, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-5-surface-true.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "f1"
)    
dev.off()
"""

#===========================================================#
# Example 6 - 1 continuous predictor (mixture distribution) #
#===========================================================#

function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1, 1), (-1, 0.25)], [0.5, 0.5])
    x0 = LinRange(-2, 2, N0) |> collect
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 8, Boundary.knots = c(-5, 5)))")
    y0 = x0 + rand(dϵ, N0)
    x_grid = LinRange(-2, 2, N1)
    y_grid = LinRange(-3, 3, N1)
    R"""
    grid <- expand.grid(y1 = $y_grid, x1 = $x_grid)
    y1 <- grid$y1
    x1 <- grid$x1
    """
    @rget y1 x1
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 8, Boundary.knots = c(-5, 5)), $x1))")
    # Standardization
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0[:, 1] .= 1 
    X1[:, 1] .= 1 
    return y0, x0, X0, y1, x1, X1, mean_y0, std_y0
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, x0, X0, y1, x1, X1, mean_y0, std_y0 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

dy1(x) = MixtureModel(Normal[
        Normal((x - 1 - mean_y0[1]) / std_y0[1], 0.25 / std_y0[1]), 
        Normal((x + 1 - mean_y0[1]) / std_y0[1], 1 / std_y0[1])
    ],
    [0.5, 0.5]
)
f0 = pdf.(dy1.(x1), y1);
fh = mean(chainf);
df = DataFrame(; y1, x1, f0, fh);

# Save the estimated surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-f0) |>
    tidyr::pivot_wider(values_from = fh, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-6-surface-hat.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "fh"
)    
dev.off()
"""

# Save the true surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-fh) |>
    tidyr::pivot_wider(values_from = f0, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-6-surface-true.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "f1"
)    
dev.off()
"""

R"""
png("figures/ex-normal-6-scatter.png")
plot($x0, $y0)
dev.off()
"""

#==================================#
# Example 5 (now without bsplines) #
#==================================#

function simulate_sample(N0, N1)
    dϵ = Normal(0.0, 1.0)
    x0 = LinRange(-2, 2, N0) |> collect
    X0 = x0[:, :]
    y0 = x0 + rand(dϵ, N0)
    x_grid = LinRange(-2, 2, N1) |> collect 
    y_grid = LinRange(-3, 3, N1) |> collect
    R"""
    grid <- expand.grid(y1 = $y_grid, x1 = $x_grid)
    y1 <- grid$y1
    x1 <- grid$x1
    """
    @rget y1 x1
    X1 = x1[:, :]
    # Standardization
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0 = [ones(size(X0, 1)) X0]
    X1 = [ones(size(X1, 1)) X1]
    return y0, x0, X0, y1, x1, X1, mean_y0, std_y0
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, x0, X0, y1, x1, X1, mean_y0, std_y0 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, _ = BNP.sample!(smpl; mcmcsize = 10000);
end;

dy1(x) = Normal((x - mean_y0[1]) / std_y0[1], 1 / std_y0[1])
f0 = pdf.(dy1.(x1), y1);
fh = mean(chainf);
df = DataFrame(; y1, x1, f0, fh);

# Save the estimated surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-f0) |>
    tidyr::pivot_wider(values_from = fh, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-5-surface-no-bspline-hat.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "fh"
)    
dev.off()
"""

# Save the true surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-fh) |>
    tidyr::pivot_wider(values_from = f0, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-5-surface-no-bspline-true.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "f1"
)    
dev.off()
"""

#==============================#
# Example 6 (without bsplines) #
#==============================#

function simulate_sample(N0, N1)
    dϵ = MixtureModel(Normal, [(1, 1), (-1, 0.25)], [0.5, 0.5])
    x0 = LinRange(-2, 2, N0) |> collect
    X0 = x0[:, :]
    y0 = x0 + rand(dϵ, N0)
    x_grid = LinRange(-2, 2, N1) |> collect 
    y_grid = LinRange(-3, 3, N1) |> collect
    R"""
    grid <- expand.grid(y1 = $y_grid, x1 = $x_grid)
    y1 <- grid$y1
    x1 <- grid$x1
    """
    @rget y1 x1
    X1 = x1[:, :]
    # Standardization
    mean_y0, std_y0 = mean_and_std(y0, 1)
    mean_X0, std_X0 = mean_and_std(X0, 1)
    y0 = (y0 .- mean_y0) ./ std_y0
    X0 = (X0 .- mean_X0) ./ std_X0
    X1 = (X1 .- mean_X0) ./ std_X0
    X0 = [ones(size(X0, 1)) X0]
    X1 = [ones(size(X1, 1)) X1]
    return y0, x0, X0, y1, x1, X1, mean_y0, std_y0
end

begin
    Random.seed!(1)
    N0, N1 = 1000, 50
    y0, x0, X0, y1, x1, X1, mean_y0, std_y0 = simulate_sample(N0, N1);
    smpl = BNPRegressionGGA2021.DGSBPNormal(; y0, X0, y1, X1);
    chainf, chainβ = BNPRegressionGGA2021.sample!(smpl; mcmcsize = 10000);
end;

dy1(x) = MixtureModel(Normal[
        Normal((x - 1 - mean_y0[1]) / std_y0[1], 0.25 / std_y0[1]), 
        Normal((x + 1 - mean_y0[1]) / std_y0[1], 1 / std_y0[1])
    ],
    [0.5, 0.5]
)
f0 = pdf.(dy1.(x1), y1);
fh = mean(chainf);
df = DataFrame(; y1, x1, f0, fh);

# Save the estimated surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-f0) |>
    tidyr::pivot_wider(values_from = fh, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-6-no-bspline-surface-hat.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "fh"
)    
dev.off()
"""

# Save the true surface
R"""
xv <- sort(unique($df[["x1"]]))
yv <- sort(unique($df[["y1"]]))
zv <- $df |>
    dplyr::select(-fh) |>
    tidyr::pivot_wider(values_from = f0, names_from = y1) |>
    dplyr::select(-x1) |>
    as.matrix()
png("figures/ex-normal-6-no-bspline-surface-true.png")
fig <- persp(
    x = xv, y = yv, z = zv, phi = 45, theta = 45, 
    xlab = "x", ylab = "y", zlab = "f1"
)    
dev.off()
"""

R"""
png("figures/ex-normal-6-no-bspline-scatter.png")
plot($x0, $y0)
dev.off()
"""