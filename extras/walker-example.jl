# Load the required packages
begin 
    using Base.Iterators
    using BNPRegressionGGA2021
    using CSV
    using DataFrames
    using Distributions
    using LinearAlgebra
    using Random
    using RCall
    const BNP = BNPRegressionGGA2021
end;

# Define the DGP
function simulate_sample(N0, N1)
    # Set the conditional density
    dy(xc) = Normal(0.2 * xc^3, 0.5)
    
    # Simulate a sample
    x0 = collect(LinRange(-3, 3, N0))
    y0 = [rand(dy(xc)) for xc in x0]
    X0 = rcopy(R"cbind(1, splines::bs($x0, df = 6, Boundary.knots = c(-3, 3)))")

    # Generate the grid points
    y_grid = LinRange(-3.0, 3.0, N1)
    x_grid = [-1, 1]
    y1, x1 = Iterators.product(y_grid, x_grid) |> x -> zip(x...) .|> collect
    X1 = rcopy(R"cbind(1, predict(splines::bs($x0, df = 6, Boundary.knots = c(-3, 3)), $x1))")
    return dy, y0, x0, X0, y1, x1, X1
end

# Simulate the data
begin
    Random.seed!(1);
    N0, N1 = 500, 50;
    dy, y0, x0, X0, y1, x1, X1 = simulate_sample(N0, N1);
end;

# Fit the model 
begin
    mapping = [[1], collect(2:size(X0, 2))];
    m = BNP.DGSBPNormal(; y0, X0, y1, X1, mapping, update_γ = false);
    chainf, chainβ, chaing = BNP.sample!(m; mcmcsize = 4000);
end; 

# Collect the estimated densities
df = DataFrame(y = y1, x = x1, fh = mean(chainf), f0 = pdf.(dy.(x1), y1))

# Save the results
CSV.write("data/walker-example.csv", df)

# Create a plot
R"""
# Munge the data
df <- 
    readr::read_csv("data/walker-example.csv") |>
    dplyr::mutate(x = factor(x)) |>
    tidyr::pivot_longer(c(fh, f0))

# Plot the conditional posterior predictive densities
p <- 
    df |> 
    ggplot2::ggplot(ggplot2::aes(x = y, y = value, color = name)) + 
    ggplot2::facet_grid(row = ggplot2::vars(x)) +
    ggplot2::geom_line()

# Save the plot in png format
"figures/walker-example.png" |>
    ggplot2::ggsave(plot = p, width = 5 * 8 / 9, height = 3)
"""
