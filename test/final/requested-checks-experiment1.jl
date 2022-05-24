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
    import Distributions: pdf
    import Base: rand
end;

# Generate a sample ready for our estimation procedure
function generate_sample(dy; N0, N1, Nrep = 10)
    # Simulate the data
    X0d = rand([0, 1], N0, 2) 
    x0c = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    y0 = @. rand(dy(x0c, X0d[:, 2]))
    event0 = y0 .< 5

    # Generate the grid
    y1 = LinRange(-3, 3, N1) |> collect
    x1d1 = [0, 1]
    x1d2 = [0, 1]
    x1c = [0, 1]

    # Expand the grid
    grid = rcopy(R"expand.grid(y1 = $y1, x1c = $x1c, x1d1 = $x1d1, x1d2 = $x1d2)")
    y1 = grid[!, :y1]
    x1c = grid[!, :x1c]
    x1d1 = grid[!, :x1d1]
    x1d2 = grid[!, :x1d2]
    X1d = [x1d1 x1d2]
    
    # Expand x0-x1 using splines
    R"""
    x0cmin <- min($x0c)
    x0cmax <- max($x0c)
    X0c <- splines::bs($x0c, df = 6, Boundary.knots = c(x0cmin, x0cmax))
    X1c <- predict(X0c, $x1c)  
    """
    @rget X0c X1c

    # Standardise responses and predictors
    mX0c, sX0c = mean_and_std(X0c, 1)
    for col in 1:size(X0c, 2)
        if sX0c[col] > 0
            X0c[:, col] = (X0c[:, col] .- mX0c[col]) ./ sX0c[col]
            X1c[:, col] = (X1c[:, col] .- mX0c[col]) ./ sX0c[col]
        end
    end
    my0, sy0 = mean_and_std(y0, 1)
    y0 = (y0 .- my0) ./ sy0

    # Add a constant term to the design matrices
    X0 = [ones(size(X0c, 1)) X0c X0d]
    X1 = [ones(size(X1c, 1)) X1c X1d]

    # Generate the true (standardized) density over the grid
    f1 = pdf.(dy.(x1c, x1d2), my0 .+ y1 .* sy0) .* sy0

    # Set the mapping 
    mapping = [[1], collect(2:7), [8], [9]]

    # Return the preprocessed results
    return y0, event0, y1, X0, X1, x1c, f1, mapping
end;

# First experiment: 1 relevant variable
begin 
    # Set the seed
    Random.seed!(1) 
    # Set the true conditional distribution
    function dy(xc, xd)
        atoms = [(3 + xd, 0.8 + 0.2xd), (3 - xd, 0.8)]
        weights = [.4, .6]
        MixtureModel(Normal, atoms, weights)
    end
    # Generate the sample
    y0, event0, y1, X0, X1, x1c, f1, mapping = 
        generate_sample(dy; N0 = 5000, N1 = 50, Nrep = 10)

    # Fit the model
    smpl = BNP.DGSBPNormalDependent(; y0, event0, X0, y1, X1, mapping)
    chainf, chainβ, chaing, chainnclus = BNP.sample!(smpl; mcmcsize = 10000)
    chaing = join.(map.(Int, chaing))

    # Print some summary statistics
    println(mean(chainnclus)) # average number of clusters
    println([mean_and_var(y0) maximum(y0)]) # mean/var of the responses
    println([mean_and_var(smpl.ỹ0) maximum(smpl.ỹ0)]) # mean/var of the responses
    println(DataStructures.counter(chaing)) # freq table for gamma
end

# Plot a kdensity of the sample (ignoring the censoring)
R"""
library(magrittr)
data.frame(
        y0 = $y0,
        x0d2 = $(X0[:, 9])
    ) %>%
    ggplot2::ggplot(ggplot2::aes(x = y0, color = factor(x0d2))) +
    ggplot2::geom_histogram()
    ggplot2::ggsave("tmp1_1.png")
"""

# Save the simulation results in a format amenable for later plots
R"""
library(magrittr)
simulation_results <-
    purrr::map2_df(
        $chainf,
        $chaing,
        ~ data.frame(
            x1c1 = $x1c,
            x1d1 = $(X1[:, 8]),
            x1d2 = $(X1[:, 9]),
            y1 = $y1,
            f1 = $f1,
            fh = .x,
            gamma = .y
        ), 
        .id = "gibbs_iter"
    )
NULL
"""

# Plot the true and the fitted density, according to our method
R"""
library(magrittr)
p <-
    simulation_results %>%
    dplyr::filter(x1c1 == 1, x1d1 == 1) %>%
    dplyr::group_by(x1c1, x1d1, x1d2, y1) %>%
    dplyr::summarize(f1 = mean(f1), fh = mean(fh)) %>%
    tidyr::pivot_longer(f1:fh) %>%
    ggplot2::ggplot(
        ggplot2::aes(
            x = y1, 
            y = value, 
            color = factor(x1d2),
            linetype = factor(name)
        )
    ) +
    ggplot2::geom_line()
ggplot2::ggsave("tmp1_2.png")
"""

# Plot the true and the fitted density, according to our method
R"""
library(magrittr)
p <-
    simulation_results %>%
    dplyr::filter(x1c1 == 1, x1d1 == 1) %>%
    dplyr::group_by(x1c1, x1d1, x1d2, y1, gamma) %>%
    dplyr::summarize(f1 = mean(f1), fh = mean(fh)) %>%
    tidyr::pivot_longer(f1:fh) %>%
    ggplot2::ggplot(
        ggplot2::aes(
            x = y1, 
            y = value, 
            color = factor(x1d2),
            linetype = factor(name)
        )
    ) +
    ggplot2::geom_line() +
    ggplot2::facet_grid(rows = ggplot2::vars(gamma))
ggplot2::ggsave("tmp1_3.png")
"""