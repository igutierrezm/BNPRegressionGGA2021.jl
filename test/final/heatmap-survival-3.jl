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

# Helper functions ------------------------------------------------------------

# function run_experiment_bnp(dy; N0, Nrep, Niter, id, filename)
#     Random.seed!(1) # seed
#     samples = [generate_sample(dy; N0, Nrep) for _ in 1:Niter] # data
#     gammas = [get_gamma_bnp(samples[id]) for id in 1:Niter] # map gamma
#     reduced_gammas = reduce_gammas(gammas; method = "bnp", id = id, N0 = N0)
#     CSV.write(filename, reduced_gammas) # csv file
# end

# function run_experiment_freq(dy; N0, Nrep, Niter, id, filename)
#     Random.seed!(1) # seed
#     samples = [generate_sample(dy; N0, Nrep) for _ in 1:Niter] # data
#     gammas = [get_gamma_freq(samples[id]) for id in 1:Niter] # map gamma
#     reduced_gammas = reduce_gammas(gammas; method = "freq", id = id, N0 = N0)
#     CSV.write(filename, reduced_gammas) # csv file
# end

function generate_sample(dy; N0 = 50, N1 = 0, Nrep = 10)
    # Simulate the data
    X0d = rand([0, 1], N0, 2)
    x0c = repeat(LinRange(-2, 2, N0 ÷ Nrep), Nrep)
    y0 = @. rand(dy(X0d[:, 2]))
    event0 = y0 .< 5

    # Generate the grid
    y1 = LinRange(-3, 3, N1) |> collect
    x1d1 = [0, 1]
    x1d2 = [0, 1]
    x1c = [0, 1]

    # Expand the grid
    grid = 
        R"""expand.grid(y1 = $y1, x1c = $x1c, x1d1 = $x1d1, x1d2 = $x1d2)""" |>
        rcopy
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
    f1 = pdf.(dy.(x1d2), my0 .+ y1 .* sy0) .* sy0

    # Set the mapping 
    mapping = [[1], collect(2:7), [8], [9]]

    # Return the preprocessed results
    return y0, event0, y1, X0, X1, x1c, f1, mapping
end;

# # Get the MAP estimator of gamma using BNP
# function get_gamma_bnp(sample)
#     y0, event0, y1, X0, X1, f1, mapping = sample
#     smpl = BNP.DGSBPNormalDependent(; y0, event0, X0, y1, X1, mapping)
#     _, _, chaing = BNP.sample!(smpl; mcmcsize = 10000)
#     gamma = begin
#         chaing |>
#         DataStructures.counter |>
#         collect |>
#         x -> sort(x, by = x -> x[2], rev = true) |>
#         x -> getindex(x, 1) |>
#         x -> getindex(x, 1)
#     end
#     return gamma
# end

# # Estimate gamma using a frequentist alternative
# function get_gamma_freq(sample)
#     y0, event0, x0, y1, x1, X0, X1, f1, mapping = sample
#     R"""
#     library(dplyr)
#     data = data.frame(
#         status = $event0,
#         y = $y0,
#         x2 = $x0, 
#         x3 = $X0[, 8],
#         x4 = $X0[, 9],
#         x5 = $X0[, 10],
#         x6 = $X0[, 11]
#     )
#     fitted_model <- survival::coxph(survival::Surv(y, status) ~ ., data = data)
#     fitted_bestmodel <- MASS::stepAIC(fitted_model)
#     best_model_trues <- 
#         names(fitted_bestmodel$coefficients) %>%
#         gsub("x", "", .) %>%
#         as.numeric()
#     best_gamma <- rep(FALSE, 6)
#     best_gamma[best_model_trues] <- TRUE
#     best_gamma[1] <- TRUE
#     """
#     @rget best_gamma
#     return best_gamma
# end

# # Reduce the MAP estimators after their estimation
# function reduce_gammas(gammas; method, id, N0)
#     names = "g" .* string.(1:6)
#     mat = reduce(vcat, gammas')
#     df = DataFrame(mat, names)
#     df[!, :method] .= method
#     df[!, :id] .= id
#     df[!, :N0] .= N0
#     return df
# end

# Experiment 1, N0 = 50 (bnp)
begin 
    filename = "data/final/survival-gamma-loggamma-bnp-50.csv"
    dy(x) = MixtureModel(Normal, [(3 + x, 0.8 + 0.2x), (3 - x, 0.8)], [.4, .6])
    # dy(x) = Normal(2 + x, 1)
    y0, event0, y1, X0, X1, x1c, f1, mapping = generate_sample(dy; N0 = 1000, N1 = 50, Nrep = 10)
    smpl = BNP.DGSBPNormalDependent(; y0, event0, X0, y1, X1, mapping)
    chainf, chainβ, chaing, chainnclus = BNP.sample!(smpl; mcmcsize = 10000)
    println(mean(chainnclus))
    println([mean_and_var(y0) maximum(y0)])
    println([mean_and_var(smpl.ỹ0) maximum(smpl.ỹ0)])
    println(DataStructures.counter(chaing))
end

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
data.frame(
        y0 = $y0,
        x0d2 = $(X0[:, 9])
    ) %>%
    ggplot2::ggplot(ggplot2::aes(x = y0, color = factor(x0d2))) +
    ggplot2::geom_density()
    ggplot2::ggsave("tmp1.png")

p <-
    data.frame(
        y1 = $y1,
        f1 = $f1, 
        fh = $(mean(chainf)), 
        x1c = $x1c, 
        x1d1 = $(X1[:, 8]), 
        x1d2 = $(X1[:, 9])
    ) %>%
    dplyr::filter(x1c == 1, x1d1 == 1) %>%
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
ggplot2::ggsave("tmp2.png")
"""


# R"""
# library(ggplot2)
# library(dplyr)
# df <- 
#     data.frame(y1 = $y1, x1d2 = $(X1[:, 9]), f1 = $f1) %>%
#     dplyr::group_by(y1, x1d2) %>%
#     dplyr::summarize(f1 = mean(f1))
# ggplot2::ggplot(df, ggplot2::aes(x = y1, y = f1, color = as.factor(x1d2))) +
#     ggplot2::geom_line()
# ggplot2::ggsave("tmp.png")
# """

# # Experiment 1, N0 = 50 (freq)
# begin 
#     filename = "data/final/survival-gamma-loggamma-freq-50.csv"
#     dy(x) = MixtureModel(Normal, [(3 + x, 0.8 + 0.2x), (3 - x, 0.8)], [.4, .6])
#     run_experiment_freq(dy; N0 = 50, Nrep = 5, Niter = 1000, id = 1, filename)
# end 

# # Experiment 1, N0 = 100 (bnp)
# begin 
#     filename = "data/final/survival-gamma-loggamma-bnp-100.csv"
#     dy(x) = MixtureModel(Normal, [(3 + x, 0.8 + 0.2x), (3 - x, 0.8)], [.4, .6])
#     run_experiment_bnp(dy; N0 = 100, Nrep = 5, Niter = 10, id = 1, filename)
# end 

# # Experiment 1, N0 = 100 (freq)
# begin 
#     filename = "data/final/survival-gamma-loggamma-freq-100.csv"
#     dy(x) = MixtureModel(Normal, [(3 + x, 0.8 + 0.2x), (3 - x, 0.8)], [.4, .6])
#     run_experiment_freq(dy; N0 = 100, Nrep = 5, Niter = 1000, id = 1, filename)
# end 

# # Experiment 1, N0 = 200 (bnp)
# begin 
#     filename = "data/final/survival-gamma-loggamma-bnp-200.csv"
#     dy(x) = MixtureModel(Normal, [(3 + x, 0.8 + 0.2x), (3 - x, 0.8)], [.4, .6])
#     run_experiment_bnp(dy; N0 = 200, Nrep = 5, Niter = 10, id = 1, filename)
# end 

# # Experiment 1, N0 = 200 (freq)
# begin 
#     filename = "data/final/survival-gamma-loggamma-freq-200.csv"
#     dy(x) = MixtureModel(Normal, [(3 + x, 0.8 + 0.2x), (3 - x, 0.8)], [.4, .6])
#     run_experiment_freq(dy; N0 = 200, Nrep = 5, Niter = 1000, id = 1, filename)
# end 

# # Summary of the results
# R"""
# data <- 
#     list.files(
#         path = "data/final", 
#         pattern = "survival-gamma*", 
#         full.names = TRUE
#     ) |>
#     purrr::map(readr::read_csv, show_col_types = FALSE) |>
#     dplyr::bind_rows() |>
#     dplyr::group_by(method, id, N0, g1, g2, g3, g4, g5, g6) |>
#     dplyr::count(name = "frequency") |>
#     dplyr::ungroup() |>
#     dplyr::mutate(gamma = paste0(g1, g2, g3, g4, g5, g6)) |>
#     dplyr::select(gamma, frequency, id, method, N = N0)
# p <- 
#     data |>
#     dplyr::mutate(
#         method = method |>
#             dplyr::recode(bnp = "BNP", freq = "BSS (Best Subset Selection)"),
#         distribution = id |>
#         dplyr::recode(`1` = "Normal", `2` = "Skew Normal"),
#         N = factor(N, ordered = TRUE)
#     ) |>
#     ggplot2::ggplot(ggplot2::aes(y = gamma, x = N, fill = frequency)) +
#     ggplot2::geom_tile(color = "grey90") +
#     ggplot2::geom_text(ggplot2::aes(label = frequency), color = "red") +
#     ggplot2::facet_grid(
#         distribution ~ method, 
#         labeller = ggplot2::labeller(
#             distribution = ggplot2::label_both,
#             method = ggplot2::label_both
#         )
#     ) +
#     ggplot2::theme_classic() +
#     ggplot2::theme(legend.position = "top") +
#     ggplot2::scale_fill_gradient(low = "white", high = "black") + 
#     ggplot2::labs(
#         x = "N (sample size)",
#         y = "gamma (hypothesis vector), as a string of 0s and 1s"
#     )
# ggplot2::ggsave("figures/final/heatmap-survival.png")
# """

# TODO:
# Save the number of clusters in each iteration
# Try a model with 3 variables and examine the results
# Plot the density marginal and conditional