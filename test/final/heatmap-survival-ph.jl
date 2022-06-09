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

R"""
library(methods)
Sys.setenv(JAVA_HOME = "/usr/lib/jvm/default-java")
Sys.setenv(LD_LIBRARY_PATH = "/usr/local/lib/R/lib:/usr/lib/jvm/default-java/lib:/usr/lib/jvm/default-java/lib/server")
Sys.getenv("JAVA_HOME")
Sys.getenv("LD_LIBRARY_PATH")
system("R CMD javareconf")
library(rJava)
# library(leaps)
# library(glmulti)
# .jinit() # this starts the JVM
# s <- .jnew("java/lang/String", "Hello World!")
# .jcall(s,"I","length")
"""

# Helper functions ------------------------------------------------------------

function run_experiment_bnp(dy; N0, Nrep, Niter, id, filename)
    println(filename)
    Random.seed!(1) # seed
    samples = [generate_sample(dy; N0, Nrep) for _ in 1:Niter] # data
    gammas = [get_gamma_bnp(samples[id], id) for id in 1:Niter] # map gamma
    reduced_gammas = reduce_gammas(gammas; method = "bnp", id = id, N0 = N0)
    CSV.write(filename, reduced_gammas) # csv file
end

function run_experiment_freq(dy; N0, Nrep, Niter, id, filename)
    Random.seed!(1) # seed
    samples = [generate_sample(dy; N0, Nrep) for _ in 1:Niter] # data
    gammas = [get_gamma_freq_bss(samples[id]) for id in 1:Niter] # map gamma
    reduced_gammas = reduce_gammas(gammas; method = "freq", id = id, N0 = N0)
    # CSV.write(filename, reduced_gammas) # csv file
end

function generate_sample(dy; N0 = 50, Nrep = 10)
    # Simulate the data
    X0d = rand([0.0, 1.0], N0, 4)
    X0c = repeat(LinRange(-1, 1, N0 ÷ Nrep), Nrep)
    y0 = @. rand(dy(X0c, X0d[:, 4]))
    X0 = [X0c X0d]
    event0 = y0 .< 5

    # Standardise responses and predictors and add a constant term
    my0, sy0 = mean_and_std(y0, 1)
    mX0, sX0 = mean_and_std(X0, 1)
    y0 .= (y0 .- my0) ./ sy0
    X0 .= (X0 .- mX0) ./ sX0
    X0 = [ones(N0) X0]

    # Generate a grid for prediction
    X1 = zeros(Float64, 0, size(X0, 2))
    y1 = zeros(Float64, 0)

    # Set the mapping 
    mapping = [[1], [2], [3], [4], [5], [6]]

    # Return the preprocessed results
    return event0, y0, y1, X0, X1, mapping
end;

# Get the MAP estimator of gamma using BNP
function get_gamma_bnp(sample, id)
    println(id)
    event0, y0, y1, X0, X1, mapping = sample
    smpl = BNP.DGSBPNormalDependent(; y0, event0, X0, y1, X1, mapping)
    _, _, chaing = BNP.sample!(smpl; mcmcsize = 10000)
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
    event0, y0, y1, X0, X1, mapping = sample
    R"""
    library(dplyr)
    data = data.frame(
        status = $event0,
        y = $y0,
        x2 = $X0[, 2], 
        x3 = $X0[, 3],
        x4 = $X0[, 4],
        x5 = $X0[, 5],
        x6 = $X0[, 6]
    )
    fitted_model <- survival::coxph(survival::Surv(y, status) ~ ., data = data)
    fitted_bestmodel <- MASS::stepAIC(fitted_model)
    best_model_trues <- 
        names(fitted_bestmodel$coefficients) %>%
        gsub("x", "", .) %>%
        as.numeric()
    best_gamma <- rep(0L, 6)
    best_gamma[best_model_trues] <- 1L
    best_gamma[1] <- 1L
    """
    @rget best_gamma
    return best_gamma
end

# Estimate gamma using a frequentist alternative
function get_gamma_freq_bss(sample)
    event0, y0, y1, X0, X1, mapping = sample
    R"""
    library(dplyr)
    # data = data.frame(
    #     y = survival::Surv($y0, $event0),
    #     x2 = $X0[, 2], 
    #     x3 = $X0[, 3],
    #     x4 = $X0[, 4],
    #     x5 = $X0[, 5],
    #     x6 = $X0[, 6]
    # )
    # fitted_best_model <-
    #     glmulti::glmulti(
    #         formula = y ~ .,
    #         data = data,
    #         plotty = F,                   # No plot
    #         report = F,                   # No interim reports
    #         level = 1,                    # No interaction considered
    #         method = "h",                 # Exhaustive approach
    #         crit = "aic",                 # AIC as criteria
    #         confsetsize = 1,              # Keep best model
    #         fitfunction = survival::coxph # coxph function
    #     )
    # best_model_trues <- 
    #     names(fitted_best_model@objects[[1]]$coefficients) %>%
    #     gsub("x", "", .) %>%
    #     as.integer()
    best_gamma <- rep(0L, 6)
    # best_gamma[best_model_trues] <- 1L
    # best_gamma[1] <- 1L
    """
    @rget best_gamma
    return best_gamma
end


# Reduce the MAP estimators after their estimation
function reduce_gammas(gammas; method, id, N0)
    names = "g" .* string.(1:6)
    mat = reduce(vcat, gammas')
    df = DataFrame(mat, names)
    df[!, :method] .= method
    df[!, :id] .= id
    df[!, :N0] .= N0
    return df
end

function dy(xc, xd)
    Weibull(2, 1.5 + xc + xd)
end

# Experiment 1, N0 = 50 (bnp)
begin 
    filename = "data/final/survival-gamma-ph-bnp-50.csv"
    run_experiment_bnp(dy; N0 = 50, Nrep = 5, Niter = 100, id = 1, filename)
end

# Experiment 1, N0 = 50 (freq)
begin 
    filename = "data/final/survival-gamma-ph-freq-50.csv"
    run_experiment_freq(dy; N0 = 50, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 100 (bnp)
begin 
    filename = "data/final/survival-gamma-ph-bnp-100.csv"
    run_experiment_bnp(dy; N0 = 100, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 100 (freq)
begin 
    filename = "data/final/survival-gamma-ph-freq-100.csv"
    run_experiment_freq(dy; N0 = 100, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 200 (bnp)
begin 
    filename = "data/final/survival-gamma-ph-bnp-200.csv"
    run_experiment_bnp(dy; N0 = 200, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 200 (freq)
begin 
    filename = "data/final/survival-gamma-ph-freq-200.csv"
    run_experiment_freq(dy; N0 = 200, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 500 (bnp)
begin 
    filename = "data/final/survival-gamma-ph-bnp-500.csv"
    run_experiment_bnp(dy; N0 = 500, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 500 (freq)
begin 
    filename = "data/final/survival-gamma-ph-freq-500.csv"
    run_experiment_freq(dy; N0 = 500, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 1000 (bnp)
begin 
    filename = "data/final/survival-gamma-ph-bnp-1000.csv"
    run_experiment_bnp(dy; N0 = 1000, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Experiment 1, N0 = 1000 (freq)
begin 
    filename = "data/final/survival-gamma-ph-freq-1000.csv"
    run_experiment_freq(dy; N0 = 1000, Nrep = 5, Niter = 100, id = 1, filename)
end 

# Summary of the results
R"""
data <- 
    list.files(
        path = "data/final", 
        pattern = "survival-gamma-ph*", 
        full.names = TRUE
    ) |>
    purrr::map(readr::read_csv, show_col_types = FALSE) |>
    dplyr::bind_rows() |>
    dplyr::group_by(method, id, N0, g1, g2, g3, g4, g5, g6) |>
    dplyr::count(name = "frequency") |>
    dplyr::ungroup() |>
    dplyr::mutate(gamma = paste0(g1, g2, g3, g4, g5, g6)) |>
    dplyr::select(gamma, frequency, id, method, N = N0)
p <- 
    data |>
    dplyr::mutate(
        method = method |>
            dplyr::recode(bnp = "BNP", freq = "Stepwise (AIC)"),
        distribution = id |>
        dplyr::recode(`1` = "Normal mixture"),
        N = factor(N, ordered = TRUE)
    ) |>
    ggplot2::ggplot(ggplot2::aes(y = gamma, x = N, fill = frequency)) +
    ggplot2::geom_tile(color = "grey90") +
    ggplot2::geom_text(ggplot2::aes(label = frequency), color = "red") +
    ggplot2::facet_grid(
        distribution ~ method, 
        labeller = ggplot2::labeller(
            distribution = ggplot2::label_both,
            method = ggplot2::label_both
        )
    ) +
    ggplot2::theme_classic() +
    ggplot2::theme(legend.position = "top") +
    ggplot2::scale_fill_gradient(low = "white", high = "black") + 
    ggplot2::labs(
        x = "N (sample size)",
        y = "gamma (hypothesis vector), as a string of 0s and 1s"
    )
ggplot2::ggsave("figures/final/heatmap-survival-ph.png")
"""

# TODO:
# Save the number of clusters in each iteration (done)
# Try a model with 3 variables and examine the results
# Plot the density marginal and conditional 
# Save the posterior and prior probability of each gamma
# From time to time in the Gibbs, draw a gamma completely at random
# Do the following experiment: What happens if N = 50 and the the true hypotehesis is (1,...,1)? (done)