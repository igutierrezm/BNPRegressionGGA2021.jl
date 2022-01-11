library(broom)
library(dplyr)
library(ggplot2)
library(purrr)
library(survival)
library(svglite)
library(glmulti)
library(ggpubr)

# Simulate a dataset from the DGP described in 5.2
simulate_sample <- function(D0, N0) {
  X0 <- sample(0:1, N0 * D0, replace = TRUE) |> matrix(nrow = N0, ncol = D0)
  c0 <- rexp(n = N0, rate = 0.1)
  y0 <- rep(0, N0)
  for (i in 1:N0) {
    if (X0[i, 5] == 1) {
      di = runif(1) < 0.6
      y0[i] = di * rlnorm(1, 0.0,  1.0) + (1 - di) * rlnorm(1, 0.3,  1.2)
    } else {
      y0[i] = rlnorm(1, 0.3, 0.7)
    }
  }
  t0 <- Surv(time = pmin(y0, c0), event = y0 < c0)
  df <- map_df(data.frame(X0), as.factor)
  df$time <- t0
  df$y0 <- y0
  return(df)
}

get_true_beta5 <- function() {
    z0 <- seq(0, 6, length.out = 50)
    f0 <- dlnorm(z0, 0.3, 0.7)
    F0 <- plnorm(z0, 0.3, 0.7)
    f1 <- 0.6 * dlnorm(z0, 0.0,  1.0) + 0.4 * dlnorm(z0, 0.3,  1.2)
    F1 <- 0.6 * plnorm(z0, 0.0,  1.0) + 0.4 * plnorm(z0, 0.3,  1.2)
    h0 <- f0 / (1 - F0)
    h1 <- f1 / (1 - F1)
    b5 <- log(h1 / h0)
    return(b5)    
}

b5 <- get_true_beta5()

simulate_test_results <- function(filename) {
    data <- readr::read_csv(filename, show_col_types = FALSE) |>
        dplyr::mutate(
            time = Surv(time = pmin(z0, c0), event = z0 < c0),
            across(starts_with("x"), as.factor)
        ) |>
        dplyr::select(-c(x1, z0, c0)) |>
        dplyr::rename_with(toupper, starts_with("x"))
    glmulti.coxph.out <-
        glmulti::glmulti(
            data = data,
            time ~ X2 + X3 + X4 + X5 + X6,
            level = 1,            # No interaction considered
            method = "h",         # Exhaustive approach
            crit = "aic",         # AIC as criteria
            confsetsize = 1,      # Keep 5 best models
            plotty = FALSE,       # No plot
            report = FALSE,       # No interim reports
            fitfunction = "coxph" # coxph function
        )
    relevant_covariantes <-
        glmulti.coxph.out@formulas[[1]] |>
        (\(x) as.character(x)[3]) ()  |>
        (\(x) gsub(" + X", "", x, fixed = TRUE)) () |>
        (\(x) sub("1", "", x, fixed = TRUE)) () |>
        (\(x) strsplit(x, "")[[1]]) () |>
        as.numeric()
    gamma <- rep(0, D0)
    gamma[relevant_covariantes] <- 1
    return(gamma)
}

# Implement the experiment
set.seed(1)
results <-
    list.files("data-survival-input-data", full.names = TRUE) |>
    purrr::map(simulate_test_results) |>
    purrr::reduce(rbind) |> 
    as.data.frame() |>
    dplyr::count(V1, V2, V3, V4, V5, sort = TRUE) |> 
    dplyr::arrange(desc(n)) |> 
    dplyr::mutate(
        frac = n / 100,
        gamma = paste0("(", paste(V1, V2, V3, V4, V5, sep = ","), ")")
    ) |> 
    dplyr::ungroup()

# Save the results
write.csv(results, "barplot-gamma-cox-data.csv")

# Generate a barplot with the results
p <- 
    results |>
    dplyr::slice(1:5) |>
    ggplot(aes(x = reorder(gamma, frac), y = frac)) + 
    geom_bar(stat = "identity") +
    geom_text(aes(label = frac), hjust = -0.1) +
    coord_flip() + 
    theme_classic() +
    labs(x = "$\\gamma$", y = "relative frequency")
p

# Save the plot in svg format
"barplot-gamma-cox.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3)

#===============================================================================
# Create the second plot for b5
#===============================================================================

simulate_test_results_2 <- function(filename) {
    data <- readr::read_csv(filename, show_col_types = FALSE) |>
        dplyr::mutate(
            time = Surv(time = pmin(z0, c0), event = z0 < c0),
            across(starts_with("x"), as.factor)
        ) |>
        dplyr::select(-c(x1, z0, c0)) |>
        dplyr::rename_with(toupper, starts_with("x"))
    cox_fit <- coxph(time ~ X2 + X3 + X4 + X5 + X6, data = data)
    b5 <- 
        broom::tidy(cox_fit) |>
        dplyr::filter(term == "X61") |>
        dplyr::pull(estimate)
    return(b5)
}

results <-
    list.files("data-survival-input-data", full.names = TRUE) |>
    purrr::map_dbl(simulate_test_results_2)

p_b5_fitted <- 
    data.frame(x = 1:100, y = results) |>
    ggplot(aes(x = x, y = y)) +
    geom_point() +
    theme_classic() + 
    labs(x = "iter", y = "$\\hat{\\beta}_5$")

p_b5_true <- 
    data.frame(x = seq(0, 6, length.out = 50), y = get_true_beta5()) |>
    ggplot(aes(x = x, y = y)) +
    geom_line() +
    theme_classic() +
    labs(x = "$z_i$", y = "log-hazard ratio for $x_{i5}$ (1 vs 0)")

p_b5_fitted
p_b5_true
combined_plot <-
    ggarrange(p_b5_fitted, p_b5_true, labels = c("A", "B"), ncol = 2, nrow = 1)

# Save the plot in svg format
"barplot-gamma-cox-b5.svg" |>
    ggsave(plot = combined_plot, width = 5 * 8 / 9, height = 3)
