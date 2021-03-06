library(broom)
library(dplyr)
library(ggplot2)
library(purrr)
library(survival)
library(svglite)
library(glmulti)

simulate_test_results <- function(filename) {
    data <- readr::read_csv(filename, show_col_types = FALSE) |>
        dplyr::mutate(
            time = Surv(time = pmin(y0, c0), event = y0 < c0),
            across(starts_with("x"), as.factor)
        ) |>
        dplyr::select(-c(x1, c0)) |>
        dplyr::rename_with(toupper, starts_with("x"))
    best_fit <-
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
    best_subset <-
        best_fit@formulas[[1]] |>
        (\(x) as.character(x)[3]) ()  |>
        (\(x) gsub(" + X", "", x, fixed = TRUE)) () |>
        (\(x) sub("1", "", x, fixed = TRUE)) () |>
        (\(x) strsplit(x, "")[[1]]) () |>
        as.numeric()
    gamma <- rep(0, 6)
    gamma[best_subset] <- 1
    return(gamma[2:6])
}

# Get all the relevant filenames
filenames <-
    list.files(
        pattern = "simulation-example-erlang-proportional-data", 
        full.names = TRUE,
        path = "data"
    )

# Implement the experiment
set.seed(1)
results <-
    filenames |>
    purrr::map(simulate_test_results) |>
    purrr::reduce(rbind) |> 
    as.data.frame() 

# Save the results
rownames(results) <- NULL
names(results) <- paste0("x", 1:5)
write.csv(results, "data/simulation-erlang-ph-gamma-cox.csv")

# a <- |>
#     dplyr::count(V1, V2, V3, V4, V5, sort = TRUE) |> 
#     dplyr::arrange(desc(n)) |> 
#     dplyr::mutate(
#         frac = n / 100,
#         gamma = paste0("(", paste(V1, V2, V3, V4, V5, sep = ","), ")")
#     ) |> 
#     dplyr::ungroup()
# results

# # Save the results
# write.csv(results, "barplot-gamma-erlang-proportion-cox.csv")

# # Generate a barplot with the results
# p <- 
#     results |>
#     dplyr::slice(1:5) |>
#     ggplot(aes(x = reorder(gamma, frac), y = frac)) + 
#     geom_bar(stat = "identity") +
#     geom_text(aes(label = frac), hjust = -0.1) +
#     coord_flip() + 
#     theme_classic() +
#     labs(x = "$\\gamma$", y = "relative frequency")
# p

# # Save the plot in svg format
# "barplot-gamma-cox.svg" |>
#     ggsave(plot = p, width = 5 * 8 / 9, height = 3)

# #===============================================================================
# # Create the second plot for b5
# #===============================================================================

# simulate_test_results_2 <- function(filename) {
#     data <- readr::read_csv(filename, show_col_types = FALSE) |>
#         dplyr::mutate(
#             time = Surv(time = pmin(z0, c0), event = z0 < c0),
#             across(starts_with("x"), as.factor)
#         ) |>
#         dplyr::select(-c(x1, z0, c0)) |>
#         dplyr::rename_with(toupper, starts_with("x"))
#     cox_fit <- coxph(time ~ X2 + X3 + X4 + X5 + X6, data = data)
#     b5 <- 
#         broom::tidy(cox_fit) |>
#         dplyr::filter(term == "X61") |>
#         dplyr::pull(estimate)
#     return(b5)
# }

# results <-
#     list.files("data-survival-input-data", full.names = TRUE) |>
#     purrr::map_dbl(simulate_test_results_2)

# p_b5_fitted <- 
#     data.frame(x = 1:100, y = results) |>
#     ggplot(aes(x = x, y = y)) +
#     geom_point() +
#     theme_classic() + 
#     labs(x = "iter", y = "$\\hat{\\beta}_5$")

# p_b5_true <- 
#     data.frame(x = seq(0, 6, length.out = 50), y = get_true_beta5()) |>
#     ggplot(aes(x = x, y = y)) +
#     geom_line() +
#     theme_classic() +
#     labs(x = "$z_i$", y = "log-hazard ratio for $x_{i5}$ (1 vs 0)")

# p_b5_fitted
# p_b5_true
# combined_plot <-
#     ggarrange(p_b5_fitted, p_b5_true, labels = c("A", "B"), ncol = 2, nrow = 1)

# # Save the plot in svg format
# "barplot-gamma-cox-b5.svg" |>
#     ggsave(plot = combined_plot, width = 5 * 8 / 9, height = 3)
