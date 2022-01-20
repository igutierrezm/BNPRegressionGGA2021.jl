library(betareg)
library(broom)
library(dplyr)
library(glmulti)
library(purrr)
library(readr)
set.seed(1)

get_best_gamma <- function(data) {
    best_fit <-
        glmulti::glmulti(
            data = data,
            y0 ~ x2 + x3 + x4 + x5 + x6,
            level = 1,              # No interaction considered
            method = "h",           # Exhaustive approach
            crit = "aic",           # AIC as criteria
            confsetsize = 1,        # Keep 5 best models
            plotty = FALSE,         # No plot
            report = FALSE,         # No interim reports
            fitfunction = "betareg" # fit function
        )

    best_subset <-
        best_fit@formulas[[1]] |>
        (\(x) as.character(x)[3]) ()  |>
        (\(x) gsub(" + x", "", x, fixed = TRUE)) () |>
        (\(x) sub("1", "", x, fixed = TRUE)) () |>
        (\(x) strsplit(x, "")[[1]]) () |>
        as.numeric()
    gamma <- rep(0, 6)
    gamma[best_subset] <- 1
    return(gamma[2:6])
}

# Get the simulation results
results <-
    "data/simulated-data-beta-beta.csv" |>
    readr::read_csv() |>
    dplyr::select(-x1) |>
    dplyr::group_by(iter) |>
    dplyr::group_map(~ get_best_gamma(.x)) |>
    purrr::reduce(rbind) |>
    as.data.frame()

# Save the results
rownames(results) <- NULL
colnames(results) <- paste0("x", 2:6)
write.csv(results, "data/simulated-best-gamma-beta-beta-freq.csv", row.names = FALSE)

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



    # data <- readr::read_csv(filename, show_col_types = FALSE) |>
    #     dplyr::mutate(
    #         time = Surv(time = pmin(y0, c0), event = y0 < c0),
    #         across(starts_with("x"), as.factor)
    #     ) |>
    #     dplyr::select(-c(x1, c0)) |>
    #     dplyr::rename_with(toupper, starts_with("x"))
