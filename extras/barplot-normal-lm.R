library(broom)
library(dplyr)
library(ggplot2)
library(purrr)
library(survival)
library(svglite)
library(glmulti)

simulate_test_results <- function(filename) {
    data <-
        filename |>
        readr::read_csv(show_col_types = FALSE) |>
        dplyr::select(-x1) |> 
        dplyr::mutate(
            dplyr::across(dplyr::starts_with("x"), as.factor)
        )

    best_fit <-
        glmulti::glmulti(
            data = data,
            y0 ~ x2 + x3 + x4 + x5 + x6,
            level = 1,         # No interaction considered
            method = "h",      # Exhaustive approach
            crit = "aic",      # AIC as criteria
            confsetsize = 1,   # Keep the best models
            plotty = FALSE,    # No plot
            report = FALSE,    # No interim reports
            fitfunction = "lm" # coxph function
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

# Get all the relevant filenames
filenames <-
    list.files(
        pattern = "simulation-example-normal-data", 
        full.names = TRUE,
        path = "data"
    )

# Get the simulation results
set.seed(1)
results <-
    filenames |>
    purrr::map(simulate_test_results) |>
    purrr::reduce(rbind) |> 
    as.data.frame() |>
    dplyr::count(V1, V2, V3, V4, V5, sort = TRUE) |> 
    dplyr::arrange(desc(n)) |> 
    dplyr::mutate(
        frac = n / 100,
        gamma = paste0("(", paste(V1, V2, V3, V4, V5, sep = ","), ")")
    ) |> 
    dplyr::ungroup() |>
    dplyr::select(-dplyr::starts_with("V"), n)

# Save the results
write.csv(results, "data/map-gamma-normal.csv")

# Generate a barplot with the results
p <- 
    results |>
    dplyr::slice(1:5) |>
    ggplot(aes(x = reorder(gamma, frac), y = frac)) + 
    geom_bar(stat = "identity") +
    geom_text(aes(label = frac), hjust = -0.1) +
    ylim(0, 1) +
    coord_flip() + 
    theme_classic() +
    labs(x = "$\\gamma$", y = "relative frequency")
p

# Save the plot in svg format
"figures/barplot-normal-gamma-bestsubset.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3)
