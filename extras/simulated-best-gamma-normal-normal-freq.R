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

# Get the simulation results
results <-
    "data/simulated-data-normal-normal.csv" |>
    readr::read_csv() |>
    dplyr::select(-x1) |> 
    dplyr::group_by(iter) |>
    dplyr::group_map(~ get_best_gamma(.x)) |>
    purrr::reduce(rbind) |>
    as.data.frame()

# Save the results
rownames(results) <- NULL
colnames(results) <- paste0("x", 2:6)
write.csv(results, "data/simulated-best-gamma-normal-normal-freq.csv", row.names = FALSE)
