Sys.setenv(JAVA_HOME = "/usr/lib/jvm/java-11-openjdk-amd64/bin/java")
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
            y0 ~ .,
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
    gamma <- rep(0, 11)
    gamma[best_subset] <- 1
    return(gamma[2:11])
}

# Example 1 --------------------------------------------------------------------

# Get the simulation results
results <-
    "data/encargo-2022-03-31-data-ex1.csv" |>
    readr::read_csv() |>
    dplyr::select(-x1) |> 
    dplyr::group_by(iter) |>
    dplyr::group_map(~ get_best_gamma(.x)) |>
    purrr::reduce(rbind) |>
    as.data.frame()

# Save the results
rownames(results) <- NULL
colnames(results) <- paste0("x", 2:6)
write.csv(
    results,
    "simulated-best-normal-freq-ex-01-2022-03-31.csv",
    row.names = FALSE
)

# Example 2 --------------------------------------------------------------------

# Get the simulation results
results <-
    "data/encargo-2022-03-31-data-ex2.csv" |>
    readr::read_csv() |>
    dplyr::select(-x1) |> 
    dplyr::group_by(iter) |>
    dplyr::group_map(~ get_best_gamma(.x)) |>
    purrr::reduce(rbind) |>
    as.data.frame()

# Save the results
rownames(results) <- NULL
colnames(results) <- paste0("x", 2:6)
write.csv(
    results,
    "simulated-best-normal-freq-ex-02-2022-03-31.csv",
    row.names = FALSE
)

# Example 3 --------------------------------------------------------------------

# Get the simulation results
results <-
    "data/encargo-2022-03-31-data-ex3.csv" |>
    readr::read_csv() |>
    dplyr::select(-x1) |> 
    dplyr::group_by(iter) |>
    dplyr::group_map(~ get_best_gamma(.x)) |>
    purrr::reduce(rbind) |>
    as.data.frame()

# Save the results
rownames(results) <- NULL
colnames(results) <- paste0("x", 2:6)
write.csv(
    results,
    "simulated-best-normal-freq-ex-03-2022-03-31.csv",
    row.names = FALSE
)

# Example 4 --------------------------------------------------------------------

# Get the simulation results
results <-
    "data/encargo-2022-03-31-data-ex4.csv" |>
    readr::read_csv() |>
    dplyr::select(-x1) |> 
    dplyr::group_by(iter) |>
    dplyr::group_map(~ get_best_gamma(.x)) |>
    purrr::reduce(rbind) |>
    as.data.frame()

# Save the results
rownames(results) <- NULL
colnames(results) <- paste0("x", 2:6)
write.csv(
    results,
    "simulated-best-normal-freq-ex-04-2022-03-31.csv",
    row.names = FALSE
)
