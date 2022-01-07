library(ggplot2)
library(svglite)
library(readr)
library(tidyr)

# # Simulate the results
# df <- 
#     sample(0:1, size = 100 * 5, replace = TRUE) |>
#     matrix(nrow = 100, ncol = 5) |>
#     as.data.frame()

df <- read_csv("barplot-gamma.csv")
names(df) <- paste0("V", 1:6)

# Count the data
counts <- 
    df |> 
    purrr::map_df(as.integer) |>
    dplyr::count(V1, V2, V3, V4, V5, V6) |>
    dplyr::arrange(desc(n)) |>
    dplyr::mutate(
        frac = n / sum(n),
        frac_cumsum = cumsum(frac),
        gamma = paste0(V1, V2, V3, V4, V5, V6)
    ) |>
    dplyr::filter(frac_cumsum <= 0.8) |>
    dplyr::select(gamma, frac)

# Create the plot
p <- 
    counts |>
    ggplot(aes(x = reorder(gamma, frac), y = frac)) + 
    geom_bar(stat = "identity") +
    coord_flip() + 
    theme_classic() +
    labs(x = "$\\gamma$ (as a string of 0/1 values)", y = "relative frequency")
p

# Save the plot in svg format
"crossing-survival-curves-fit.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3 * 4 / 5)
