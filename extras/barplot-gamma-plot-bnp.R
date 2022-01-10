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
    dplyr::count(V2, V3, V4, V5, V6) |>
    dplyr::arrange(desc(n)) |>
    dplyr::mutate(
        frac = n / sum(n),
        frac_cumsum = cumsum(frac),
        gamma = paste0("(", paste(V2, V3, V4, V5, V6, sep = ","), ")")
    ) |>
    dplyr::filter(frac_cumsum <= 0.9) |>
    dplyr::select(gamma, frac)

# Add a remainer
counts <-
    counts |>
    dplyr::add_row(gamma = "others", frac = 1 - sum(counts$frac))

# Create the plot
p <- 
    counts |>
    ggplot(aes(x = reorder(gamma, frac - (gamma == "others")), y = frac)) + 
    geom_bar(stat = "identity") +
    geom_text(aes(label = frac), hjust = -0.1) +
    coord_flip() + 
    theme_classic() +
    labs(x = "$\\gamma$", y = "relative frequency")
p

# Save the plot in svg format
"barplot-gamma-bnp.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3)
