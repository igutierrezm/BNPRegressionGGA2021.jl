library(ggplot2)
library(svglite) # Run 'apt -y install libfontconfig1-dev' before
library(readr)
library(tidyr)

df <- read_csv("data/simulation-example-normal-gamma.csv")
names(df) <- paste0("V", 1:6)
head(df)

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
    dplyr::slice(1:5) |>
    dplyr::select(gamma, frac)

# Create the plot
p <- 
    counts |>
    ggplot(aes(x = reorder(gamma, frac), y = frac)) + 
    geom_bar(stat = "identity") +
    geom_text(aes(label = frac), hjust = -0.1) +
    ylim(0, 1) +
    coord_flip() + 
    theme_classic() +
    labs(x = "$\\gamma$", y = "relative frequency")
p

# Save the plot in svg format
"figures/barplot-normal-gamma-bnp.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3)
