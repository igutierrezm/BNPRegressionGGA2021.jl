library(ggplot2)
library(readr)
library(tidyr)

# Load the data
df <- readr::read_csv("data/walker-example.csv")

# Munge the data
df <- 
    df |> 
    dplyr::mutate(x = factor(x)) |>
    tidyr::pivot_longer(c(fh, f0))
head(df)

# Plot the conditional posterior predictive densities
p <- 
    df |> 
    ggplot2::ggplot(ggplot2::aes(x = y, y = value, color = name)) + 
    ggplot2::facet_grid(row = ggplot2::vars(x)) +
    ggplot2::geom_line()
p

# Save the plot in svg format
"figures/dgp-normal-normal-example1-500.pdf" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3)
