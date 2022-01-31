library(ggplot2)
library(readr)
library(tidyr)

# Load the data
df <- readr::read_csv("data/dgp-normal-normal-example2.csv")

# Munge the data
df <- 
    df |> 
    dplyr::mutate(x = factor(x)) # |>
    # tidyr::pivot_longer(c(fh, f0))
head(df)

# Plot the conditional posterior predictive densities
df |> 
    ggplot2::ggplot(ggplot2::aes(x = y, y = fh)) + 
    ggplot2::facet_grid(row = ggplot2::vars(x)) +
    ggplot2::geom_line()
