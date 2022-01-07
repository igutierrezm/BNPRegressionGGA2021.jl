library(ggplot2)
library(svglite)
library(readr)
library(tidyr)

# Load the data (created in Julia)
df <- 
    readr::read_csv("data/crossing-survival-curves-fitted.csv") |>
    tidyr::pivot_longer(c(St, Sh), names_to = "variable", values_to = "value") |>
    dplyr::mutate(
        x5_str = ifelse(x5, "$x_{i5} = 1$", "$x_{i5} = 0$"),
        variable = ifelse(variable == "Sh", "posterior mean", "true value")
    )
df

# Create the plot
p <- 
    df |>
    ggplot(aes(x = z)) + 
    geom_line(aes(y = value, linetype = variable)) +
    geom_ribbon(aes(ymin = lb, ymax = ub), fill = "black", alpha = 0.2) +
    labs(x = "$z_i$", y = "Survival function", linetype = "") +
    facet_grid(cols = vars(x5_str)) + 
    theme_classic() +
    theme(legend.position = "top")
p

# Save the plot in svg format
"crossing-survival-curves-fit.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3 * 4 / 5)
