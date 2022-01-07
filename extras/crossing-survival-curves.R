library(ggplot2)
library(svglite)

# Replicate the underlying data
df <-
  expand.grid(
    zi = seq(from = 0, to = 6, length.out = 50),
    xi5 = 0:1
  ) |>
  dplyr::mutate(
    Szi0 = 1 - plnorm(zi, 0.3, 0.7),
    Szi1 = 1 - 0.6 * plnorm(zi, 0.0, 1.0) - 0.4 * plnorm(zi, 0.3, 1.2),
    Szi = ifelse(xi5, Szi1, Szi0),
    xi5_str = as.character(xi5)
  )

# Replicate the plot
p <- 
  df |>
  ggplot(aes(x = zi, y = Szi, group = xi5_str, linetype = xi5_str)) + 
  geom_line() +
  labs(x = "$z_i$", y = "survival function", linetype = "$x_{i5}$") +
  theme_classic() +
    theme(legend.position = "top")
p

# Save the plot in svg format
"crossing-survival-curves.svg" |>
    ggsave(plot = p, width = 5 * 3 / 5, height = 3 * 4 / 5)
