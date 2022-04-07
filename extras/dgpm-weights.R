library(ggplot2)
library(svglite)
library(hypergeo)

# Replicate the underlying data
df <-
    expand.grid(
        j = 1:20,
        s = c(1, 2, 4),
        p = c(0.1, 0.3, 0.5)
    ) |>
    dplyr::mutate(
        w = 1 / j * choose(j + s - 2, j - 1) * p^s * (1 - p)^(j - 1) * Re(hypergeo(j + s - 1, 1, j + 1, 1 - p)),
        p_str = paste0("$\\varphi(\\bm{x}_i \\cdot \\beta) = ", p, "$"),
        s_str = as.character(s)
    )

# Replicate the plot
p <- 
    df |>
    ggplot(aes(x = j, y = w, group = s_str, linetype = s_str)) + 
    facet_grid(cols = vars(p_str)) +
    geom_line() +
    labs(x = "$j$", y = "$w_j(\\bm{x}_i)$", linetype = "s") +
    theme_classic() +
    theme(legend.position = "top")
p

# Save the plot in svg format
ggsave("dgpm-weights.svg", plot = p, width = 6 * 4 / 5, height = 3 * 4 / 5)
