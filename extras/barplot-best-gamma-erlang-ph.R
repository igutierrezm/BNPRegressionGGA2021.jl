library(dplyr)
library(ggplot2)

# Get the frequentist results
freq <-
    "data/simulated-best-gamma-erlang-ph-freq.csv" |>
    readr::read_csv() |>
    dplyr::select(-dplyr::one_of("x1")) |> 
    dplyr::mutate(method = "frequentist")

# Get BNP results
bnp <-
    "data/simulated-best-gamma-erlang-ph-bnp.csv" |>
    readr::read_csv() |>
    dplyr::select(-dplyr::one_of("x1")) |> 
    dplyr::mutate(dplyr::across(dplyr::starts_with("x"), as.numeric)) |>
    dplyr::mutate(method = "bnp")

# Get the combined data
data <- rbind(freq, bnp)

# Get the relevant data for the barplot
best_gamma_ranking <- 
    data |>
    dplyr::count(method, x2, x3, x4, x5, x6, sort = TRUE) |>
    dplyr::mutate(
        frac = n / 100,
        method = factor(method),
        gamma = paste0("(", paste(x2, x3, x4, x5, x6, sep = ","), ")")
    ) |> 
    dplyr::group_by(method) |>
    dplyr::slice(1:5) |>
    dplyr::mutate(gamma = paste0(strrep(" ", as.numeric(method) - 1), gamma))

# Get the plot
p <- 
    best_gamma_ranking |> 
    ggplot2::ggplot(aes(x = reorder(gamma, frac), y = frac)) + 
    ggplot2::geom_bar(stat = "identity") +
    ggplot2::facet_wrap(vars(method), scales = "free_y", nrow = 2) +
    geom_text(aes(label = sprintf("%1.2f", frac)), hjust = -0.1) +
    ylim(0, 1) +
    coord_flip() + 
    theme_classic() +
    labs(x = "$\\gamma$", y = "relative frequency")
p

# Save the plot in svg format
"figures/barplot-gamma-erlang-ph.svg" |>
    ggsave(plot = p, width = 5 * 8 / 9, height = 3)
