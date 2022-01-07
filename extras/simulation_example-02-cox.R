library(broom)
library(dplyr)
library(ggplot2)
library(purrr)
library(survival)

simulate_sample <- function(D0, N0) {
  X0 <- sample(0:1, N0 * D0, replace = TRUE) |> matrix(nrow = N0, ncol = D0)
  c0 <- rexp(n = N0, rate = 0.1)
  y0 <- rep(0, N0)
  for (i in 1:N0) {
    if (X0[i, 5] == 1) {
      di = runif(1) < 0.6
      y0[i] = di * rlnorm(1, 0.0,  1.0) + (1 - di) * rlnorm(1, 0.3,  1.2)
    } else {
      y0[i] = rlnorm(1, 0.3, 0.7)
    }
  }
  t0 <- Surv(time = pmin(y0, c0), event = y0 < c0)
  df <- map_df(data.frame(X0), as.factor)
  df$time <- t0
  df$y0 <- y0
  return(df)
}

data <- simulate_sample(5, 500)
fit <- coxph(time ~ X1 + X2 + X3 + X4 + X5, data = data)
broom::tidy(fit)

simulate_test_results <- function(D0, N0, alpha) {
  data <- simulate_sample(D0, N0)
  fit <- coxph(time ~ X1 + X2 + X3 + X4 + X5, data = data)
  test_result <- broom::tidy(fit)$p.value < alpha
  return(test_result)
}

# Implement the experiment
set.seed(1)
results <- 
  purrr::map(1:100, ~ simulate_test_results(5, 1000, 0.05)) |>
  purrr::reduce(rbind) |> 
  colMeans()
write.csv(results, "simulation-experiment-02-coxph.csv")
