library(BhGLM)
library(sets)
library(dplyr)

# Simulate data from an exponential regression model
simulate_exponential_data <- function(n, k) {
    b <- c(1, rep(0, k - 1))
    status <- rep(TRUE, n)
    x <- runif(n * k) |> matrix(ncol = k)
    y <- exp(x %*% b) |> rexp() |> survival::Surv(status)
    out <- data.frame(
        y = y, 
        x1 = x[, 1],
        x2 = x[, 2],
        x3 = x[, 3],
        x4 = x[, 4],
        x5 = x[, 5]
    )
    return(out)
}

data <- simulate_exponential_data(1000, 10)
# survival::coxph(y ~ ., data)

bss_coxph <- function(formula, data) {
    xvars <- all.vars(terms(y ~ ., data = data))[-1]
}

a <- c("a", "b", "c")
b <- sets::set_power(a)
lapply(b, function(.x) .x[[1]])
str(b)

# An example where the true model is the null model
set.seed(1234)
lambda0 <- BhGLM::glmNet(data$x, data$y, family = "cox", ncv = 10)$prior.scale
fit <- BhGLM::bmlasso(data$x, data$y, family = "cox", ss = c(lambda0, 0.5))
fit1$coef

fit <- survival::coxph.fit(data$x, data$y, strata = NULL)
fit1$coef


# png("Rplots.png")
# plot.bh(coef = fit$coef, threshold = 10, gap = 100)
# dev.off()
# An example where the only relevant variable is x1
set.seed(1234)
K <- 5
N <- 100
x <- runif(N * K) |> matrix(ncol = 5)
y <- rgamma(N, shape = 1.0, rate = 1.0)
status <- rep(TRUE, N)
ysurv <- survival::Surv(y, status)
fit <- BhGLM::bmlasso(x, ysurv, family = "cox")
summary(fit)
fit$coefs

set.seed(1234)
data <- simulate_exponential_data(1000, 5)
df <-
    data.frame(
        y  = data$y,
        x1 = data$x[, 1],
        x2 = data$x[, 2],
        x3 = data$x[, 3],
        x4 = data$x[, 4],
        x5 = data$x[, 5]
    )
fitted_best_model <-
    glmulti::glmulti(
        y ~ .,
        data = df,
        level = 1,                    # No interaction considered
        method = "h",                 # Exhaustive approach
        crit = "aic",                 # AIC as criteria
        confsetsize = 1,              # Keep best model
        plotty = F, report = F,       # No plot or interim reports
        fitfunction = survival::coxph # coxph function
    )
best_model_trues <- 
    names(fitted_best_model@objects[[1]]$coefficients) %>%
        gsub("x", "", .) %>%
        as.integer()
best_model_trues

best_gamma <- rep(0L, 5)
best_gamma[best_model_trues] <- 1L
best_gamma
