# Sys.setenv(http_proxy="http://iegutierrezm:IG17026903-9@astaro.ine.cl:8080")
# Sys.setenv(https_proxy="http://iegutierrezm:IG17026903-9@astaro.ine.cl:8080")
# remotes::install_github("nyiuab/BhGLM", force = TRUE, build_vignettes = TRUE)

# library(BhGLM)
# library(dplyr)
# library(purrr)
# library(dplyr)

get_all_subsets <- function(x) {
  out <- 
    1:length(x) %>%
    purrr::map(
      ~ utils::combn(x, .x, simplify = FALSE) %>%
        map(~ paste0(.x, collapse = " + "))
    ) %>%
    unlist(recursive = TRUE)
  return(out)
}

get_gamma <- function(x, k) {
  xvars <- all.vars(as.formula(x))[-1]
  trues <- xvars %>% gsub("x", "", .) %>% as.integer()
  gamma <- rep(0L, k)
  gamma[trues] <- 1L
  out <- data.frame(t(gamma))
  names(out) <- paste0("g", 1:k)
  return(out)
}

# Simulate data from an exponential regression model
simulate_exponential_data <- function(n, k) {
  b <- c(1, rep(0, k - 1))
  status <- rep(TRUE, n)
  x <- runif(n * k) %>% matrix(ncol = k)
  y <- exp(x %*% b) %>% rexp() %>% survival::Surv(status)
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

bss_coxph <- function(formula, data) {
  # Construct a list which all possible submodels
  varlist <- all.vars(terms(formula, data = data))
  xvars <- varlist[-1]
  yvar <- varlist[1]
  submodels <- 
    paste(yvar, get_all_subsets(xvars), sep = " ~ ")
  # Compute the aic for each submodel
  aics <- 
    submodels %>%
    purrr::map_dbl(
      function(.x) {
        fit <- survival::coxph(as.formula(.x), data)
        out <- 2 * (length(fit$assign) - fit$loglik[2])
        return(out)
      }
  )
  # Select the best model
  best_submodel <- submodels[which.min(aics)]
  
  # Compute gamma
  out <- get_gamma(best_submodel, length(xvars))
  return(out)
}

# Complete Example -------------------------------------------------------------

set.seed(1)
out <- 
  1:100 %>%
  purrr::map_df(
    function(x) {
      data <- simulate_exponential_data(100, 5)
      gamma <- bss_coxph(y ~ ., data)
      return(gamma)
    }
  )

#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



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