



#' interaction detection for longitudinal outcomes using fused lasso
#'
#' @param x list of design matrices; one for each point in time. If a matrix is given, the design matrix is assumed to
#' be constant over time
#' Each row is an observation, each column corresponds to a covariate
#' @param y list of numeric response vectors of length nobs(t); one element with a response vector for time t.
#' @param trt list of length equal to the number of time periods. each element of trt is a vector of treatment indicators
#' @param family "gaussian" for least squares problems, "binomial" for binary response.
#' "coxph" for time-to-event outcomes
#' @param weights observation weights
#' @param type.measure  one of c("mse","deviance","class","auc","mae") to be used for cross validation
#' @param lambda tuning parameter values for lasso penalty
#' @param gamma ratio of fused lasso to lasso tuning parameter
#' @param nlambda number of tuning parameter values - default is 100.
#' @param lasso.penalize list of length equal to the number of time periods. each element of lasso.penalize is a vector
#' of length equal to the number of variables in x[[t]] (design matrix at time t) with values either 0 or 1 where a 1
#' in the jth position indicates that the jth variable will be penalized with the lasso and 0 otherwise. Defaults to all
#' variables being penalized with the lasso.
#' @param nfolds number of folds for cross validation. If given value 0, 1, or NULL, no cross validation will be performed
#' @param foldid an optional vector of values between 1 and \code{nfolds} specifying which fold each observation belongs to.
#' @param boot logical, whether or not to perform bootstrap for benefit confidence intervals. default is FALSE
#' for no bootstrap computation
#' @param B integer number of resamples for bootstrap - default is 100
#' @param boot.type one of \code{"replacement"}, \code{"mofn"}, specifies what type of bootstrap to use: with replacement (standard)
#' or m-out-of-n bootstrap, in which case the number of samples are taken to be the integer part of
#' n^\code{m.frac}, where the user specifies \code{m.frac} as some number strictly between 0 and 1. Samples
#' are taken without replacemet for the m-out-of-n bootstrap
#' @param m.frac scalar number strictly between 0 and 1,
#' @param parallel boolean indicator of whether or not to utilize parallel computation for cross validation
#' @param abs.tol absolute tolerance for convergence for ADMM. Defaults to 1e-5
#' @param rel.tol relative tolerance for convergence for ADMM. Defaults to 1e-5
#' @param maxit maximum number of ADMM iterations
#' @param maxit.cv maximum number of ADMM iterations for cross validation runs
#' @param rho scalar positive number value. This is the ADMM hyperparameter. If unspecified,
#' code will default to a reasonable choice. Bad values of rho may lead to extraordinarily slow
#' convergence of ADMM
#' @param ... other arguments to be passed to cv.fusedlasso
#' @return An object with S3 class "subgroupLong"
#' @useDynLib personalizedLong
#' @import Rcpp
#' @import Matrix
#' @import foreach
#' @export
#' @examples
#' set.seed(123)
#' nobs       <- 100
#' nobs.test  <- 1e5
#' nvars      <- 20
#' periods    <- 6
#' sd         <- 2
#'
#' beta.nz <- rbind(c( 1,    1,    1,    1.5,  1.5,  1.5),
#'                  c(-1,   -1,   -1,   -0.5, -0.5, -0.5),
#'                  c( 1,    1,    1,   -1,   -1,   -1),
#'                  c( 1,    1,    1,    1,    1,    1),
#'                  c(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5))
#'
#' beta <- data.matrix(rbind(beta.nz, matrix(0, nvars - 5, periods)))
#'
#' trt <- rbinom(nobs, 1, 0.5)
#' x   <- matrix(rnorm(nobs * nvars), ncol = nvars); colnames(x) <- paste0("V", 1:ncol(x))
#' y   <- x %*% (beta * 0.5) + (2 * trt - 1) * (x %*% beta) +
#'     matrix(rnorm(nobs * periods, sd = sd), ncol = periods)
#' y   <- apply(y, 2, function(yy) yy - (2 * trt - 1))
#'
#' plot(x = NULL, xlim = c(1,6), ylim = range(y))
#' for (i in 1:nobs)
#' {
#'     lines(x = 1:6, y = y[i,], col = colors()[i+1])
#' }
#'
#' x.list <- rep(list(x), periods)
#' y.list <- lapply(apply(y, 2, function(x) list(x) ), function(x) x[[1]])
#'
#' fit <- subgroupLong(x = x.list, y = y.list, trt = trt, gamma = c(0.05, 0.1, 1, 2, 5, 10))
#'
#' round(matrix(fit$cv.model$best.cv.fit$beta, ncol = 6), 4)
#'
#' bfit <- subgroupLong(x = x.list, y = y.list,
#'                           trt = trt, gamma = c(0.05, 0.1, 1),
#'                           boot = TRUE, B = 50L)
#'
#' plot(bfit)
#'
#' lapply(bfit$boot.res, function(x) round(colMeans(x, na.rm = TRUE), 4) )
#' ## bootstrap CI
#' CIs <- lapply(bfit$boot.res, function(x) apply(x, 2, function(xx)
#'      quantile(xx, probs = c(0.025, 0.975), na.rm = TRUE)) )
#'
#' D <- lapply(1:periods, function(i) 1 * (drop(x.list[[i]] %*%  beta[,i]) > 0) )
#' res.vec <- numeric(11L)
#' names(res.vec) <- colnames(bfit$boot.res[[1]])
#' oracle.results <- rep(list(res.vec), periods)
#' for (t in 1:periods)
#' {
#'      y.cur <- y.list[[t]]
#'      D.cur <- D[[t]]
#'      weights.cur <- rep(1, length(y.cur))
#'      # Emprical average among all
#'      oracle.results[[t]][1] <- sum(y.cur * weights.cur) / sum(weights.cur)
#'
#'      # Fit statistics among Trt=D
#'      sub.agree                <- trt == D.cur
#'      oracle.results[[t]][2] <- sum(y.cur[sub.agree] * weights.cur[sub.agree]) /
#'          sum(weights.cur[sub.agree])
#'
#'      # Fit statistics among Trt!=D
#'      sub.disagree             <- trt != D.cur
#'      oracle.results[[t]][3] <- sum(y.cur[sub.disagree] * weights.cur[sub.disagree]) /
#'          sum(weights.cur[sub.disagree])
#'
#'      # Fit statistics among Trt=D=1
#'      sub.11                   <- (trt == 1) & (D.cur == 1)
#'      oracle.results[[t]][4] <- sum(y.cur[sub.11] * weights.cur[sub.11]) /
#'          sum(weights.cur[sub.11])
#'
#'      # Fit statistics among Trt=D=0
#'      sub.00                   <- (trt == 0) & (D.cur == 0)
#'      oracle.results[[t]][5] <- sum(y.cur[sub.00] * weights.cur[sub.00]) /
#'          sum(weights.cur[sub.00])
#'
#'      # Fit statistics among Trt=0, D=1
#'      sub.01                   <- (trt == 0) & (D.cur == 1)
#'      oracle.results[[t]][6] <- sum(y.cur[sub.01] * weights.cur[sub.01]) /
#'          sum(weights.cur[sub.01])
#'
#'      # Fit statistics among Trt=1, D=0
#'      sub.10                   <- (trt == 1) & (D.cur == 0)
#'      oracle.results[[t]][7] <- sum(y.cur[sub.10] * weights.cur[sub.10]) /
#'          sum(weights.cur[sub.10])
#'
#'      # Fit statistics among D=1
#'      sub.x1                   <- (D.cur == 1)
#'      oracle.results[[t]][8] <- sum(y.cur[sub.x1] * weights.cur[sub.x1]) /
#'          sum(weights.cur[sub.x1])
#'
#'      # Fit statistics among D=0
#'      sub.x0                   <- (D.cur == 0)
#'      oracle.results[[t]][9] <- sum(y.cur[sub.x0] * weights.cur[sub.x0]) /
#'          sum(weights.cur[sub.x0])
#'
#'      # Fit statistics among Trt=1
#'      sub.1x                   <- (trt == 1)
#'      oracle.results[[t]][10] <- sum(y.cur[sub.1x] * weights.cur[sub.1x]) /
#'          sum(weights.cur[sub.1x])
#'
#'      # Fit statistics among Trt=0
#'      sub.0x                   <- (trt == 0)
#'      oracle.results[[t]][11] <- sum(y.cur[sub.0x] * weights.cur[sub.0x]) /
#'          sum(weights.cur[sub.0x])
#' }
#'
#'
subgroupLong <- function(x,
                         y,
                         trt,
                         family         = c("gaussian", "binomial", "coxph"),
                         #method   = c("tian", "owl"),
                         weights        = NULL,
                         type.measure   = c("mse","deviance","class","auc","mae"),
                         lambda         = numeric(0),
                         lasso.penalize = NULL,
                         gamma          = 1,
                         nlambda        = 100L,
                         nfolds         = 10L,
                         foldid         = NULL,
                         boot           = FALSE,
                         B              = 100L,
                         boot.type      = c("replacement", "mofn"),
                         m.frac         = 0.9,
                         parallel       = FALSE,
                         abs.tol        = 1e-5,
                         rel.tol        = 1e-5,
                         maxit          = 250,
                         maxit.cv       = 250,
                         rho            = NULL,
                         ...)
{
    family       <- match.arg(family)
    #method <- match.arg(method)
    type.measure <- match.arg(type.measure)
    boot.type    <- match.arg(boot.type)

    if (family != "gaussian")
    {
        stop(paste(family, "family not implemented yet."))
    }

    m.frac <- as.double(m.frac[1])

    if (m.frac >= 1 || m.frac <= 0) stop("m.frac must be between 0 and 1")

    #dim.y <- dim(y)
    #if (is.null(dim.y))
    #{
    #    stop("y must be a matrix with at least 2 columns. Each column corresponds to
    #         a different time point.")
    #}
    #nperiods <- dim.y[2]

    if (!is.list(y))
    {
        stop("y must be a list with length equal to the number of time points. Each list item corresponds to
             the response vector for a different time point.")
    } else
    {
        nperiods  <- length(y)
        nobs.vec  <- numeric(nperiods)
        nvars.vec <- numeric(nperiods)
        for (t in 1:nperiods)
        {
            nobs.vec[t] <- NROW(y[[t]])
            y[[t]]      <- drop(y[[t]])
        }
    }
    total.nobs <- sum(nobs.vec)

    ## check to make sure trt is valid
    if (!is.list(trt))
    {
        if (!all(nobs.vec == NROW(trt)))
        {
            stop("trt must be same length as each element in y list.")
        }
        trt <- rep(list(trt), nperiods)
    } else
    {
        if (length(trt) != nperiods)
        {
            stop("trt must be a list with same length as y")
        }
        for (t in 1:nperiods)
        {
            if ( nobs.vec[t] != NROW(trt[[t]]) )
            {
                stop(paste("Length of element", t, "in trt is different than length of element", t, "in y."))
            }
            trt[[t]] <- drop(trt[[t]])
        }
    }


    trt.vals <- sort(unique( unlist(trt) ))
    if (length(trt.vals) == 1)
    {
        stop("Please provide more than one treatment.")
    } else if (length(trt.vals) > 2)
    {
        stop("trt with more than 2 levels not allowed.")
    }

    if (!all(trt.vals == c(0,1))) stop("trt must take values 0 and 1")

    ## check to make sure weights are valid
    if (!is.null(weights))
    {
        if (!is.list(weights))
        {
            if (!all(nobs.vec == NROW(weights)))
            {
                stop("weights must be same length as each element in y list.")
            }
            weights <- rep(list(weights), nperiods)
        } else
        {
            if (length(weights) != nperiods)
            {
                stop("weights must be a list with same length as y")
            }
            for (t in 1:nperiods)
            {
                if ( nobs.vec[t] != NROW(weights[[t]]) )
                {
                    stop(paste("Length of element", t, "in weights is different than length of element", t, "in y."))
                }
                weights[[t]] <- drop(weights[[t]])
            }
        }
    } else
    {
        weights <- vector(mode = "list", length = nperiods)
        for (t in 1:nperiods) weights[[t]] <- rep(1, nobs.vec[t])
    }


    x.dim.list <- varnames.list <- vector(mode = "list", length = nperiods)
    created.list <- FALSE

    ## check to make sure x is valid
    if (!is.list(x))
    {
        if (is.matrix(x))
        {
            warning("Assuming x is the same over time, as only one x given.")
            x <- rep(list(x), nperiods)
            created.list <- TRUE
        }
    } else
    {
        if (length(x) != nperiods)
        {
            if (length(x) == 1)
            {
                if (is.matrix(x[[1]]))
                {
                    warning("Assuming x is the same over time, as only one x given.")
                    x <- rep(list(x[[1]]), nperiods)
                    created.list <- TRUE
                }
            }
            stop("x must be a list of matrices with length equal to
                 the number of columns in y; one matrix for each time point.")
        }
        for (t in 1:nperiods)
        {
            if (!is.matrix(x[[t]]))
            {
                stop("Each item in the list x should be a matrix")
            }

            x.dim.tmp  <- dim(x[[t]])
            if (is.null(x.dim.tmp))
            {
                stop(paste("element number", t, "in x has null dimension."))
            }
            x.dim.list[[t]] <- x.dim.tmp
            if (x.dim.tmp[1] != nobs.vec[t])
            {
                stop(paste("Number of rows in element", t, "in x is different than length of element", t, "in y."))
            }
            nvars.vec[t] <- x.dim.tmp[2]

            ## code to make sure variable names are consistent over time
            vnames.tmp      <- colnames(x[[t]])
            if (is.null(vnames.tmp))
            {
                vnames.tmp <- paste0("V", 1:x.dim.tmp[2])
                if (!created.list) warning("No variable names given. Assuming variables have same ordering in x
                                           for each time point.")

                if (t > 1)
                {
                    if (x.dim.list[[t - 1]][2] > x.dim.list[[t]][2])
                    {
                        vnames.tmp <- varnames.list[[t - 1]][1:x.dim.list[[t]][2]]
                    } else if (x.dim.list[[t - 1]][2] < x.dim.list[[t]][2])
                    {
                        vnames.tmp <- c(varnames.list[[t - 1]],
                                        vnames.tmp[(x.dim.list[[t - 1]][2] + 1):x.dim.list[[t]][2]])
                    } else
                    {
                        vnames.tmp <- varnames.list[[t - 1]]
                    }
                }
            }
            varnames.list[[t]] <- vnames.tmp
        }
    }


    ## check to make sure lasso.penalize is valid
    if (!is.null(lasso.penalize))
    {
        if (!is.list(lasso.penalize))
        {
            if (!all(nvars.vec == NROW(lasso.penalize)))
            {
                stop("lasso.penalize must be same length as each number of columns in x list.")
            }
            lasso.penalize <- rep(list(lasso.penalize), nperiods)
        } else
        {
            if (length(lasso.penalize) != nperiods)
            {
                stop("lasso.penalize must be a list with same length as y")
            }
            for (t in 1:nperiods)
            {
                if ( nvars.vec[t] != NROW(lasso.penalize[[t]]) )
                {
                    stop(paste("Length of element", t, "in lasso.penalize is
                               different than length of element", t, "in y."))
                }
                lasso.penalize[[t]] <- drop(lasso.penalize[[t]])
            }
        }
    } else
    {
        lasso.penalize <- vector(mode = "list", length = nperiods)
        for (t in 1:nperiods) lasso.penalize[[t]] <- rep(1, nvars.vec[t])
    }


    W.list <- vector(mode = "list", length = nperiods)


    # needed later for getting betas from each time period
    nrow.vec.cumsum     <- c(0, cumsum(nobs.vec))
    ncol.vec.cumsum     <- c(0, cumsum(nvars.vec))
    ncol.vec.trt.cumsum <- c(0, cumsum(nvars.vec + 1))

    cumsum.nobs.vec     <- c(0, cumsum(nobs.vec))
    y.full              <- numeric(total.nobs)
    weight.full         <- numeric(total.nobs)
    lasso.penalize.vec  <- numeric(sum(nvars.vec) + nperiods)

    total.colnames      <- numeric(sum(nvars.vec) + nperiods)
    for (t in 1:nperiods)
    {
        W.list[[t]] <- (2 * trt[[t]] - 1) * cbind(1, x[[t]])
        y.full[((cumsum.nobs.vec[t]) + 1):cumsum.nobs.vec[t+1]]                     <- y[[t]] - mean(y[[t]])
        weight.full[((cumsum.nobs.vec[t]) + 1):cumsum.nobs.vec[t+1]]                <- weights[[t]]
        lasso.penalize.vec[((ncol.vec.trt.cumsum[t]) + 1):ncol.vec.trt.cumsum[t+1]] <- c(0, lasso.penalize[[t]]) ## don't lasso-penalize treatment
        total.colnames[((ncol.vec.trt.cumsum[t]) + 1):ncol.vec.trt.cumsum[t+1]]     <- c("(Treatment)", paste0(varnames.list[[t]], "_period_", t))
    }

    W.full <- as.matrix(bdiag(W.list))
    colnames(W.full) <- total.colnames
    x.full <- bdiag(lapply(x, function(xx) cbind(1, xx)))

    varnames.list.trt <- lapply(varnames.list, function(ll) c("(Treatment)", ll))
    D.mat <- createLongitudinalDVarnames(varnames.list.trt)

    if (is.null(nfolds) || nfolds %in% c(0, 1)) ## don't perform cross validation
    {

    } else
    {
        if(is.null(foldid))
        {
            ## make sure to do stratified k-fold sampling within each time period
            foldid <- unlist(lapply(nobs.vec, function(n) sample(rep(seq(nfolds), length = n))  ))
        } else
        {
            nfolds <- max(foldid)
        }

        cv.model <- cv.fusedlasso(x              = W.full,
                                  y              = y.full,
                                  D              = D.mat,
                                  weights        = weight.full,
                                  lambda         = lambda,
                                  nlambda        = nlambda,
                                  gamma          = gamma,
                                  type.measure   = type.measure,
                                  foldid         = foldid,
                                  lasso.penalize = lasso.penalize.vec,
                                  parallel       = parallel,
                                  abs.tol        = abs.tol,
                                  rel.tol        = rel.tol,
                                  maxit          = maxit,
                                  maxit.cv       = maxit.cv,
                                  rho            = rho,
                                  ...)

        stat.names <- c("Emp Avg All", "Fit Stat: Trt = Rec", "Fit Stat: Trt != Rec",
                        "Fit Stat: Trt = Rec = 1", "Fit Stat: Trt = Rec = 0",
                        "Fit Stat: Trt = 0, Rec = 1", "Fit Stat: Trt = 1, Rec = 0",
                        "Fit Stat: Rec = 1", "Fit Stat: Rec = 0",
                        "Fit Stat: Trt = 1", "Fit Stat: Trt = 0")

        train.res <- numeric(11L)
        names(train.res) <- stat.names
        train.res.list <- rep(list(train.res), nperiods)

        ## compute statistics on original sample

        beta.hat.orig        <- cv.model$best.cv.fit$beta


        nobs.vec.full     <- unlist(lapply(x, NROW))
        cum.nobs.vec.full <- c(0, cumsum(nobs.vec.full))

        beta.mat.all <- matrix(0, nrow = max(nvars.vec) + 1, ncol = nperiods)

        for (t in 1:nperiods)
        {
            idx.cur.row  <- (cum.nobs.vec.full[t] + 1):(cum.nobs.vec.full[t+1])
            idx.cur.col  <- (ncol.vec.trt.cumsum[t] + 1):(ncol.vec.trt.cumsum[t+1])

            beta.hat.t   <- beta.hat.orig[idx.cur.col]

            beta.mat.all[,t] <- beta.hat.t

            ## apply scores on original sample
            xbeta.cur.all               <- drop(cbind(1, x[[t]]) %*% beta.hat.t)

            if (t == 1)
            {
                xbeta.all <- xbeta.cur.all
            } else
            {
                xbeta.all <- xbeta.cur.all + xbeta.all
            }
        }

        xbeta.all <- xbeta.all / nperiods

        D.fused.train                  <- sign(xbeta.all)
        D.fused.train[D.fused.train == -1] <- 0                     # recode as 0/1

        for (t in 1:nperiods)
        {
            idx.cur.row  <- (cum.nobs.vec.full[t] + 1):(cum.nobs.vec.full[t+1])
            idx.cur.col  <- (ncol.vec.trt.cumsum[t] + 1):(ncol.vec.trt.cumsum[t+1])

            beta.hat.t   <- beta.hat.orig[idx.cur.col]

            ## apply scores on original sample
            xbeta.cur               <- drop(cbind(1, x[[t]]) %*% beta.hat.t)
            D.cur                   <- D.fused.train
            y.cur                   <- y[[t]]
            trt.cur                 <- trt[[t]]
            weights.cur             <- weights[[t]]

            # Emprical average among all
            train.res.list[[t]][1]      <- sum(y.cur * weights.cur) / sum(weights.cur)

            # Fit statistics among Trt=D
            sub.agree                <- trt.cur == D.cur
            train.res.list[[t]][2]   <- sum(y.cur[sub.agree] * weights.cur[sub.agree]) /
                sum(weights.cur[sub.agree])

            # Fit statistics among Trt!=D
            sub.disagree             <- trt.cur != D.cur
            train.res.list[[t]][3]   <- sum(y.cur[sub.disagree] * weights.cur[sub.disagree]) /
                sum(weights.cur[sub.disagree])

            # Fit statistics among Trt=D=1
            sub.11                   <- (trt.cur == 1) & (D.cur == 1)
            train.res.list[[t]][4]   <- sum(y.cur[sub.11] * weights.cur[sub.11]) /
                sum(weights.cur[sub.11])

            # Fit statistics among Trt=D=0
            sub.00                   <- (trt.cur == 0) & (D.cur == 0)
            train.res.list[[t]][5]   <- sum(y.cur[sub.00] * weights.cur[sub.00]) /
                sum(weights.cur[sub.00])

            # Fit statistics among Trt=0, D=1
            sub.01                   <- (trt.cur == 0) & (D.cur == 1)
            train.res.list[[t]][6]   <- sum(y.cur[sub.01] * weights.cur[sub.01]) /
                sum(weights.cur[sub.01])

            # Fit statistics among Trt=1, D=0
            sub.10                   <- (trt.cur == 1) & (D.cur == 0)
            train.res.list[[t]][7]   <- sum(y.cur[sub.10] * weights.cur[sub.10]) /
                sum(weights.cur[sub.10])

            # Fit statistics among D=1
            sub.x1                   <- (D.cur == 1)
            train.res.list[[t]][8]   <- sum(y.cur[sub.x1] * weights.cur[sub.x1]) /
                sum(weights.cur[sub.x1])

            # Fit statistics among D=0
            sub.x0                   <- (D.cur == 0)
            train.res.list[[t]][9]   <- sum(y.cur[sub.x0] * weights.cur[sub.x0]) /
                sum(weights.cur[sub.x0])

            # Fit statistics among Trt=1
            sub.1x                    <- (trt.cur == 1)
            train.res.list[[t]][10]   <- sum(y.cur[sub.1x] * weights.cur[sub.1x]) /
                sum(weights.cur[sub.1x])

            # Fit statistics among Trt=0
            sub.0x                    <- (trt.cur == 0)
            train.res.list[[t]][11]   <- sum(y.cur[sub.0x] * weights.cur[sub.0x]) /
                sum(weights.cur[sub.0x])



        }


        if (boot)
        {
            boot.res.mat <- array(0, dim = c(B, 11L))
            colnames(boot.res.mat) <- stat.names
            boot.res.list <- boot.res.orig.list <- rep(list(boot.res.mat), nperiods)
            if (boot.type == "mofn")
                m.vec <- sapply(nobs.vec, function(n) round((n ^ m.frac), digits = 0))
            for (b in 1:B)
            {
                if (boot.type == "mofn")
                {
                    samp.all <- sample.int(total.nobs, size = round(total.nobs ^ m.frac, digits = 0), replace = FALSE)
                    #samp.idx.list <- lapply(1:nperiods, function(i) sample.int(nobs.vec[i], size = m.vec[i], replace = FALSE))
                } else
                {
                    samp.all <- sample.int(total.nobs, size = total.nobs, replace = TRUE)
                    #samp.idx.list <- lapply(1:nperiods, function(i) sample.int(nobs.vec[i], size = nobs.vec[i], replace = TRUE))
                }

                if (boot.type == "mofn")
                {
                    samp.all <- sample.int(nrow(x[[1]]), size = round(nrow(x[[1]]) ^ m.frac, digits = 0), replace = FALSE)
                } else
                {
                    samp.all <- sample.int(nrow(x[[1]]), size = total.nobs, replace = TRUE)
                }
                samp.idx.list <- vector(mode = "list", length = nperiods)
                for (t in 1:nperiods)
                {
                    #samp.idx.list[[t]] <- samp.all[samp.all > nrow.vec.cumsum[t] & samp.all <= nrow.vec.cumsum[t + 1]] - nrow.vec.cumsum[t]
                    samp.idx.list[[t]] <- samp.all
                }

                nobs.samp     <- length(unlist(samp.idx.list))
                W.list.samp   <- trt.list.samp <- y.list.samp <- weight.list.samp <- vector(mode = "list", length = nperiods)
                nobs.vec.samp <- unlist(lapply(samp.idx.list, length))
                cum.nobs.vec  <- c(0, cumsum(nobs.vec.samp))
                y.full.samp   <- weights.full.samp <- trt.full.samp <- numeric(nobs.samp)

                for (t in 1:nperiods)
                {
                    W.list.samp[[t]]      <- W.list[[t]][samp.idx.list[[t]], ]
                    y.tmp                 <- y[[t]] # - mean(y[[t]])
                    y.list.samp[[t]]      <- y.tmp[samp.idx.list[[t]]]# - mean(y.tmp[samp.idx.list[[t]]])
                    trt.list.samp[[t]]    <- trt[[t]][samp.idx.list[[t]]]
                    weight.list.samp[[t]] <- weights[[t]][samp.idx.list[[t]]]

                    trt.full.samp[(cum.nobs.vec[t] + 1):cum.nobs.vec[t+1]]     <- trt.list.samp[[t]]
                    weights.full.samp[(cum.nobs.vec[t] + 1):cum.nobs.vec[t+1]] <- weight.list.samp[[t]]
                    y.full.samp[(cum.nobs.vec[t] + 1):cum.nobs.vec[t+1]]       <- y.list.samp[[t]]
                }
                W.full.samp <- as.matrix(bdiag(W.list.samp))

                folds.1 <- sample(rep(seq(nfolds), length = nobs.vec.samp[1]))
                ## make sure to do stratified k-fold sampling within each time period
                foldid.samp <- unlist(lapply(nobs.vec.samp, function(n) sample(rep(seq(nfolds), length = n))  ))
                foldid.samp <- unlist(rep(list(folds.1), length(nobs.vec.samp)))

                cv.model.samp <- cv.fusedlasso(x              = W.full.samp,
                                               y              = y.full.samp,
                                               D              = D.mat,
                                               weights        = weights.full.samp,
                                               lambda         = lambda,
                                               nlambda        = nlambda,
                                               gamma          = gamma,
                                               type.measure   = type.measure,
                                               foldid         = foldid.samp,
                                               lasso.penalize = lasso.penalize.vec,
                                               parallel       = parallel,
                                               abs.tol        = abs.tol,
                                               rel.tol        = rel.tol,
                                               maxit          = maxit,
                                               maxit.cv       = maxit.cv,
                                               rho            = rho,
                                               ...)

                beta.hat.s        <- cv.model.samp$best.cv.fit$beta

                for (t in 1:nperiods)
                {
                    idx.cur.row      <- (cum.nobs.vec[t] + 1):(cum.nobs.vec[t+1])
                    idx.cur.row.full <- (cum.nobs.vec.full[t] + 1):(cum.nobs.vec.full[t+1])
                    idx.cur.col      <- (ncol.vec.trt.cumsum[t] + 1):(ncol.vec.trt.cumsum[t+1])

                    beta.hat.s.t     <- beta.hat.s[idx.cur.col]

                    ## apply scores on bootstrap sample
                    xbeta.cur          <- drop(cbind(1, x[[t]][samp.idx.list[[t]], ]) %*% beta.hat.s.t)

                    ## apply scores on original sample
                    xbeta.cur.orig               <- drop(cbind(1, x[[t]]) %*% beta.hat.s.t)

                    if (t == 1)
                    {
                        xbeta.all <- xbeta.cur
                        xbeta.all.orig <- xbeta.cur.orig
                    } else
                    {
                        xbeta.all <- xbeta.all + xbeta.cur
                        xbeta.all.orig <- xbeta.all.orig + xbeta.cur.orig
                    }
                }

                xbeta.all <- xbeta.all / nperiods
                xbeta.all.orig <- xbeta.all.orig / nperiods

                D.fused                <- sign(xbeta.all)
                D.fused[D.fused == -1] <- 0      # recode as 0/1

                D.fused.orig                     <- sign(xbeta.all.orig)
                D.fused.orig[D.fused.orig == -1] <- 0                     # recode as 0/1

                for (t in 1:nperiods)
                {
                    idx.cur.row      <- (cum.nobs.vec[t] + 1):(cum.nobs.vec[t+1])
                    idx.cur.row.full <- (cum.nobs.vec.full[t] + 1):(cum.nobs.vec.full[t+1])
                    idx.cur.col      <- (ncol.vec.trt.cumsum[t] + 1):(ncol.vec.trt.cumsum[t+1])

                    beta.hat.s.t     <- beta.hat.s[idx.cur.col]

                    ## apply scores on bootstrap sample
                    xbeta.cur          <- xbeta.all

                    D.cur <- D.fused

                    y.cur              <- y[[t]][samp.idx.list[[t]]] #y.list.samp[[t]]
                    trt.cur            <- trt.list.samp[[t]]
                    weights.cur        <- weight.list.samp[[t]]


                    D.cur.orig <- D.fused.orig

                    y.cur.orig                   <- y[[t]]
                    trt.cur.orig                 <- trt[[t]]
                    weights.cur.orig             <- weights[[t]]

                    # Empirical average among all
                    boot.res.list[[t]][b, 1]      <- sum(y.cur * weights.cur) / sum(weights.cur)
                    boot.res.orig.list[[t]][b, 1] <- sum(y.cur.orig * weights.cur.orig) / sum(weights.cur.orig)


                    # Fit statistics among Trt=D
                    sub.agree                <- trt.cur == D.cur
                    boot.res.list[[t]][b, 2] <- sum(y.cur[sub.agree] * weights.cur[sub.agree]) /
                        sum(weights.cur[sub.agree])

                    sub.agree.orig                <- trt.cur.orig == D.cur.orig
                    boot.res.orig.list[[t]][b, 2] <- sum(y.cur.orig[sub.agree.orig] * weights.cur.orig[sub.agree.orig]) /
                        sum(weights.cur.orig[sub.agree.orig])

                    # Fit statistics among Trt!=D
                    sub.disagree             <- trt.cur != D.cur
                    boot.res.list[[t]][b, 3] <- sum(y.cur[sub.disagree] * weights.cur[sub.disagree]) /
                        sum(weights.cur[sub.disagree])

                    sub.disagree.orig             <- trt.cur.orig != D.cur.orig
                    boot.res.orig.list[[t]][b, 3] <- sum(y.cur.orig[sub.disagree.orig] * weights.cur.orig[sub.disagree.orig]) /
                        sum(weights.cur.orig[sub.disagree.orig])

                    # Fit statistics among Trt=D=1
                    sub.11                   <- (trt.cur == 1) & (D.cur == 1)
                    boot.res.list[[t]][b, 4] <- sum(y.cur[sub.11] * weights.cur[sub.11]) /
                        sum(weights.cur[sub.11])

                    sub.11.orig                   <- (trt.cur.orig == 1) & (D.cur.orig == 1)
                    boot.res.orig.list[[t]][b, 4] <- sum(y.cur.orig[sub.11.orig] * weights.cur.orig[sub.11.orig]) /
                        sum(weights.cur.orig[sub.11.orig])

                    # Fit statistics among Trt=D=0
                    sub.00                   <- (trt.cur == 0) & (D.cur == 0)
                    boot.res.list[[t]][b, 5] <- sum(y.cur[sub.00] * weights.cur[sub.00]) /
                        sum(weights.cur[sub.00])

                    sub.00.orig                   <- (trt.cur.orig == 0) & (D.cur.orig == 0)
                    boot.res.orig.list[[t]][b, 5] <- sum(y.cur.orig[sub.00.orig] * weights.cur.orig[sub.00.orig]) /
                        sum(weights.cur.orig[sub.00.orig])

                    # Fit statistics among Trt=0, D=1
                    sub.01                   <- (trt.cur == 0) & (D.cur == 1)
                    boot.res.list[[t]][b, 6] <- sum(y.cur[sub.01] * weights.cur[sub.01]) /
                        sum(weights.cur[sub.01])

                    sub.01.orig                   <- (trt.cur.orig == 0) & (D.cur.orig == 1)
                    boot.res.orig.list[[t]][b, 6] <- sum(y.cur.orig[sub.01.orig] * weights.cur.orig[sub.01.orig]) /
                        sum(weights.cur.orig[sub.01.orig])

                    # Fit statistics among Trt=1, D=0
                    sub.10                   <- (trt.cur == 1) & (D.cur == 0)
                    boot.res.list[[t]][b, 7] <- sum(y.cur[sub.10] * weights.cur[sub.10]) /
                        sum(weights.cur[sub.10])

                    sub.10.orig                   <- (trt.cur.orig == 1) & (D.cur.orig == 0)
                    boot.res.orig.list[[t]][b, 7] <- sum(y.cur.orig[sub.10.orig] * weights.cur.orig[sub.10.orig]) /
                        sum(weights.cur.orig[sub.10.orig])


                    # Fit statistics among D=1
                    sub.x1                   <- (D.cur == 1)
                    boot.res.list[[t]][b, 8] <- sum(y.cur[sub.x1] * weights.cur[sub.x1]) /
                        sum(weights.cur[sub.x1])

                    sub.x1.orig                   <- (D.cur.orig == 1)
                    boot.res.orig.list[[t]][b, 8] <- sum(y.cur.orig[sub.x1.orig] * weights.cur.orig[sub.x1.orig]) /
                        sum(weights.cur.orig[sub.x1.orig])


                    # Fit statistics among D=0
                    sub.x0                   <- (D.cur == 0)
                    boot.res.list[[t]][b, 9] <- sum(y.cur[sub.x0] * weights.cur[sub.x0]) /
                        sum(weights.cur[sub.x0])

                    sub.x0.orig                   <- (D.cur.orig == 0)
                    boot.res.orig.list[[t]][b, 9] <- sum(y.cur.orig[sub.x0.orig] * weights.cur.orig[sub.x0.orig]) /
                        sum(weights.cur.orig[sub.x0.orig])


                    # Fit statistics among Trt=1
                    sub.1x                    <- (trt.cur == 1)
                    boot.res.list[[t]][b, 10] <- sum(y.cur[sub.1x] * weights.cur[sub.1x]) /
                        sum(weights.cur[sub.1x])

                    sub.1x.orig                    <- (trt.cur.orig == 1)
                    boot.res.orig.list[[t]][b, 10] <- sum(y.cur.orig[sub.1x.orig] * weights.cur.orig[sub.1x.orig]) /
                        sum(weights.cur.orig[sub.1x.orig])

                    # Fit statistics among Trt=0
                    sub.0x                    <- (trt.cur == 0)
                    boot.res.list[[t]][b, 11] <- sum(y.cur[sub.0x] * weights.cur[sub.0x]) /
                        sum(weights.cur[sub.0x])

                    sub.0x.orig                    <- (trt.cur.orig == 0)
                    boot.res.orig.list[[t]][b, 11] <- sum(y.cur.orig[sub.0x.orig] * weights.cur.orig[sub.0x.orig]) /
                        sum(weights.cur.orig[sub.0x.orig])


                }

            }
        } else
        {
            boot.res.list <- boot.res.orig.list <- NULL
        }


        best.fit <- cv.model$best.cv.fit$genlasso.fit

        ## Model with lambda.min
        gamma.min1       <- cv.model$gamma.min
        lambda.min1      <- cv.model$best.cv.fit$lambda.min
        beta.hat1        <- cv.model$best.cv.fit$beta
        var.names1       <- colnames(W.full)
        beta.hat1        <- as.vector(beta.hat1)
        names(beta.hat1) <- var.names1
        var.slc1         <- names(beta.hat1[-(1:2)])[beta.hat1[-(1:2)]!=0]
        beta.nonzero1    <- beta.hat1[-(1:2)][beta.hat1[-(1:2)]!=0]

        var.slc1.list      <- beta.hat1.list <- vector(mode = "list", length = nperiods)
        beta.nonzero1.list <- fit.list <- beta.hat1.list

        for (t in 1:nperiods)
        {
            idx.cur.row  <- (nrow.vec.cumsum[t] + 1):(nrow.vec.cumsum[t+1])
            idx.cur.col  <- (ncol.vec.cumsum[t] + 1):(ncol.vec.cumsum[t+1])

            beta.hat1.list[[t]]     <- beta.hat1[idx.cur.col]
            var.slc1.list[[t]]      <- names(beta.hat1.list[[t]][-(1:2)])[beta.hat1.list[[t]][-(1:2)] != 0]
            beta.nonzero1.list[[t]] <- beta.hat1.list[[t]][-(1:2)][beta.hat1.list[[t]][-(1:2)] != 0]

            fit.list[[t]] <- list(lambda    = lambda.min1,
                                  gamma     = gamma.min1,
                                  selected  = var.slc1.list[[t]] ,
                                  beta.hat  = beta.hat1.list[[t]],
                                  nonzero   = beta.nonzero1.list[[t]])
        }


    }

    fit <- list(lambda    = lambda.min1,
                selected  = var.slc1,
                beta.hat  = beta.hat1,
                nonzero   = beta.nonzero1,
                beta.list = beta.hat1.list,
                gamma     = gamma.min1)

    # Benefit scores and recommended treatment for every episode test set
    benefit.all.full <- as.vector(x.full %*% (as.vector(beta.hat1)))


    benefit.score.list <- trt.assignment.list <- vector(mode = "list", length = nperiods)
    trt.assignments <- numeric(total.nobs)
    names(benefit.score.list) <- names(trt.assignment.list) <- paste0("Time", 1:nperiods)
    names(beta.hat1.list) <- names(benefit.score.list)
    for (t in 1:nperiods)
    {
        idx.cur.row  <- (nrow.vec.cumsum[t] + 1):(nrow.vec.cumsum[t + 1])
        idx.cur.col  <- (ncol.vec.cumsum[t] + 1):(ncol.vec.cumsum[t + 1])

        benefit.all <- benefit.all.full[idx.cur.row]

        benefit.score.list[[t]] <- benefit.all

        Trt <- trt[[t]]

        D.all <- sign(benefit.all)
        D.all[D.all == -1] <- 0
        trt.assignments[idx.cur.row] <- D.all
        trt.assignment.list[[t]]     <- D.all
    }


    if (boot)
    {
        train.bias.corrected.res.list <- bias.res.list <- train.res.list

        ## calculate bootstrap bias and adjust for it
        for (t in 1:nperiods)
        {
            bias.res.list[[t]] <- colMeans(boot.res.list[[t]] - boot.res.orig.list[[t]])
            train.bias.corrected.res.list[[t]] <- train.res.list[[t]] - bias.res.list[[t]]
        }
    } else
    {
        train.bias.corrected.res.list <- bias.res.list <- NULL
    }

    vnames <- colnames(x[[1]])

    if (is.null(vnames))
    {
        vnames <- c("Trt", paste0("V", 1:ncol(x[[1]])))
    } else
    {
        vnames <- c("Trt", colnames(x[[1]]) )
    }


    colnames(beta.mat.all) <- paste0(t, 1:nperiods)
    rownames(beta.mat.all) <- vnames

    ret <- list(beta.mat            = beta.mat.all,
                benefit.scores      = benefit.all.full,
                trt.assignments     = trt.assignments,
                benefit.score.list  = benefit.score.list,
                trt.assignment.list = trt.assignment.list,
                coefficients        = beta.hat1.list,
                cv.model            = cv.model,
                best.fit            = best.fit,
                fit.stats           = train.res.list,
                fit.stats.bias.adj  = train.bias.corrected.res.list,
                boot.res            = boot.res.list,
                boot.res.orig       = boot.res.orig.list,
                family              = family,
                lambda              = lambda,
                gamma               = gamma,
                lambda.min          = cv.model$best.cv.fit$lambda.min,
                gamma.min           = cv.model$gamma.min,
                which.lambda.min    = which.min(cv.model$best.cv.fit$cvm),
                which.gamma.min     = which.min(cv.model$cvm),
                x.dims              = x.dim.list,
                varnames            = varnames.list)
    class(ret) <- "subgroupLong"
    ret
}


#' Plot method for subgroupLong fitted objects
#'
#' @param x fitted "subgroupLong" model object
#' @param legend a keyword specifying the legend location. See details for legend. If \code{legend = 'none'},
#' no legend will be made
#' @param ... other graphical parameters for the plot
#' @rdname plot
#' @importFrom graphics legend lines par plot
#' @importFrom stats quantile
#' @export
plot.subgroupLong <- function(x, legend = NULL, ...)
{
    if (!is.null(x$boot.res))
    {

        means <- lapply(x$boot.res, function(x) apply(x, 2, function(xx)
            mean(xx, na.rm = TRUE)) )
        CIs <- lapply(x$boot.res, function(x) apply(x, 2, function(xx)
            quantile(xx, probs = c(0.025, 0.975), na.rm = TRUE)) )

        periods <- length(CIs)

        y.rng <- range(unlist(lapply(CIs, function(xx) range(xx[,4:7]))))

        CI.array <- array(NA, dim = c(dim(CIs[[1]]), periods))
        mean.mat <- array(NA, dim = c(length(means[[1]]), periods))

        rownames(mean.mat) <- names(means[[1]])
        colnames(mean.mat) <- paste0("Time", 1:periods)
        dimnames(CI.array) <- c(dimnames(CIs[[1]]), list(paste0("Time", 1:periods)))


        for (t in 1:periods)
        {
            CI.array[,,t] <- CIs[[t]]
            mean.mat[,t]  <- means[[t]]
        }

        par(mfrow = c(1, 2))
        plot(0, xlim = c(1, periods),
             ylim = y.rng, cex = 0,
             col = "white",
             main = "Recommended Trt",
             xlab = "Time", ...)
        lines(x = 1:periods, y = mean.mat[4,], lwd = 2, col = "#0000FFD9")
        lines(x = 1:periods, y = mean.mat[6,], lwd = 2, col = "#FF0000D9")

        # add vertical lines for TRT = 1
        segments(x0 = 1:periods, x1 = 1:periods,
                 y0 = CI.array[1,4,], y1 = CI.array[2,4,], col = "#0000FFD9")

        # add horizontal bars for TRT = 1
        bar.width <- 0.15
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[1,4,], y1 = CI.array[1,4,], col = "#0000FFD9")
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[2,4,], y1 = CI.array[2,4,], col = "#0000FFD9")

        # add vertical lines for TRT = 0
        segments(x0 = 1:periods, x1 = 1:periods,
                 y0 = CI.array[1,6,], y1 = CI.array[2,6,], col = "#FF0000D9")

        # add horizontal bars for TRT = 0
        bar.width <- 0.15
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[1,6,], y1 = CI.array[1,6,], col = "#FF0000D9")
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[2,6,], y1 = CI.array[2,6,], col = "#FF0000D9")

        plot(0, xlim = c(1, periods),
             ylim = y.rng, cex = 0,
             col = "white",
             main = "Recommended Ctrl",
             xlab = "Time", ...)
        lines(x = 1:periods, y = mean.mat[5,], lwd = 2, col = "#FF0000D9")
        lines(x = 1:periods, y = mean.mat[7,], lwd = 2, col = "#0000FFD9")

        if (!is.null(legend))
        {
            if (legend != "none")
            {
                legend(legend, legend = c("Received Trt", "Received Ctrl"),
                       col = c("#0000FFD9", "#FF0000D9"), lty = 1, lwd = 2)
            }
        } else
        {
            legend("topright", legend = c("Received Trt", "Received Ctrl"),
                   col = c("#0000FFD9", "#FF0000D9"), lty = 1, lwd = 2)
        }




        # add vertical lines for TRT = 1
        segments(x0 = 1:periods, x1 = 1:periods,
                 y0 = CI.array[1,7,], y1 = CI.array[2,7,], col = "#0000FFD9")

        # add horizontal bars for TRT = 1
        bar.width <- 0.15
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[1,7,], y1 = CI.array[1,7,], col = "#0000FFD9")
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[2,7,], y1 = CI.array[2,7,], col = "#0000FFD9")

        # add vertical lines for TRT = 0
        segments(x0 = 1:periods, x1 = 1:periods,
                 y0 = CI.array[1,5,], y1 = CI.array[2,5,], col = "#FF0000D9")

        # add horizontal bars for TRT = 0
        bar.width <- 0.15
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[1,5,], y1 = CI.array[1,5,], col = "#FF0000D9")
        segments(x0 = 1:periods - bar.width, x1 = 1:periods + bar.width,
                 y0 = CI.array[2,5,], y1 = CI.array[2,5,], col = "#FF0000D9")


    }
}

# #' cross validation for interaction detection for longitudinal outcomes using fused lasso
# #'
# #' @param x input matrix.
# #' Each row is an observation, each column corresponds to a covariate
# #' @param y numeric response vector of length nobs.
# #' @param family "gaussian" for least squares problems, "binomial" for binary response.
# #' "coxph" for time-to-event outcomes
# #' @param lambda tuning parameter values for lasso penalty
# #' @param gamma ratio of fused lasso to lasso tuning parameter
# #' @return An object with S3 class "cv.subgroupLong"
# #' @param nlambda number of tuning parameter values - default is 100.
# #' @param ... other parameters to be passed to subgroupLong
# #' @import Rcpp
# #' @import Matrix
# #' @import foreach
# #' @export
# #' @examples
# #' set.seed(123)
# #' n.obs <- 100
# #' n.vars <- 10
# #'
# #' true.beta <- c(runif(3, -0.25, 0.25), rep(0, n.vars - 3))
# #' true.beta.int <- c(rep(0, n.vars - 3), runif(3, -0.5, 0.5))
# #'
# #' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
# #' trt <- rbinom(n.obs, 1, 0.5)
# #' y <- rnorm(n.obs, sd = 3) + x %*% true.beta + trt * (x %*% true.beta.int)
# #'
# #' fit <- cv.subgroupLong(x = x, y = y)
# #'
# cv.subgroupLong <- function(x,
#                                  y,
#                                  family   = c("gaussian", "binomial", "coxph"),
#                                  lambda   = numeric(0),
#                                  nlambda  = 100L,
#                                  nfolds   = 10L,
#                                  foldid   = NULL,
#                                  ...)
# {
#     ## subgroupLong.fit should actually
#     ## be a fitted subgroupLong object
#     fitobj <- "test.object"
#
#     family <- match.arg(family)
#     ret <- list(subgroupLong.fit = fitobj, family = family, lambda = lambda)
#     class(ret) <- "cv.subgroupLong"
#     ret
# }


cv.fusedlasso <- function(x,
                          y,
                          D,
                          gamma,
                          weights,
                          lambda         = numeric(0),
                          nlambda        = 100L,
                          type.measure   = c("mse","deviance","class","auc","mae"),
                          lasso.penalize = NULL,
                          nfolds         = 10,
                          foldid         = NULL,
                          grouped        = TRUE,
                          keep           = FALSE,
                          parallel       = FALSE,
                          abs.tol        = 1e-5,
                          rel.tol        = 1e-5,
                          maxit          = 250,
                          maxit.cv       = 250,
                          rho            = NULL,
                          ...) {
    N = nrow(x)
    P = ncol(x)
    if (missing(weights))
    {
        weights <- rep(1.0, N)
    }
    if (is.null(lasso.penalize))
    {
        lasso.penalize <- rep(1, P)
    } else if (length(lasso.penalize) == 1)
    {
        if (lasso.penalize)
        {
            lasso.penalize <- rep(1, P)
        } else
        {
            lasso.penalize <- NULL
        }
    }
    if (missing(gamma)) {
        D.list <- list(D)
        return(cv.genlasso(x            = x,
                           y            = y,
                           D            = D,
                           weights      = weights,
                           lambda       = lambda,
                           nlambda      = nlambda,
                           type.measure = type.measure,
                           nfolds       = nfolds,
                           foldid       = foldid,
                           grouped      = grouped,
                           keep         = keep,
                           parallel     = parallel,
                           abs.tol      = abs.tol,
                           rel.tol      = rel.tol,
                           maxit        = maxit,
                           maxit.cv     = maxit.cv,
                           rho          = rho,
                           ...))
    } else {
        if (any( apply(x, 2, sd) == 0 ))
        {
            intercept <- TRUE
        } else
        {
            intercept <- FALSE
        }
        ngamma <- length(gamma)
        model.list <- D.list <- vector(mode = "list", length = ngamma)
        if (!parallel) {
            for (g in 1:ngamma) {
                gamma.cur <- gamma[g]

                if (!is.null(lasso.penalize))
                {
                    D.lasso  <- gamma.cur * diag(ncol(D))
                    keep.idx <- which((1 * lasso.penalize) != 0)

                    if (length(keep.idx) > 0)
                    {
                        D.lasso <- D.lasso[keep.idx, ]
                        D.cur   <- rbind(D, D.lasso)
                    } else
                    {
                        D.cur <- D
                    }
                } else
                {
                    D.cur <- D
                }

                D.list[[g]] <- D.cur
                model.list[[g]] <- cv.genlasso(x            = x,
                                               y            = y,
                                               D            = D.cur,
                                               weights      = weights,
                                               lambda       = lambda,
                                               nlambda      = nlambda,
                                               type.measure = type.measure,
                                               nfolds       = nfolds,
                                               foldid       = foldid,
                                               grouped      = grouped,
                                               keep         = keep,
                                               parallel     = parallel,
                                               abs.tol      = abs.tol,
                                               rel.tol      = rel.tol,
                                               maxit        = maxit,
                                               maxit.cv     = maxit.cv,
                                               rho          = rho,
                                               ...)
            }
        } else {
            outlist = foreach (g=seq(ngamma), .packages = c("personalizedLong") ) %dopar% {
                                               gamma.cur <- gamma[g]

                                               if (!is.null(lasso.penalize)) {
                                                   D.lasso  <- gamma.cur * diag(ncol(D))
                                                   keep.idx <- which((1 * lasso.penalize) != 0)

                                                   if (length(keep.idx) > 0) {
                                                       D.lasso <- D.lasso[keep.idx, ]
                                                       D.cur   <- rbind(D, D.lasso)
                                                   } else {
                                                       D.cur <- D
                                                   }
                                               } else {
                                                   D.cur <- D
                                               }

                                               D.tmp <- D.cur
                                               model.tmp <- cv.genlasso(x            = x,
                                                                        y            = y,
                                                                        D            = D.cur,
                                                                        weights      = weights,
                                                                        lambda       = lambda,
                                                                        nlambda      = nlambda,
                                                                        type.measure = type.measure,
                                                                        nfolds       = nfolds,
                                                                        foldid       = foldid,
                                                                        grouped      = grouped,
                                                                        keep         = keep,
                                                                        parallel     = FALSE,
                                                                        abs.tol      = abs.tol,
                                                                        rel.tol      = rel.tol,
                                                                        maxit        = maxit,
                                                                        maxit.cv     = maxit.cv,
                                                                        rho          = rho,
                                                                        ...)
                                               list(D.tmp, model.tmp)
                                           }

            D.list     <- lapply(outlist, function(x) x[[1]])
            model.list <- lapply(outlist, function(x) x[[2]])
        }

        best.idx <- sapply(model.list, function(x) which.min(x$cvm))

        cvm <- cvsd <- cvup <- cvlo <- numeric(ngamma)
        for (g in 1:ngamma) {
            mod.tmp <- model.list[[g]]
            cvm[g]  <- mod.tmp$cvm[best.idx[g]]
            cvsd[g] <- mod.tmp$cvsd[best.idx[g]]
            cvup[g] <- mod.tmp$cvup[best.idx[g]]
            cvlo[g] <- mod.tmp$cvlo[best.idx[g]]
        }
        best.gamma.idx <- which.min(cvm)
        gamma.min      <- gamma[best.gamma.idx]
        best.fit       <- model.list[[best.gamma.idx]]

        retlist <- list(gamma       = gamma,
                        cvm         = cvm,
                        cvsd        = cvsd,
                        cvup        = cvup,
                        cvlo        = cvlo,
                        best.cv.fit = best.fit,
                        gamma.min   = gamma.min,
                        D.list      = D.list)
        class(retlist) <- "cv.gamma.genlasso"
        return(retlist)
    }
}

cv.genlasso <- function(x,
                        y,
                        D,
                        weights,
                        type.measure = c("mse","deviance","class","auc","mae"),
                        lambda       = numeric(0),
                        nlambda      = 100,
                        nfolds       = 10,
                        foldid       = NULL,
                        grouped      = TRUE,
                        keep         = FALSE,
                        parallel     = FALSE,
                        abs.tol      = 1e-5,
                        rel.tol      = 1e-5,
                        maxit        = 250,
                        maxit.cv     = 250,
                        rho          = NULL,
                        ...)
{
    if (missing(type.measure))
    {
        type.measure <- "default"
    } else
    {
        type.measure <- match.arg(type.measure)
    }
    #if(!is.null(lambda)&&length(lambda)<2)stop("Need more than one value of lambda for cv.glmnet")
    N <- nrow(x)
    if(missing(weights))weights=rep(1.0,N)else weights=as.double(weights)
    ###Fit the model once to get dimensions etc of output
    y=drop(y) # we dont like matrix responses unless we need them
    ###Next we construct a call, that could recreate a genlasso object - tricky
    ### This if for predict, exact=TRUE
    genlasso.call=match.call(expand.dots=TRUE)
    which = match(c("type.measure","nfolds","foldid","grouped","keep"),names(genlasso.call),FALSE)
    if(any(which))genlasso.call=genlasso.call[-which]
    genlasso.call[[1]]=as.name("genlasso")


    if (is.null(rho))
    {
        eigs <- eigen(crossprod(x * sqrt(weights)))$values

        mineig <- eigs[eigs > 1e-6][sum(eigs > 1e-6)]

        if (length(mineig) == 0)
        {
            mineig <- 1e-6
        }

        rho <- sqrt(eigs[1] * (mineig + 1e-4) )
    }

    weights[weights <= 0] <- 1e-5

    genlasso.object <- admm.genlasso(x       = x * sqrt(weights),
                                     y       = y * sqrt(weights),
                                     D       = D,
                                     lambda  = lambda,
                                     nlambda = nlambda,
                                     abs.tol = abs.tol,
                                     rel.tol = rel.tol,
                                     maxit   = maxit,
                                     rho     = rho)

    # this is a list of indices for each tuning parameter value
    # of all the coefficients which are zero
    # make sure to take out intercept of beta.aug, which is the first row
    genlasso.object$beta   <- genlasso.object$beta[-1,] # remove the intercept (we aren't using it)
    augmented.var.zero.idx <- apply(as.matrix(genlasso.object$beta.aug[-1,]), 2, function(xx) which(xx == 0))
    lasso.idx              <- which(rowSums(D != 0) == 1)

    if (is.null(dim(augmented.var.zero.idx)))
    {
        len_ii <- length(augmented.var.zero.idx)
    } else
    {
        len_ii <- ncol(augmented.var.zero.idx)
    }

    for (ii in 1:len_ii)
    {
        if (is.list(augmented.var.zero.idx))
        {
            zero.idx <- intersect(augmented.var.zero.idx[[ii]], lasso.idx)
        } else
        {
            zero.idx <- intersect(augmented.var.zero.idx[,ii], lasso.idx)
        }
        # get beta index of all coefficients which are zero
        if (length(zero.idx))
        {
            beta.zero.idx <- apply(D[zero.idx,,drop = FALSE], 1, function(xx) which(xx != 0))
            if (length(beta.zero.idx))
            {
                genlasso.object$beta[beta.zero.idx, ii] <- 0
            }
        }
    }

    genlasso.object$call <- genlasso.call
    lambda               <- genlasso.object$lambda
    nlams                <- length(lambda)
    if (nlams > 2)
    {
        lambda <- lambda[1:(floor(0.975 * nlams))]
    }
    min.lam <- min(lambda)

    nz <- colSums(genlasso.object$beta != 0)
    if(is.null(foldid)) foldid <- sample(rep(seq(nfolds),length=N)) else nfolds = max(foldid)
    if(nfolds < 3) stop("nfolds must be bigger than 3; nfolds=10 recommended")
    outlist <- as.list(seq(nfolds))
    ###Now fit the nfold models and store them
    ###First try and do it using foreach if parallel is TRUE
    if (parallel)
    {
        outlist = foreach (i=seq(nfolds), .packages=c("personalizedLong")) %dopar% {
            which = foldid==i
            if(is.matrix(y))y_sub=y[!which,]else y_sub=y[!which]

            admm.genlasso(x       = x[!which,,drop=FALSE] * sqrt(weights[!which]),
                          y       = y_sub * sqrt(weights[!which]),
                          D       = D,
                          lambda  = lambda,
                          maxit   = maxit.cv,
                          abs.tol = abs.tol,
                          rel.tol = rel.tol)

        }
    } else
    {
        for(i in seq(nfolds))
        {
            which = foldid==i
            if(is.matrix(y))y_sub=y[!which,]else y_sub=y[!which]

            outlist[[i]] = admm.genlasso(x       = x[!which,,drop=FALSE] * sqrt(weights[!which]),
                                         y       = y_sub * sqrt(weights[!which]),
                                         D       = D,
                                         lambda  = lambda,
                                         maxit   = maxit.cv,
                                         abs.tol = abs.tol,
                                         rel.tol = rel.tol)
            lambda.tmp   = outlist[[i]]$lambda
        }
    }

    minlams     <- c(min.lam, sapply(outlist, function(x) min(x$lambda)))
    max.min.lam <- max(minlams)
    lambda      <- lambda[which(lambda >= max.min.lam)]
    genlasso.object$lambda <- lambda
    ###What to do depends on the type.measure and the model fit
    #fun=paste("cv",class(genlasso.object)[[1]],sep="_") #cv_genlasso
    fun     = "cv_genlasso"
    cvstuff = do.call(fun,list(outlist,lambda,x,y,weights,foldid,type.measure,grouped,keep))
    cvm     = cvstuff$cvm
    cvsd    = cvstuff$cvsd
    cvname  = cvstuff$name
    best.beta <- genlasso.object$beta[,which.min(cvm)]

    out = list(beta         = best.beta,
               lambda       = lambda,
               cvm          = cvm,
               cvsd         = cvsd,
               cvup         = cvm + cvsd,
               cvlo         = cvm - cvsd,
               nzero        = nz,
               name         = cvname,
               genlasso.fit = genlasso.object)

    if(keep)out <- c(out, list(fit.preval = cvstuff$fit.preval, foldid = foldid))
    lamin = if(type.measure=="auc") getmin(lambda,-cvm, cvsd)
    else getmin(lambda, cvm, cvsd)
    obj=c(out,as.list(lamin))
    class(obj)="cv.genlasso"
    obj
}



#' Prediction method for subgroupLong fitted objects
#'
#' @param object fitted "subgroupLong" model object
#' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix; can be sparse as in Matrix package.
#' This argument is not used for type=c("coefficients","nonzero")
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create
#' the model.
#' @param type Type of prediction required. Type == "link" gives the linear predictors for the "binomial" model; for "gaussian" models it gives the fitted values.
#' Type == "response" gives the fitted probabilities for "binomial". Type "coefficients" computes the coefficients at the requested values for s.
#' Type "class" applies only to "binomial" and produces the class label corresponding to the maximum probability.
#' @param ... not used
#' @return An object depending on the type argument
#' @export
#' @examples
#' set.seed(123)
#' nobs       <- 100
#' nobs.test  <- 1e5
#' nvars      <- 20
#' periods    <- 6
#' sd         <- 2
#'
#' beta.nz <- rbind(c( 1,    1,    1,    1.5,  1.5,  1.5),
#'                  c(-1,   -1,   -1,   -0.5, -0.5, -0.5),
#'                  c( 1,    1,    1,   -1,   -1,   -1),
#'                  c( 1,    1,    1,    1,    1,    1),
#'                  c(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5))
#'
#' beta <- data.matrix(rbind(beta.nz, matrix(0, nvars - 5, periods)))
#'
#' trt <- rbinom(nobs, 1, 0.5)
#' x   <- matrix(rnorm(nobs * nvars), ncol = nvars); colnames(x) <- paste0("V", 1:ncol(x))
#' y   <- x %*% (beta * 0.5) + (2 * trt - 1) * (x %*% beta) +
#'     matrix(rnorm(nobs * periods, sd = sd), ncol = periods)
#' y   <- apply(y, 2, function(yy) yy - (2 * trt - 1))
#'
#' plot(x = NULL, xlim = c(1,6), ylim = range(y))
#' for (i in 1:nobs)
#' {
#'     lines(x = 1:6, y = y[i,], col = colors()[i+1])
#' }
#'
#' x.list <- rep(list(x), periods)
#' y.list <- lapply(apply(y, 2, function(x) list(x) ), function(x) x[[1]])
#'
#' fit <- subgroupLong(x = x.list, y = y.list, trt = trt, gamma = c(0.05, 0.1, 1, 2, 5, 10))
#'
#' preds <- predict(fit, newx = x)
#'
#'
predict.subgroupLong <- function(object, newx, s = NULL,
                                      type = c("link",
                                               "response",
                                               "coefficients",
                                               "nonzero",
                                               "class"), ...)
{

}


# #' Prediction method for cv.subgroupLong fitted objects
# #'
# #' @param object fitted "cv.subgroupLong" model object
# #' @param newx Matrix of new values for x at which predictions are to be made. Must be a matrix; can be sparse as in Matrix package.
# #' This argument is not used for type=c("coefficients","nonzero")
# #' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create
# #' the model.
# #' @param type Type of prediction required. Type == "link" gives the linear predictors for the "binomial" model; for "gaussian" models it gives the fitted values.
# #' Type == "response" gives the fitted probabilities for "binomial". Type "coefficients" computes the coefficients at the requested values for s.
# #' Type "class" applies only to "binomial" and produces the class label corresponding to the maximum probability.
# #' @param ... not used
# #' @return An object depending on the type argument
# #' @method predict cv.subgroupLong
# #' @export
# #' @examples
# #' set.seed(123)
# #' n.obs <- 100
# #' n.vars <- 10
# #'
# #' true.beta <- c(runif(3, -0.25, 0.25), rep(0, n.vars - 3))
# #' true.beta.int <- c(rep(0, n.vars - 3), runif(3, -0.5, 0.5))
# #'
# #' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
# #' trt <- rbinom(n.obs, 1, 0.5)
# #' y <- rnorm(n.obs, sd = 3) + x %*% true.beta + trt * (x %*% true.beta.int)
# #'
# #' fit <- cv.subgroupLong(x = x, y = y)
# #'
# #' preds <- predict(fit, newx = x)
# #'
# #'
# predict.cv.subgroupLong <- function(object, newx,
#                                          s=c("lambda.1se","lambda.min"), ...)
# {
#     if(is.numeric(s))lambda=s
#     else
#         if(is.character(s)){
#             s=match.arg(s)
#             lambda=object[[s]]
#         }
#
#     else stop("Invalid form for s")
#     predict(object$subgroupLong.fit, newx, s=lambda, ...)
# }



#' Summary method for subgroupLong fitted objects
#'
#' @param object fitted "subgroupLong" model object
#' @param ... not used
#' @return An object depending on the type argument
#' @rdname summary
#' @method summary subgroupLong
#' @export
#' @examples
#' set.seed(123)
#' nobs       <- 100
#' nobs.test  <- 1e5
#' nvars      <- 20
#' periods    <- 6
#' sd         <- 2
#'
#' beta.nz <- rbind(c( 1,    1,    1,    1.5,  1.5,  1.5),
#'                  c(-1,   -1,   -1,   -0.5, -0.5, -0.5),
#'                  c( 1,    1,    1,   -1,   -1,   -1),
#'                  c( 1,    1,    1,    1,    1,    1),
#'                  c(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5))
#'
#' beta <- data.matrix(rbind(beta.nz, matrix(0, nvars - 5, periods)))
#'
#' trt <- rbinom(nobs, 1, 0.5)
#' x   <- matrix(rnorm(nobs * nvars), ncol = nvars); colnames(x) <- paste0("V", 1:ncol(x))
#' y   <- x %*% (beta * 0.5) + (2 * trt - 1) * (x %*% beta) +
#'     matrix(rnorm(nobs * periods, sd = sd), ncol = periods)
#' y   <- apply(y, 2, function(yy) yy - (2 * trt - 1))
#'
#' plot(x = NULL, xlim = c(1,6), ylim = range(y))
#' for (i in 1:nobs)
#' {
#'     lines(x = 1:6, y = y[i,], col = colors()[i+1])
#' }
#'
#' x.list <- rep(list(x), periods)
#' y.list <- lapply(apply(y, 2, function(x) list(x) ), function(x) x[[1]])
#'
#' fit <- subgroupLong(x = x.list, y = y.list, trt = trt, gamma = c(0.05, 0.1, 1, 2, 5, 10))
#'
#' summary(fit)
#'
#'
summary.subgroupLong <- function(object, ...)
{

}



