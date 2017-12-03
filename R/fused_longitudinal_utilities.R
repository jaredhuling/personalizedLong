## files to make code for fusedLongitudinal cleaner and re-usable


betaLong2Mat <- function(beta, n.time) {
    n.vars <- as.integer(length(beta) / n.time)
    beta.mat <- array(NA, dim = c(n.vars, n.time))
    for (i in 1:n.time) {
        beta.mat[,i] <- beta[((i-1) * n.vars + 1):(i * n.vars)]
    }
    beta.mat
}

createLongitudinalD <- function(n.time, n.vars, one.intercept = FALSE) {
    D1 <- array(0, dim = c( (n.time - 1) * n.vars, n.vars * n.time ))
    k <- 0
    if (one.intercept) {
        j.idx <- 1:n.vars
    } else {
        j.idx <- 2:n.vars
    }
    for (j in j.idx)
    {
        for (i in 1:( n.time - 1))
        {
            k <- k + 1
            D1[k, c( ((i-1) * (n.vars) + j ),
                     ((i) * (n.vars) + j) )] <- c(1, -1)
        }
    }
    all.zeros <- which(rowSums(abs(D1)) == 0)
    if (length(all.zeros) > 0) {
        D1 <- D1[-all.zeros,]
    }
    D1
}

createLongitudinalDVarnames <- function(varname.list)
{
    n.time     <- length(varname.list)
    nvars.vec  <- unlist(lapply(varname.list, length))
    cum.nvars  <- c(0, cumsum(nvars.vec))
    total.vars <- sum(nvars.vec)
    D1 <- array(0, dim = c( (n.time - 1) * max(nvars.vec), total.vars ))
    k  <- 0

    for (t in 1:(n.time - 1))
    {
        idx.cur.col  <- (cum.nvars[t] + 1):(cum.nvars[t + 1])
        idx.next.col <- (cum.nvars[t + 1] + 1):(cum.nvars[t + 2])
        for (v in 1:length(varname.list[[t]]))
        {
            vname.cur <- varname.list[[t]][v]
            idx.next.period <- match(vname.cur, varname.list[[t + 1]])
            if (!is.na(idx.next.period))
            {
                k <- k + 1
                D1[k, c(idx.cur.col[v], idx.next.col[idx.next.period])] <- c(1, -1)
            }
        }
    }
    all.zeros <- which(rowSums(abs(D1)) == 0)
    if (length(all.zeros) > 0) {
        D1 <- D1[-all.zeros,]
    }
    D1
}



#' Fitting A Generalized Lasso Model Using ADMM Algorithm
#'
#' @description Estimation of a linear model with the lasso penalty. The function
#' \eqn{\beta} minimizes
#' \deqn{\frac{1}{2n}\Vert y-X\beta\Vert_2^2+\lambda\Vert\beta\Vert_1}{
#' 1/(2n) * ||y - X * \beta||_2^2 + \lambda * ||D\beta||_1}
#'
#' where \eqn{n} is the sample size and \eqn{\lambda} is a tuning
#' parameter that controls the sparseness of \eqn{\beta}.
#'
#' @param x The design matrix
#' @param y The response vector
#' @param D The specified penalty matrix
#' @param intercept Whether to fit an intercept in the model. Default is \code{FALSE}.
#' @param standardize Whether to standardize the design matrix before
#'                    fitting the model. Default is \code{FALSE}. Fitted coefficients
#'                    are always returned on the original scale.
#' @param lambda A user provided sequence of \eqn{\lambda}. If set to
#'                      \code{NULL}, the program will calculate its own sequence
#'                      according to \code{nlambda} and \code{lambda_min_ratio},
#'                      which starts from \eqn{\lambda_0} (with this
#'                      \eqn{\lambda} all coefficients will be zero) and ends at
#'                      \code{lambda0 * lambda_min_ratio}, containing
#'                      \code{nlambda} values equally spaced in the log scale.
#'                      It is recommended to set this parameter to be \code{NULL}
#'                      (the default).
#' @param nlambda Number of values in the \eqn{\lambda} sequence. Only used
#'                       when the program calculates its own \eqn{\lambda}
#'                       (by setting \code{lambda = NULL}).
#' @param lambda.min.ratio Smallest value in the \eqn{\lambda} sequence
#'                                as a fraction of \eqn{\lambda_0}. See
#'                                the explanation of the \code{lambda}
#'                                argument. This parameter is only used when
#'                                the program calculates its own \eqn{\lambda}
#'                                (by setting \code{lambda = NULL}). The default
#'                                value is the same as \pkg{glmnet}: 0.0001 if
#'                                \code{nrow(x) >= ncol(x)} and 0.01 otherwise.
#' @param maxit Maximum number of admm iterations.
#' @param abs.tol Absolute tolerance parameter.
#' @param rel.tol Relative tolerance parameter.
#' @param rho ADMM step size parameter. If set to \code{NULL}, the program
#'                   will compute a default one which has good convergence properties.
#' @references
#' \url{https://projecteuclid.org/euclid.aos/1304514656}
#'
#' \url{http://stanford.edu/~boyd/admm.html}
#' @examples set.seed(123)
#' n = 1000
#' p = 50
#' b = c(runif(10), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#'
#' D <- c(1, -1, rep(0, p - 2))
#' for (i in 1:20) {D <- rbind(D, c(rep(0, 2 * i), 1, -1, rep(0, p - 2 - 2 * i)))}
#' D <- rbind(D, diag(p))
#'
#' ## fit lasso model with 100 tuning parameter values
#' res <- admm.genlasso(x, y, D = D)
#'
#'
#' @export
#' @importFrom graphics abline axis points segments
#' @importFrom methods as
#' @importFrom stats predict sd weighted.mean
admm.genlasso <- function(x,
                          y,
                          D                = NULL,
                          lambda           = numeric(0),
                          nlambda          = 100L,
                          lambda.min.ratio = NULL,
                          intercept        = FALSE,
                          standardize      = FALSE,
                          maxit            = 5000L,
                          abs.tol          = 1e-7,
                          rel.tol          = 1e-7,
                          rho              = NULL
)
{
    n <- nrow(x)
    p <- ncol(x)

    if (is.null(lambda.min.ratio))
    {
        ifelse(n < p, 0.01, 0.0001)
    }

    if (is.null(D)) {
        warning("D is missing, defaulting to regular lasso")
        D <- as(diag(p), "sparseMatrix")
    } else {
        D <- as(D, "sparseMatrix")
    }

    x = as.matrix(x)
    y = as.numeric(y)
    intercept = as.logical(intercept)
    standardize = as.logical(standardize)

    if (n != length(y)) {
        stop("number of rows in x not equal to length of y")
    }

    lambda_val = sort(as.numeric(lambda), decreasing = TRUE)

    if(any(lambda_val <= 0))
    {
        stop("lambda must be positive")
    }

    if(nlambda[1] <= 0)
    {
        stop("nlambda must be a positive integer")
    }

    if(is.null(lambda.min.ratio))
    {
        lmr_val <- ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
    } else
    {
        lmr_val <- as.numeric(lambda.min.ratio)
    }

    if(lmr_val >= 1 | lmr_val <= 0)
    {
        stop("lambda.min.ratio must be within (0, 1)")
    }

    lambda           <- lambda_val
    nlambda          <- as.integer(nlambda[1])
    lambda.min.ratio <- lmr_val


    if(maxit <= 0)
    {
        stop("maxit should be positive")
    }
    if(abs.tol < 0 | rel.tol < 0)
    {
        stop("abs.tol and rel.tol should be nonnegative")
    }

    #if(isTRUE(rho <= 0))
    #{
    #    stop("rho should be positive")
    #}

    maxit   <- as.integer(maxit)
    abs.tol <- as.numeric(abs.tol)
    rel.tol <- as.numeric(rel.tol)
    rho     <- if(is.null(rho))  -1.0  else  as.numeric(rho)

    res <- .Call("admm_genlasso",
                 x, y, D,
                 lambda,
                 nlambda,
                 lambda.min.ratio,
                 standardize,
                 intercept,
                 list(maxit   = maxit,
                      eps_abs = abs.tol,
                      eps_rel = rel.tol,
                      rho     = rho),
                 PACKAGE = "personalizedLong")
    res
}

## taken from glmnet
getmin <- function(lambda, cvm, cvsd)
{
    cvmin      = min(cvm, na.rm = TRUE)
    idmin      = cvm <= cvmin
    lambda.min = max(lambda[idmin], na.rm = TRUE)
    idmin      = match(lambda.min, lambda)
    semin      = (cvm + cvsd)[idmin]
    idmin      = cvm <= semin
    lambda.1se = max(lambda[idmin], na.rm = TRUE)
    list(lambda.min = lambda.min, lambda.1se = lambda.1se)
}

## taken from glmnet
cvcompute <- function(mat,weights,foldid,nlams)
{
    ###Computes the weighted mean and SD within folds, and hence the se of the mean
    wisum  <- tapply(weights, foldid, sum)
    nfolds <- max(foldid)
    outmat <- matrix(NA, nfolds, ncol(mat))
    good   <- matrix(0,  nfolds, ncol(mat))
    mat[is.infinite(mat)]=NA#just in case some infinities crept in
    for(i in seq(nfolds)){
        mati       <- mat[foldid == i,,drop = FALSE]
        wi         <- weights[foldid == i]
        outmat[i,] <- apply(mati, 2, weighted.mean, w = wi, na.rm = TRUE)
        good[i,seq(nlams[i])] <- 1
    }
    N <- apply(good,2,sum)
    list(cvraw = outmat, weights = wisum, N = N)
}

## taken from glmnet
error.bars <- function(x, upper, lower, width = 0.02, ...)
{
    xlim <- range(x)
    barw <- diff(xlim) * width
    segments(x, upper, x, lower, ...)
    segments(x - barw, upper, x + barw, upper, ...)
    segments(x - barw, lower, x + barw, lower, ...)
    range(upper, lower)
}



cv_genlasso=function(outlist,lambda,x,y,weights,foldid,type.measure,grouped,keep=FALSE){
    typenames=c(deviance="Mean-Squared Error",mse="Mean-Squared Error",mae="Mean Absolute Error")
    if(type.measure=="default")type.measure="mse"
    if(!match(type.measure,c("mse","mae","deviance"),FALSE)){
        warning("Only 'mse', 'deviance' or 'mae'  available for Gaussian models; 'mse' used")
        type.measure="mse"
    }

    predmat=matrix(NA,length(y),length(lambda))
    nfolds=max(foldid)
    nlams=double(nfolds)
    for(i in seq(nfolds)){
        which=foldid==i
        fitobj=outlist[[i]]
        #preds=predict(fitobj, Xnew = x[which,,drop=FALSE], lambda = lambda)$fit
        preds = x[which,,drop=FALSE] %*% fitobj$beta[-1,]
        nlami=length(lambda)
        predmat[which,seq(nlami)]=preds
        nlams[i]=nlami
    }

    N=length(y) - apply(is.na(predmat),2,sum)
    cvraw=switch(type.measure,
                 "mse"=(y-predmat)^2,
                 "deviance"=(y-predmat)^2,
                 "mae"=abs(y-predmat)
    )
    if( (length(y)/nfolds <3)&&grouped){
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold",call.=FALSE)
        grouped=FALSE
    }
    if(grouped){
        cvob=cvcompute(cvraw,weights,foldid,nlams)
        cvraw=cvob$cvraw;weights=cvob$weights;N=cvob$N
    }

    cvm=apply(cvraw,2,weighted.mean,w=weights,na.rm=TRUE)
    cvsd=sqrt(apply(scale(cvraw,cvm,FALSE)^2,2,weighted.mean,w=weights,na.rm=TRUE)/(N-1))
    out=list(cvm=cvm,cvsd=cvsd,name=typenames[type.measure])
    if(keep)out$fit.preval=predmat
    out
}

predict.cv.genlasso=function(object,newx,s=c("lambda.1se","lambda.min"),...){
    if(is.numeric(s))lambda=s
    else
        if(is.character(s)){
            s=match.arg(s)
            lambda=object[[s]]
        }
    else stop("Invalid form for s")
    predict(object$genlasso.fit,newx,lambda=lambda,...)$fit
}

predict.cv.gamma.genlasso=function(object,newx,s=c("lambda.1se","lambda.min"),...){
    if(is.numeric(s))lambda=s
    else
        if(is.character(s)){
            s=match.arg(s)
            lambda=object[[s]]
        }
    else stop("Invalid form for s")
    predict(object$best.cv.fit$genlasso.fit,newx,lambda=lambda,...)$fit
}

plot.cv.genlasso=function(x,sign.lambda=1,...){
    cvobj=x
    xlab="log(Lambda)"
    if(sign.lambda<0)xlab=paste("-",xlab,sep="")
    plot.args=list(x=sign.lambda*log(cvobj$lambda),y=cvobj$cvm,ylim=range(cvobj$cvup,cvobj$cvlo),xlab=xlab,ylab=cvobj$name,type="n")
    new.args=list(...)
    if(length(new.args))plot.args[names(new.args)]=new.args
    do.call("plot",plot.args)
    error.bars(sign.lambda*log(cvobj$lambda),cvobj$cvup,cvobj$cvlo,width=0.01,col="darkgrey")
    points(sign.lambda*log(cvobj$lambda),cvobj$cvm,pch=20,col="red")
    axis(side=3,at=sign.lambda*log(cvobj$lambda),labels=paste(cvobj$nz),tick=FALSE,line=0)
    abline(v=sign.lambda*log(cvobj$lambda.min),lty=3)
    abline(v=sign.lambda*log(cvobj$lambda.1se),lty=3)
    invisible()
}

plot.cv.gamma.genlasso=function(x,sign.lambda=1,...){
    cvobj=x
    xlab="Gamma"
    if(sign.lambda<0)xlab=paste("-",xlab,sep="")
    plot.args=list(x=sign.lambda*(cvobj$gamma),y=cvobj$cvm,ylim=range(cvobj$cvup,cvobj$cvlo),xlab=xlab,type="n") #ylab=cvobj$name,
    new.args=list(...)
    if(length(new.args))plot.args[names(new.args)]=new.args
    do.call("plot",plot.args)
    error.bars(sign.lambda*(cvobj$gamma),cvobj$cvup,cvobj$cvlo,width=0.01,col="darkgrey")
    points(sign.lambda*(cvobj$gamma),cvobj$cvm,pch=20,col="red")
    #axis(side=3,at=sign.lambda*(cvobj$lambda),labels=paste(cvobj$nz),tick=FALSE,line=0)
    abline(v=sign.lambda*(cvobj$gamma.min),lty=3)
    #abline(v=sign.lambda*(cvobj$lambda.1se),lty=3)
    invisible()
}

