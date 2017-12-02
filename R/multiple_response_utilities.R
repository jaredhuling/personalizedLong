#line serach for the optimal step in logistic penalty
line_schlog <- function(rho, current_fit, step_fit, Y, weight)
{
    return(sum( weight*log(1 + exp(-(2 * Y - 1) * (2 * (current_fit + rho * step_fit) - 1))) ))
}


outcome2contrast <- function(Y, X, Trt, P)
{
    n=dim(X)[1];p=dim(X)[2];q=dim(Y)[2]

    IntX = matrix(0, nrow = n, ncol = p)
    for (i in 1:n)
        IntX[i,] = X[i,] * (2 * Trt[i] - 1)
    CbX = cbind(X,(2 * Trt - 1), IntX)

    #AIPWE
    shiftY    = matrix(0, nrow = n, ncol = q)
    eff_notrt = matrix(0, nrow = n, ncol = q)
    eff_trt   = matrix(0, nrow = n, ncol = q)
    CbX_notrt = cbind(X, rep(-1, n), -X)
    CbX_trt   = cbind(X, rep(1,  n),  X)

    for (j in 1:q)
    {
        if (length(table(Y[,j])) == 2)
        {
            small_index = which(Y[,j] == min(Y[,j]))
            Y[,j][small_index]  = 0
            Y[,j][-small_index] = 1
            lasmod = cv.glmnet(y = Y[,j], x = CbX, family = "binomial")
            eff_notrt[,j] = predict(lasmod, newx = CbX_notrt, s = "lambda.min", type = "response")
            eff_trt[,j]   = predict(lasmod, newx = CbX_trt,   s = "lambda.min", type = "response")
        }
        if (length(table(Y[,j])) > 2)
        {
            lasmod = cv.glmnet(y = Y[,j], x = CbX)
            eff_notrt[,j] = predict(lasmod, newx = CbX_notrt, s = "lambda.min")
            eff_trt[,j]   = predict(lasmod, newx = CbX_trt,   s = "lambda.min")
        }
        shiftY[,j] = Y[,j] - (1 - P) * eff_trt[,j] - P * eff_notrt[,j]
    }

    Con = matrix(0, nrow = n, ncol = q) #Contrast function
    for (i in 1:n)
        for (j in 1:q)
            Con[i,j] = Trt[i] * shiftY[i,j] / P - (1 - Trt[i]) * shiftY[i,j] / (1 - P)

    sign_Con = matrix(0, nrow = n, ncol = q) #Sign of contrast function
    sign_Con[Con > 0] = 1
    W=abs(Con)

    return(list(sign_Con, W))
}
