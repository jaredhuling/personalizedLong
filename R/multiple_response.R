#' General Multivariate Boosting Algorithm
#'
#' @param Y Outcome Matrix.
#' Each row is an observation, each column corresponds to a different outcome
#' @param W Weight matrix.
#' Each row is an observation, each column corresponds to a ooutcome.
#' @param X Covariate Matrix
#' Each row is an observation, each column corresponds to a covariate
#' @param maxiter Maximum number of iterations
#' @param mu stepsize
#' @param gamma ratio of fused lasso to lasso tuning parameter
#' @param blearner The base learner to use in boosting. "spline" for P-spline with B-spline basis. "linear" for linear model. "stump" for stump.
#' @param lossf Loss function to minimize. "logistic" for logistic loss, only works if all the outcomes are binary with 0 and 1. "square"
#' for square loss, it works for all type of responses.
#' @return A object contraining information of the boosting algorithm
#' @import rpart
#' @export
Multiboost <- function(Y,W,X,maxiter,mu,lossf=c("logistic","square"),blearner=c("spline","linear","stump")){
    n=dim(X)[1];p=dim(X)[2];q=dim(Y)[2]
    mod_list=rep(list(list()),maxiter)
    cov_list=numeric(maxiter)
    res_list=numeric(maxiter)
    loss_reduce=numeric(maxiter)
    stepsize_list=numeric(maxiter)
    loss_red_ctr=matrix(0,nrow=p,ncol=q)
    current_fit<-matrix(0,nrow=n,ncol=q)
    if (lossf=="logistic")
    {
        for (j in 1:q)
            current_fit[,j]=optim(0.5,line_schlog,current_fit=rep(0,n),step_fit=rep(1,n),Y=Y[,j],weight=W[,j],method="BFGS")$par
        current_loss=sum(W*log(1+exp(-(2*Y-1)*(2*current_fit-1))))
        M=1
        while(M<=maxiter){
            presid=(2*Y-1)/(exp((2*Y-1)*(2*current_fit-1))+1)
            loss=matrix(0,nrow=p,ncol=q)
            for (i in 1:p)
                for (j in 1:q){
                    if (blearner=="linear"){
                        tempdata=as.data.frame(list(tempx=X[,i],tempy=presid[,j]))
                        mod=lm(tempy~tempx,data=tempdata,weights=W[,j])
                        mod_fit=mod$fitted
                    }
                    if (blearner=="spline"){
                        tempdata=as.data.frame(list(tempx=X[,i],tempy=presid[,j]))
                        mod=smooth.spline(y=tempdata$tempy,x=tempdata$tempx,w=W[,j],df=4)
                        mod_fit=predict(mod,x=tempdata$tempx)[[2]]
                    }
                    if (blearner=="stump"){
                        tempdata=as.data.frame(list(tempx=X[,i],tempy=presid[,j]))
                        mod=rpart(tempy~tempx,data=tempdata,weights=W[,j],control=rpart.control(maxdepth=1))
                        mod_fit=predict(mod)
                    }
                    loss[i,j]=sum(W[,j]*(presid[,j]-mod_fit)^2)/sum(W[,j]*presid[,j]^2)
                }

            cov_sel=which.min(rowSums(loss))
            res_sel=which.min(loss[cov_sel,])

            if (blearner=="linear"){
                tempdata=as.data.frame(list(tempx=X[,cov_sel],tempy=presid[,res_sel]))
                mod_sel=lm(tempy~tempx,data=tempdata,weights=W[,res_sel])
                step_fit=mod_sel$fitted
            }
            if (blearner=="spline"){
                tempdata=as.data.frame(list(tempx=X[,cov_sel],tempy=presid[,res_sel]))
                mod_sel=smooth.spline(y=tempdata$tempy,x=tempdata$tempx,w=W[,res_sel],df=4)
                step_fit=predict(mod_sel,x=tempdata$tempx)[[2]]
            }
            if (blearner=="stump"){
                tempdata=as.data.frame(list(tempx=X[,cov_sel],tempy=presid[,res_sel]))
                mod_sel=rpart(tempy~tempx,data=tempdata,weights=W[,res_sel],control=rpart.control(maxdepth=1))
                step_fit=predict(mod_sel)
            }
            opt_stepsize=optim(0,line_schlog,current_fit=current_fit[,res_sel],step_fit=step_fit,Y=Y[,res_sel],weight=W[,res_sel],method="BFGS")$par
            current_fit[,res_sel]=current_fit[,res_sel]+step_fit*opt_stepsize*mu
            loss_reduce[M]=current_loss-sum(W*log(1+exp(-(2*Y-1)*(2*current_fit-1))))
            loss_red_ctr[cov_sel,res_sel]=loss_red_ctr[cov_sel,res_sel]+loss_reduce[M]
            current_loss=sum(W*log(1+exp(-(2*Y-1)*(2*current_fit-1))))
            mod_list[[M]]=mod_sel
            cov_list[M]=cov_sel
            res_list[M]=res_sel
            stepsize_list[M]=opt_stepsize
            M=M+1
        }
        initial=numeric(q)
        for (j in 1:q)
            initial[j]=optim(0.5,line_schlog,current_fit=rep(0,n),step_fit=rep(1,n),Y=Y[,j],weight=W[,j],method="BFGS")$par
    }
    if (lossf=="square")
    {
        for (j in 1:q)
            current_fit[,j]=sum(Y[,j]*W[,j])/sum(W[,j])
        current_loss=sum(W*(Y-current_fit)^2)
        M=1
        while(M<=maxiter){
            presid=Y-current_fit
            loss=matrix(0,nrow=p,ncol=q)
            for (i in 1:p)
                for (j in 1:q){
                    if (blearner=="linear"){
                        tempdata=as.data.frame(list(tempx=X[,i],tempy=presid[,j]))
                        mod=lm(tempy~tempx,data=tempdata,weights=W[,j])
                        mod_fit=mod$fitted
                    }
                    if (blearner=="spline"){
                        tempdata=as.data.frame(list(tempx=X[,i],tempy=presid[,j]))
                        mod=smooth.spline(y=tempdata$tempy,x=tempdata$tempx,w=W[,j],df=4)
                        mod_fit=predict(mod,x=tempdata$tempx)[[2]]
                    }
                    if (blearner=="stump"){
                        tempdata=as.data.frame(list(tempx=X[,i],tempy=presid[,j]))
                        mod=rpart(tempy~tempx,data=tempdata,weights=W[,j],control=rpart.control(maxdepth=1))
                        mod_fit=predict(mod)
                    }
                    loss[i,j]=sum(W[,j]*(presid[,j]-mod_fit)^2)/sum(W[,j]*presid[,j]^2)
                }

            cov_sel=which.min(rowSums(loss))
            res_sel=which.min(loss[cov_sel,])

            if (blearner=="linear"){
                tempdata=as.data.frame(list(tempx=X[,cov_sel],tempy=presid[,res_sel]))
                mod_sel=lm(tempy~tempx,data=tempdata,weights=W[,res_sel])
                step_fit=mod_sel$fitted
            }
            if (blearner=="spline"){
                tempdata=as.data.frame(list(tempx=X[,cov_sel],tempy=presid[,res_sel]))
                mod_sel=smooth.spline(y=tempdata$tempy,x=tempdata$tempx,w=W[,res_sel],df=4)
                step_fit=predict(mod_sel,x=tempdata$tempx)[[2]]
            }
            if (blearner=="stump"){
                tempdata=as.data.frame(list(tempx=X[,cov_sel],tempy=presid[,res_sel]))
                mod_sel=rpart(tempy~tempx,data=tempdata,weights=W[,res_sel],control=rpart.control(maxdepth=1))
                step_fit=predict(mod_sel)
            }
            current_fit[,res_sel]=current_fit[,res_sel]+step_fit*mu
            loss_reduce[M]=current_loss-sum(W*(Y-current_fit)^2)
            loss_red_ctr[cov_sel,res_sel]=loss_red_ctr[cov_sel,res_sel]+loss_reduce[M]
            current_loss=sum(W*(Y-current_fit)^2)
            mod_list[[M]]=mod_sel
            cov_list[M]=cov_sel
            res_list[M]=res_sel
            stepsize_list[M]=1
            M=M+1
        }
        initial=numeric(q)
        for (j in 1:q)
            initial[j]=sum(W[,j]*Y[,j])/sum(W[,j])
    }
    return(list(mod_list,cov_list,res_list,stepsize_list,loss_red_ctr,loss_reduce,initial,lossf,blearner,mu))
}

#' Fits and cross validates general multivariate boosting algorithm. A number of iteration is chosen
#' by cross validation.
#' @param Y Outcome Matrix.
#' Each row is an observation, each column corresponds to a different outcome
#' @param W Weight matrix.
#' Each row is an observation, each column corresponds to a ooutcome.
#' @param X Covariate Matrix
#' Each row is an observation, each column corresponds to a covariate
#' @param maxiter Maximum number of iterations
#' @param mu stepsize
#' @param gamma ratio of fused lasso to lasso tuning parameter
#' @param blearner The base learner to use in boosting. "spline" for P-spline with B-spline basis. "linear" for linear model. "stump" for stump.
#' @param lossf Loss function to minimize. "logistic" for logistic loss, only works if all the outcomes are binary with 0 and 1. "square"
#' for square loss, it works for all type of responses.
#' @param folds Number of folds in cross-validation.
#' @return A object contraining information of the boosting algorithm
#' @import rpart
#' @export
cvMultiboost <- function(Y,W,X,maxiter,mu,lossf=c("logistic","square"),blearner=c("spline","linear","stump"),folds){
    k=length(folds)
    q=dim(Y)[2];p=dim(X)[2]
    cvloss=numeric(maxiter+1)
    current_fit=matrix(0,nrow=n,ncol=q)
    for (cv in 1:k){
        trainY=Y[-folds[[cv]],]
        trainW=W[-folds[[cv]],]
        trainX=X[-folds[[cv]],]
        leftY=Y[folds[[cv]],]
        leftW=W[folds[[cv]],]
        leftX=X[folds[[cv]],]

        mb_mod=Multiboost(Y=trainY,W=trainW,X=trainX,maxiter=maxiter,mu=mu,lossf=lossf,blearner=blearner)
        left_mean=left_ss=numeric(q)
        if (lossf=="square"){
            for (j in 1:q){
                left_mean[j]=sum(leftW[,j]*leftY[,j])/sum(leftW[,j])
                left_ss[j]=sum(leftW[,j]*(leftY[,j]-left_mean[j])^2)
                current_fit[,j]=sum(trainY[,j]*trainW[,j])/sum(trainW[,j])
                cvloss[1]=cvloss[1]+sum(leftW[,j]*(leftY[,j]-current_fit[,j])^2)/left_ss[j]
            }
        }
        if (lossf=="logistic"){
            for (j in 1:q){
                current_fit[,j]=optim(0.5,line_schlog,current_fit=rep(0,n),step_fit=rep(1,n),Y=trainY[,j],weight=trainW[,j],method="BFGS")$par
                left_ss[j]=optim(0.5,line_schlog,current_fit=rep(0,n),step_fit=rep(1,n),Y=leftY[,j],weight=leftW[,j],method="BFGS")$value
                cvloss[1]=cvloss[1]+sum(leftW[,j]*log(1+exp(-(2*leftY[,j]-1)*(2*current_fit[,j]-1))))/left_ss[j]
            }
        }

        for (k in 2:(maxiter+1))
        {
            mod_sel=mb_mod[[1]][[k-1]]
            cov_sel=mb_mod[[2]][k-1]
            res_sel=mb_mod[[3]][k-1]
            stepsize_sel=mb_mod[[4]][k-1]

            tempdata=as.data.frame(list(tempx=leftX[,cov_sel]))
            if (blearner=="linear")
                step_fit=predict(mod_sel,newdata=tempdata)
            if (blearner=="spline")
                step_fit=predict(mod_sel,x=tempdata$tempx)[[2]]
            if (blearner=="stump")
                step_fit=predict(mod_sel,newdata=tempdata)
            current_fit[,res_sel]=current_fit[,res_sel]+step_fit*mu*stepsize_sel

            if (lossf=="logistic"){
                for (j in 1:q){
                    cvloss[k]=cvloss[k]+sum(leftW[,j]*log(1+exp(-(2*leftY[,j]-1)*(2*current_fit[,j]-1))))/left_ss[j]
                }
            }
            if (lossf=="square"){
                for (j in 1:q){
                    cvloss[k]=cvloss[k]+sum(leftW[,j]*(leftY[,j]-current_fit[,j])^2)/left_ss[j]
                }
            }
        }
    }
    opt_iter=which.min(cvloss)-1

    if (opt_iter>0){
        mb_mod=Multiboost(Y=Y,W=W,X=X,maxiter=opt_iter-1,mu=mu,lossf=lossf,blearner=blearner)
        mb_mod=c(mb_mod,opt_iter)
    } else {
        initial=numeric(q)
        if (lossf=="square"){
            for (j in 1:q)
                initial[j]=sum(W[,j]*Y[,j])/sum(W[,j])
        }
        if (lossf=="logistic"){
            for (j in 1:q)
                initial[j]=optim(0.5,line_schlog,current_fit=rep(0,n),step_fit=rep(1,n),Y=Y[,j],weight=W[,j],method="BFGS")$par
        }
        mb_mod=list(initial,lossf,blearner,mu,opt_iter)
    }

    return(mb_mod)
}


#' Fit multivariate algorithm on the problem of multiple outcomes personalized medicine
#'
#' @param Y Outcome Matrix.
#' Each row is an observation, each column corresponds to a different outcome
#' @param W Weight matrix.
#' Each row is an observation, each column corresponds to a ooutcome.
#' @param X Covariate Matrix
#' Each row is an observation, each column corresponds to a covariate
#' @param maxiter Maximum number of iterations
#' @param mu stepsize in boosting
#' @param gamma ratio of fused lasso to lasso tuning parameter
#' @param blearner The base learner to use in boosting. "spline" for P-spline with B-spline basis. "linear" for linear model. "stump" for stump.
#' @param lossf Loss function to minimize. "logistic" for logistic loss, only works if all the outcomes are binary with 0 and 1. "square"
#' for square loss, it works for all type of responses.
#' @return A object contraining information of the boosting algorithm
#' @import rpart
#' @export

multioutcome <- function(Y,W,X,Trt,P,maxiter,mu,lossf=c("logistic","square"),blearner=c("spline","linear","stump")){
    n=dim(X)[1];p=dim(X)[2];q=dim(Y)[2]
    built_Con=outcome2contrast(Y,X,Trt,P)
    sign_Con=built_Con[[1]]
    W=build_Con[[2]]
    mb_mod=Multiboost(sign_Con,W,X,maxiter,mu,lossf=lossf,blearner=blearner)
    class(mb_mod)="multioutcome"
    return(mb_mod)
}


#' Fits and cross-validates multivariate algorithm on the problem of multiple outcomes personalized medicine
#'
#' @param Y Outcome Matrix.
#' Each row is an observation, each column corresponds to a different outcome
#' @param W Weight matrix.
#' Each row is an observation, each column corresponds to a ooutcome.
#' @param Trt Treatment assigned to the subjects.
#' It should be a binary vector with 1 for treatment and 0 for no treatment. Its length is the number of observations
#' @param P Probability each subject receive treatment
#' @param X Covariate Matrix
#' Each row is an observation, each column corresponds to a covariate
#' @param maxiter Maximum number of iterations
#' @param mu Stepsize in boosting
#' @param gamma ratio of fused lasso to lasso tuning parameter
#' @param blearner The base learner to use in boosting. "spline" for P-spline with B-spline basis. "linear" for linear model. "stump" for stump.
#' @param lossf Loss function to minimize. "logistic" for logistic loss, only works if all the outcomes are binary with 0 and 1. "square"
#' for square loss, it works for all type of responses.
#' @param k Number of folds in cross-validation
#' @return A object contraining information of the boosting algorithm
#' @import rpart
#' @import caret
#' @export

cvmultioutcome <- function(Y,W,X,Trt,P,maxiter,mu,lossf=c("logistic","square"),blearner=c("spline","linear","stump"),nfolds){
    n=dim(X)[1];p=dim(X)[2];q=dim(Y)[2]
    built_Con=outcome2contrast(Y,X,Trt,P)
    sign_Con=built_Con[[1]]
    W=build_Con[[2]]
    folds=createFolds[c(1:n),k=nfolds]
    mb_mod=cvMultiboost(sign_Con,W,X,maxiter,mu,lossf=lossf,blearner=blearner,folds=folds)
    class(mb_mod)="cvmultioutcome"
    return(mb_mod)
}


#' Predict a multivariate object
#'
#' @param mb_mod A object of class "multivariate"
#' @param iter The number of iterations for prediction
#' @param newdata New dataset with which to predict.
#' @return Prediction of a multivariate object at some iteration
#' @import rpart
#' @export
predict.multioutcome <- function(mb_mod,iter,newdata,...){
    if (iter==0)
        current_fit=mb_mod$initial
    if (iter>0){
        bleaner=mb_mod$blearner
        current_fit=mb_mod$initial
        for (k in 1:iter){
            mod_sel=mb_mod$mod_list[[k-1]]
            cov_sel=mb_mod$cov_list[k-1]
            res_sel=mb_mod$res_list[k-1]
            stepsize_sel=mb_mod$stepsize_list[k-1]
            mu=mb_model$mu

            tempdata=as.data.frame(list(tempx=newdata[,cov_sel]))
            if (blearner=="linear")
                step_fit=predict(mod_sel,newdata=tempdata)
            if (blearner=="spline")
                step_fit=predict(mod_sel,x=tempdata$tempx)[[2]]
            if (blearner=="stump")
                step_fit=predict(mod_sel,newdata=tempdata)
            current_fit[,res_sel]=current_fit[,res_sel]+step_fit*mu*stepsize_sel
        }
    }
    return(current_fit)
}


#' Predict a cvmultivariate object at the optimal number of iterations
#'
#' @param mb_mod A object of class "cvmultivariate"
#' @param newdata New dataset with which to predict.
#' @return Prediction of a multivariate object at optimal number of iterations
#' @import rpart
#' @export
predict.cvmultioutcome <- function(mb_mod,newdata,...){
    opt_iter=mb_mod$opt_iter
    if (opt_iter==0)
        current_fit=mb_mod$initial
    if (opt_iter>0){
        bleaner=mb_mod$blearner
        current_fit=mb_mod$initial
        for (k in 1:opt_iter){
            mod_sel=mb_mod$mod_list[[k-1]]
            cov_sel=mb_mod$cov_list[k-1]
            res_sel=mb_mod$res_list[k-1]
            stepsize_sel=mb_mod$stepsize_list[k-1]
            mu=mb_model$mu

            tempdata=as.data.frame(list(tempx=newdata[,cov_sel]))
            if (blearner=="linear")
                step_fit=predict(mod_sel,newdata=tempdata)
            if (blearner=="spline")
                step_fit=predict(mod_sel,x=tempdata$tempx)[[2]]
            if (blearner=="stump")
                step_fit=predict(mod_sel,newdata=tempdata)
            current_fit[,res_sel]=current_fit[,res_sel]+step_fit*mu*stepsize_sel
        }
    }
    return(current_fit)
}
