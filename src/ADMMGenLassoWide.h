#ifndef ADMMGENLASSOWIDE_H
#define ADMMGENLASSOWIDE_H

#include "ADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Spectra/SymEigsSolver.h"
#include "ADMMMatOp.h"
#include "utils.h"

#ifdef __AVX__
#include "Linalg/AVX.h"
#endif

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. Ax + z = c
//
// x => beta
// z => -X * beta
// A => X
// b => y
// c => 0
// f(x) => lambda * ||x||_1
// g(z) => 1/2 * ||z + b||^2
class ADMMGenLassoWide: public ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd, Eigen::VectorXd>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<Double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatR;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    MapMat datX;                  // data matrix
    MapVec datY;                  // response vector
    double sprad;                 // spectral radius of X'X
    Scalar lambda;                // L1 penalty
    Scalar lambda0;               // minimum lambda to make coefficients all zero
    const SpMat D;
    VectorXd Dbeta;               // D * beta
    
    int iter_counter;             // which iteration are we in?
    
    Vector cache_Ax;              // cache Ax
    Vector tmp;
#ifdef __AVX__
    vtrMatrixf vtrX;
#endif
    
    // x -> Ax
    void A_mult(Vector &res, SparseVector &beta)
    {
        res.noalias() = datX * beta;
    }
    // y -> A'y
    void At_mult(Vector &res, Vector &nu)
    {
        res.noalias() = datX.transpose() * nu;
    }
    // z -> Bz
    void B_mult(Vector &res, Vector &gamma)
    {
        res.swap(gamma);
    }
    // ||c||_2
    double c_norm() { return 0.0; }
    
    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);
        
        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }
    
    virtual void active_set_update(SparseVector &res)
    {
        const Double gamma = sprad;
        const Scalar penalty = lambda / (rho * gamma);
        tmp.noalias() = (cache_Ax + aux_gamma + dual_nu / Scalar(rho)) / gamma;
        res = main_beta;
        
        Double *val_ptr = res.valuePtr();
        const int *ind_ptr = res.innerIndexPtr();
        const int nnz = res.nonZeros();
        
#ifdef __AVX__
        vtrX.read_vec(tmp.data());
#endif
#pragma omp parallel for
        for(int i = 0; i < nnz; i++)
        {
#ifdef __AVX__
            const Double val = val_ptr[i] - vtrX.ith_inner_product(ind_ptr[i]);
#else
            const Double val = val_ptr[i] - tmp.dot(datX.col(ind_ptr[i]));
#endif
            
            if(val > penalty)
                val_ptr[i] = val - penalty;
            else if(val < -penalty)
                val_ptr[i] = val + penalty;
            else
                val_ptr[i] = 0.0;
        }
        
        res.prune(0.0);
    }
    
    // 4^k - 1, k = 0, 1, 2, ...
    static bool is_regular_update(unsigned int x)
    {
        if(x == 0 || x == 3 || x == 15 || x == 63)  return true;
        x++;
        if( x & (x - 1) )  return false;
        return x & 0x55555555;
    }
    
    virtual void next_beta(SparseVector &res)
    {
        if(lambda > lambda0 - 1e-5)
        {
            res.setZero();
            return;
        }
        
        // iter_counter = 0, 3, 15, 63, .... (4^k - 1)
        if(is_regular_update(iter_counter))
        {
            const Double gamma = sprad;
            tmp.noalias() = cache_Ax + aux_gamma + dual_nu / Scalar(rho);
            Vector vec(dim_main);
#ifdef __AVX__
            vtrX.trans_mult_vec(tmp, vec.data());
            vec *= (-1.0 / gamma);
#else
            vec.noalias() = -datX.transpose() * tmp / gamma;
#endif
            vec += main_beta;
            soft_threshold(res, vec, lambda / (rho * gamma));
        } else {
            active_set_update(res);
        }
        iter_counter++;
    }
    void next_gamma(Vector &res)
    {
#ifdef __AVX__
        vtrX.mult_spvec(main_beta, cache_Ax.data());
#else
        cache_Ax.noalias() = datX * main_beta;
#endif
        
        res.noalias() = (datY + dual_nu + Scalar(rho) * cache_Ax) / Scalar(-1 - rho);
    }
    void next_residual(Vector &res)
    {
        // res.noalias() = cache_Ax + aux_gamma;
        std::transform(cache_Ax.data(), cache_Ax.data() + dim_dual, aux_gamma.data(), res.data(), std::plus<Double>());
    }
    void rho_changed_action() {}
    
    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_gamma.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return std::sqrt(sprad) * dual_nu.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual(const Vector &new_gamma)
    {
        return rho * std::sqrt(sprad) * (new_gamma - aux_gamma).norm();
    }
    
public:
    ADMMGenLassoWide(ConstGenericMatrix &datX_, 
                     ConstGenericVector &datY_,
                     const SpMatR &D_,
                     double eps_abs_ = 1e-6,
                     double eps_rel_ = 1e-6) :
    ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd, Eigen::VectorXd>
            (datX_.cols(), D_.rows(), D_.rows(),
             eps_abs_, eps_rel_),
             datX(datX_.data(), datX_.rows(), datX_.cols()),
             datY(datY_.data(), datY_.size()),
             D(D_),
             Dbeta(D_.rows()),
             lambda0((datX.transpose() * datY).cwiseAbs().maxCoeff()),
             cache_Ax(dim_dual), tmp(dim_dual)
    {
        //Matrix XX;
        //Linalg::tcross_prod_lower(XX, datX);
        MatrixXd XX(XXt(datX));
        MatOpSymLower<Double> op(XX);
        Spectra::SymEigsSolver< Double, Spectra::LARGEST_ALGE, MatOpSymLower<Double> > eigs(&op, 1, 3);
        srand(0);
        eigs.init();
        eigs.compute(10, 0.1);
        Vector evals = eigs.eigenvalues();
        sprad = evals[0];
        
#ifdef __AVX__
        vtrX.read_mat(datX);
#endif
    }
    
    double get_lambda_zero() const { return lambda0; }
    
    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_beta.setZero();
        cache_Ax.setZero();
        aux_gamma.setZero();
        dual_nu.setZero();
        
        lambda = lambda_;
        rho = rho_;
        
        if(rho <= 0)
            rho = std::pow(lambda / sprad, 1.0 / 3);
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        
        iter_counter = 0;
        
        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_beta, aux_gamma, dual_nu and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        
        iter_counter = 0;
    }
};



#endif // ADMMGENLASSOWIDE_H
