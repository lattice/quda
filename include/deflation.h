#pragma once

#include <invert_quda.h>
#include <vector>
#include <complex_quda.h>

namespace quda {

  /**
     This struct contains all the metadata required to define the
     deflated solvers.  
   */
  struct DeflationParam {

    /** This points to the parameter struct that is passed into QUDA.
	We use this to set (per-level) parameters */
    QudaEigParam  &eig_global;

    /** Buffer for Ritz vectors 
        For staggered: we need to reduce dimensionality of each component?
     */
    ColorSpinorField *RV;  
    
    /** Inverse Ritz values*/
    double *invRitzVals;

     /** The Dirac operator to use for spinor deflation operation */
    DiracMatrix &matDeflation;

    /** Host  projection matrix (e.g. eigCG VH A V) */
    Complex *matProj; 

    /** projection matrix leading dimension */
    int ld;

    /** projection matrix full (maximum) dimension (n_ev*deflation_grid) */
    int tot_dim;

    /** current dimension (must match rhs_idx: if(rhs_idx < deflation_grid) curr_n_evs <= n_ev * rhs_idx) */
    int cur_dim;

    /** use inverse Ritz values for deflation (for the best performance) */            
    bool use_inv_ritz;

    /** Where to compute Ritz vectors */
    QudaFieldLocation location;

    /** Filename for where to load/store the deflation space */
    char filename[100];

    DeflationParam(QudaEigParam &param, ColorSpinorField *RV,  DiracMatrix &matDeflation, int cur_dim = 0) : eig_global(param), RV(RV), matDeflation(matDeflation), 
             cur_dim(cur_dim), use_inv_ritz(false), location(param.location) {

        if(param.nk == 0 || param.np == 0 || (param.np % param.nk != 0)) errorQuda("\nIncorrect deflation space parameters...\n");
        // redesign: param.nk => param.n_ev, param.np => param.deflation_grid*param.n_ev;
        tot_dim      = param.np;
        ld           = ((tot_dim+15) / 16) * tot_dim;
        //allocate deflation resources:
        matProj = static_cast<Complex *>(pool_pinned_malloc(ld * tot_dim * sizeof(Complex)));
        invRitzVals  = new double[tot_dim];

        //Check that RV is a composite field:
        if(RV->IsComposite() == false) errorQuda("\nRitz vectors must be contained in a composite field.\n");
     }

     ~DeflationParam(){
       pool_pinned_free(matProj);
       if (invRitzVals) delete[] invRitzVals;
     }
  };

  /**
     Deflation methods :
   */
  class Deflation {

  private:
    /** Local copy of the deflation metadata */
    DeflationParam &param;

    /** TimeProfile for this level */
    TimeProfile profile;

    /** High precision aux field */
    ColorSpinorField *r;

    /** High precision aux field */
    ColorSpinorField *Av;

    /** Ritz precision residual vector */
    ColorSpinorField *r_sloppy;

    /** Deflation matrix operation result */
    ColorSpinorField *Av_sloppy;


  public:
    /** 
      Constructor for Deflation class
      @param param DeflationParam struct that defines all meta data
      @param profile Timeprofile instance used to profile
    */
    Deflation(DeflationParam &param, TimeProfile &profile);

    /**
       Destructor for Deflation class. Frees any existing Deflation
       instance
     */
    virtual ~Deflation();

    /**
       This method verifies the correctness of the MG method.  It checks:
       1. eigen-vector accuracy
       2. ... 
     */
    void verify();

    /**
       In the incremental eigcg: expands deflation space.
       @param V container of new eigenvectors
       @param n_ev number of vectors to load
     */
    void increment(ColorSpinorField &V, int n_ev);

    /**
       In the incremental eigcg: reduce deflation space
       based on the following criteria:
       @param tol : keep all eigenvectors with residual norm less then tol
       @param max_n_ev : keep the lowest max_n_ev eigenvectors (conservative)
     */
    void reduce(double tol, int max_n_ev);

    /**
       This applies deflation operation on a given spinor vector(s)
       @param out The projected vector
       @param in The input vector (or equivalently the right hand side vector)
     */
    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @brief Test whether the deflation space is complete
       and therefore cannot be further extended      
     */
    bool is_complete() {return (param.cur_dim == param.tot_dim);}

    /**
       @brief return deflation space size
     */
    int size() {return param.cur_dim;}


    /**
       @brief Return the total flops done on this and all coarser levels.
     */
    double flops() const;

  };

  /**
     Following the multigrid design, this is an object that captures an entire deflation operations.  
     A bit of a hack at the moment, this is used to allow us
     to store and reuse the deflation stuff between solves.  This is use by
     the newDeflationQuda and destroyDeflationQuda interface functions.
   */
  struct deflated_solver {

    Dirac  *d;
    DiracMatrix *m;

    ColorSpinorField *RV;//Ritz vectors

    DeflationParam *deflParam;

    Deflation *defl;
    TimeProfile &profile;

    deflated_solver(QudaEigParam &eig_param, TimeProfile &profile);

    virtual ~deflated_solver()
    {
      profile.TPSTART(QUDA_PROFILE_FREE);

      if (defl) delete defl;
      if (deflParam) delete deflParam;

      if (RV) delete RV;

      if (m) delete m;
      if (d) delete d;

      profile.TPSTOP(QUDA_PROFILE_FREE);
    }
  };

} // namespace quda


