#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>

namespace quda {

  //Zero out a field, using the accessor.
  template<typename Float, typename F>
  void set_zero(F &f) {
    for(int parity = 0; parity < 2; parity++) {
      for(int x_cb = 0; x_cb < f.VolumeCB(); x_cb++) {
	for(int s = 0; s < f.Nspin(); s++) {
	  for(int c = 0; c < f.Ncolor(); c++) {
	    for(int v = 0; v < f.Nvec(); v++) {
	      f(parity,x_cb,s,c,v) = (Float) 0.0;
	    }
	  }
	}
      }
    }
  }

  template<typename Float, typename Gauge>
  void createKSCoarseLocal(Gauge &X, int ndim, const int *xc_size, double k) {
    const int nColor = X.NcolorCoarse();
    const int nSpin = X.NspinCoarse();
    if (nSpin != 2) errorQuda("\nWrong coarse spin degrees.\n");

    Float _2m = (Float) k;//mass term
    complex<Float> *Xlocal = new complex<Float>[nSpin*nSpin*nColor*nColor];
	
    for (int parity=0; parity<2; parity++) {
      for (int x_cb=0; x_cb<X.VolumeCB(); x_cb++) {

	for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	  for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
	    
	    //Copy the Hermitian conjugate term to temp location 
	    for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
	      for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		//Flip s_col, s_row on the rhs because of Hermitian conjugation.  Color part left untransposed.
		Xlocal[((nSpin*s_col+s_row)*nColor+ic_c)*nColor+jc_c] = X(0,parity,x_cb,s_row, s_col, ic_c, jc_c);
	      }	
	    }
	  }
	}
	      
	for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
	  for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
            for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
              //diagonal elements
	      if(s_row == s_col){
                X(0,parity,x_cb, s_row, s_col, ic_c,ic_c) = _2m;
              }else{
                //off-diagonal elements
	        for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
		  //Transpose color part
		  X(0,parity,x_cb,s_row,s_col,ic_c,jc_c) = (-X(0,parity,x_cb,s_row,s_col,ic_c,jc_c)+conj(Xlocal[((nSpin*s_row+s_col)*nColor+jc_c)*nColor+ic_c]));
	        } //Color column
              }
	    } //Color row
	  } //Spin column
	} //Spin row

      } // x_cb
    } //parity

    delete[] Xlocal;

    return;
  }

  //added HISQ links (single GPU support)
  template<typename Float, int dir, typename F, typename fineGauge>
  void computeKSUV(F *UV, F *UVL, const F &V, const fineGauge *FL, const fineGauge *LL, int ndim, const int *x_size) 
  {
    int coord[5];
    coord[4] = 0;

    const int stag_sp = 0;
     
    for (int parity=0; parity<2; parity++) {
      for( int x_cb = 0; x_cb < V.VolumeCB(); x_cb++){
         getCoords(coord, x_cb, x_size, parity);

         int y_cb  = linkIndexP1(coord, x_size, dir);
         int y3_cb = (LL != NULL) ? linkIndexP3(coord, x_size, dir) : 0;

	 for(int ic_c = 0; ic_c < V.Nvec(); ic_c++) {  //Coarse Color
             for(int ic = 0; ic < FL->Ncolor(); ic++) { //Fine Color rows of gauge field
		 for(int jc = 0; jc < FL->Ncolor(); jc++) {  //Fine Color columns of gauge field
		    (*UV)(parity, x_cb, stag_sp, ic, ic_c) += (*FL)(dir, parity, x_cb, ic, jc) * V((parity+1)&1, y_cb, stag_sp, jc, ic_c);//mind transformation to the opposite parity field: in UVU operation.
                    if(LL != NULL) (*UVL)(parity, x_cb, stag_sp, ic, ic_c) += (*LL)(dir, parity, x_cb, ic, jc) * V((parity+1)&1, y3_cb, stag_sp, jc, ic_c);
		 }  //Fine color columns
	      }  //Fine color rows
	  }
       }// x_cb
    } // parity

    return;
  }  //UV

  //KS (also HISQ) operator:
  template<typename Float, int dir, typename F, typename coarseGauge>
  void computeKSVUV(coarseGauge &Y, coarseGauge &X, const F *UV, const F *UVL, const F &V, const int nfinecolors,
		  const int *x_size, const int *xc_size, const int *geo_bs) {

    const int nDim = 4;
    int coarse_size = 1;

    for(int d = 0; d<nDim; d++) coarse_size *= xc_size[d];

    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    // paralleling this requires care with respect to race conditions
    // on CPU, parallelize over dimension not parity
    const int stag_sp = 0;

    //#pragma omp parallel for 
    for (int parity=0; parity<2; parity++) {
      for( int x_cb = 0; x_cb < UV->VolumeCB(); x_cb++){
         getCoords(coord, x_cb, x_size, parity);

         for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/geo_bs[d];

	 //Check to see if we are on the edge of a block, i.e.
	 //if this color matrix connects adjacent blocks.  If
	 //adjacent site is in same block, M = X, else M = Y
	 bool isDiagonal = (((coord[dir]+1)%x_size[dir])/geo_bs[dir] == coord_coarse[dir]) ? true : false;
         //
	 bool isDiagonal_long = (UVL == NULL) ? false : (((coord[dir]+3)%x_size[dir])/geo_bs[dir] == coord_coarse[dir]) ? true : false;

	 coarseGauge *M =  isDiagonal ? &X : &Y;
         coarseGauge *M_L = (UVL == NULL) ? NULL : (isDiagonal_long ? &X : &Y);
	      
         const int dim_index      = isDiagonal ? 0 : dir;
         const int dim_index_long = isDiagonal_long ? 0 : dir;

	 int coarse_parity = 0;
	 for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
	 coarse_parity &= 1;
	 coord_coarse[0] /= 2;
	 int coarse_x_cb = ((coord_coarse[3]*xc_size[2]+coord_coarse[2])*xc_size[1]+coord_coarse[1])*(xc_size[0]/2) + coord_coarse[0];
	      
	 //printf("(%d,%d)\n", coarse_x_cb, coarse_parity);
	 coord[0] /= 2;

         int coarse_spin_row = parity == 0 ? 0 : 1  ;
         int coarse_spin_col = (1 - coarse_spin_row); 

         for(int ic_c = 0; ic_c < Y.NcolorCoarse(); ic_c++) { //Coarse Color row
           for(int jc_c = 0; jc_c < Y.NcolorCoarse(); jc_c++) { //Coarse Color column
	     for(int ic = 0; ic < nfinecolors; ic++) { //Sum over fine color
		(*M)(dim_index,coarse_parity,coarse_x_cb,coarse_spin_row, coarse_spin_col,ic_c,jc_c) += conj(V(parity, x_cb, stag_sp, ic, ic_c)) * (*UV)(parity, x_cb, stag_sp, ic, jc_c);
                 if(UVL != NULL) (*M_L)(dim_index_long,coarse_parity,coarse_x_cb,coarse_spin_row, coarse_spin_col,ic_c,jc_c) += conj(V(parity, x_cb, stag_sp, ic, ic_c)) * (*UVL)(parity, x_cb, stag_sp, ic, jc_c);
	     } //Fine color
	   } //Coarse Color column
	 } //Coarse Color row
      } // x_cb
    } // parity
    
    return;
  }

 //Calculates the coarse gauge field: separated from coarseSpin = 2 computations:
  template<typename Float, typename F, typename coarseGauge, typename fineGauge>
  void calculateKSY(coarseGauge &Y, coarseGauge &X, F *UV, F *UVL, F &V, fineGauge *FL, fineGauge *LL, const int *x_size, const int *xc_size,  double k) {

    if (FL->Ndim() != 4) errorQuda("Number of dimensions not supported");

    if ( LL ) if(LL->Ndim() != 4) errorQuda("Number of long links dimensions not supported");

    const int nDim = 4;

    int geo_bs[QUDA_MAX_DIM]; 
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];

    for(int d = 0; d < nDim; d++) 
    {
      //First calculate UV
      set_zero<Float,F>(*UV);
      if( LL ) set_zero<Float,F>(*UVL);

      printfQuda("Computing KS %d UV and VUV\n", d);
      //Calculate UV and then VUV for this direction, accumulating directly into the coarse gauge field Y
      if (d==0) {
        computeKSUV<Float,0>(UV, UVL, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,0>(Y, X, UV, UVL, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      } else if (d==1) {
        computeKSUV<Float,1>(UV, UVL, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,1>(Y, X, UV, UVL, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      } else if (d==2) {
        computeKSUV<Float,2>(UV, UVL, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,2>(Y, X, UV, UVL, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      } else {
        computeKSUV<Float,3>(UV, UVL, V, FL, LL, nDim, x_size);
        computeKSVUV<Float,3>(Y, X, UV, UVL, V, FL->Ncolor(), x_size, xc_size, geo_bs);
      }

      printf("KS UV2[%d] = %e\n", d, UV->norm2());
      printf("KS Y2[%d] = %e\n", d, Y.norm2(d));
    }

    printf("KS X2 = %e\n", X.norm2(0));
    printfQuda("Computing coarse diagonal\n");
    createKSCoarseLocal<Float>(X, nDim, xc_size, k);

    printf("KS X2 = %e\n", X.norm2(0));

  }



  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int coarseColor, int coarseSpin>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double k) {

    const int fineSpin = 1;

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseSpin*coarseColor,coarseSpin,gOrder> gCoarse;

    const ColorSpinorField &v = T.Vectors();
    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F uvAccessor(const_cast<ColorSpinorField&>(*uv));
    gFine fAccessor(const_cast<GaugeField&>(*f));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X));

    if(l != NULL) {
      gFine lAccessor(const_cast<GaugeField&>(*l));
      F uvlAccessor(const_cast<ColorSpinorField&>(*uv_long));
      calculateKSY<Float>(yAccessor, xAccessor, &uvAccessor, &uvlAccessor, vAccessor, &fAccessor, &lAccessor, f->X(), Y.X(), k);
    }
    else {
      gFine *lAccessor = NULL;
      F *uvlAccessor = NULL;
      calculateKSY<Float>(yAccessor, xAccessor, &uvAccessor, uvlAccessor, vAccessor, &fAccessor, lAccessor, f->X(),Y.X(), k);
    }    
  }

  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double k) {

    if ((T.Vectors().Nspin() != 1) && (T.Vectors().Nspin()/T.Spin_bs() != 2))  errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateKSY<Float,csOrder,gOrder,fineColor,2, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 4) {
      calculateKSY<Float,csOrder,gOrder,fineColor,4, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 8) {
      calculateKSY<Float,csOrder,gOrder,fineColor,8, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 12) {
      calculateKSY<Float,csOrder,gOrder,fineColor,12, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 16) {
      calculateKSY<Float,csOrder,gOrder,fineColor,16, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 20) {
      calculateKSY<Float,csOrder,gOrder,fineColor,20, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 24) {
      calculateKSY<Float,csOrder,gOrder,fineColor,24, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 48) {
      calculateKSY<Float,csOrder,gOrder,fineColor,48, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else if (coarseColor == 96) {
      calculateKSY<Float,csOrder,gOrder,fineColor,96, coarseSpin>(Y, X, uv, uv_long, T, f, l, k);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }


  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (f->Ncolor() == 3) {
      if( l ) if( f->Ncolor() != l->Ncolor() ) errorQuda("Unsupported number of colors %d\n", l->Ncolor());

      calculateKSY<Float,csOrder,gOrder, 3>(Y, X, uv, uv_long, T, f, l, k);
    } else {
      errorQuda("Unsupported number of colors %d\n", f->Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (f->FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      if( l ) if( l->FieldOrder() != QUDA_QDP_GAUGE_ORDER ) errorQuda("Unsupported field order for long links %d\n", l->FieldOrder());

      calculateKSY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, uv, uv_long, T, f, l, k);
    } else {
      errorQuda("Unsupported field order %d\n", f->FieldOrder());
    }
  }

 template <typename Float>
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {

      calculateKSY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, uv, uv_long, T, f, l, k);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateKSY(GaugeField &Y, GaugeField &X, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double k) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv->Precision() ||
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != f->Precision())
    {
      errorQuda("Unsupported precision mix");
    }

    if( l )
    { 
      if(Y.Precision() != l->Precision() || Y.Precision() != uv_long->Precision()) errorQuda("Unsupported precision mix for long links.");
    }

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateKSY<double>(Y, X, uv, uv_long, T, f, l, k);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateKSY<float>(Y, X, uv, uv_long, T, f, l, k);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if(l)
     printfQuda("....done computing Y field for improved staggered operator\n");  
    else 
     printfQuda("....done computing Y field for naive staggered operator\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseKSOp(const Transfer &T, GaugeField &Y, GaugeField &X, const cudaGaugeField *fat_links, const cudaGaugeField *long_links,  double k) {
    QudaPrecision precision = Y.Precision();
    //First make a cpu gauge field from the cuda gauge field

    int pad = 0;
    GaugeFieldParam fat_param(fat_links->X(), precision, fat_links->Reconstruct(), pad, fat_links->Geometry());
    fat_param.order = QUDA_QDP_GAUGE_ORDER;
    fat_param.fixed = fat_links->GaugeFixed();
    fat_param.link_type = fat_links->LinkType();
    fat_param.t_boundary = fat_links->TBoundary();
    fat_param.anisotropy = fat_links->Anisotropy();
    fat_param.gauge = NULL;
    fat_param.create = QUDA_NULL_FIELD_CREATE;
    fat_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    cpuGaugeField *f = new cpuGaugeField(fat_param);
    cpuGaugeField *l = NULL;

    //Copy the cuda gauge field to the cpu
    fat_links->saveCPUField(*f, QUDA_CPU_FIELD_LOCATION);

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField *uv = new cpuColorSpinorField(UVparam);

    cpuColorSpinorField *uv_long = NULL;

    if(long_links)
    {
      GaugeFieldParam long_param(long_links->X(), precision, long_links->Reconstruct(), pad, long_links->Geometry());
      long_param.order = QUDA_QDP_GAUGE_ORDER;
      long_param.fixed = long_links->GaugeFixed();
      long_param.link_type = long_links->LinkType();
      long_param.t_boundary = long_links->TBoundary();
      long_param.anisotropy = long_links->Anisotropy();
      long_param.gauge = NULL;
      long_param.create = QUDA_NULL_FIELD_CREATE;
      long_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      //
      l = new cpuGaugeField(fat_param);
      //
      uv_long = new cpuColorSpinorField(UVparam);
      //Copy the cuda gauge field to the cpu
      long_links->saveCPUField(*l, QUDA_CPU_FIELD_LOCATION);
    }

    //If the fine lattice operator is the clover operator, copy the cudaCloverField to cpuCloverField
    calculateKSY(Y, X, uv, uv_long, T, f, l, k);

    // now exchange Y halos for multi-process dslash
    Y.exchangeGhost();

    delete uv; 
    delete f;

    if(l)
    { 
      delete l;
      delete uv_long;
    }

  }

} //namespace quda
