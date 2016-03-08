#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <typeinfo>
#include <vector>
#include <assert.h>

namespace quda {

  using namespace quda::colorspinor;

//ok for staggered: nSpin = 1 will work as well. Accessors do allow this case as well.

  // copy the null-space vectors into the V-field
  template <int nSpin, int nColor, int nVec, class V, class B>
  void fill(V &out, const B &in, int v) {
    for (int parity=0; parity<out.Nparity(); parity++) {
      for (int x_cb=0; x_cb<out.VolumeCB(); x_cb++) {
	for (int s=0; s<nSpin; s++) {
	  for (int c=0; c<nColor; c++) {
	    out(parity, x_cb, s, c, v) = in(parity, x_cb, s, c);
	  }
	}
      }
    }
  }

  template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B) {
    FieldOrderCB<Float,nSpin,nColor,nVec,order> vOrder(const_cast<ColorSpinorField&>(V));
    for (int v=0; v<nVec; v++) {
      FieldOrderCB<Float,nSpin,nColor,1,order> bOrder(const_cast<ColorSpinorField&>(*B[v]));
      fill<nSpin,nColor,nVec>(vOrder, bOrder, v);
    }
  }

//for staggered: this does not include factor 2 due to parity decomposition!

  template <typename Float, int nSpin, int nColor, QudaFieldOrder order>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (Nvec == 2) {
      FillV<Float,nSpin,nColor,2,order>(V,B);
    } else if (Nvec == 4) {
      FillV<Float,nSpin,nColor,4,order>(V,B);
    } else if (Nvec == 8) {
      FillV<Float,nSpin,nColor,8,order>(V,B);
    } else if (Nvec == 12) {
      FillV<Float,nSpin,nColor,12,order>(V,B);
    } else if (Nvec == 16) {
      FillV<Float,nSpin,nColor,16,order>(V,B);
    } else if (Nvec == 20) {
      FillV<Float,nSpin,nColor,20,order>(V,B);
    } else if (Nvec == 24) {
      FillV<Float,nSpin,nColor,24,order>(V,B);
    } else if (Nvec == 48) {
      FillV<Float,nSpin,nColor,48,order>(V,B);
    } else {
      errorQuda("Unsupported Nvec %d", Nvec);
    }
  }

//ok for 2-cycle multigrid, must be extended for more complicated version.

  template <typename Float, int nSpin, QudaFieldOrder order>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (B[0]->Ncolor()*Nvec != V.Ncolor()) errorQuda("Something wrong here");

    if (B[0]->Ncolor() == 2) {
      FillV<Float,nSpin,2,order>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 3) {
      FillV<Float,nSpin,3,order>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 8) {
      FillV<Float,nSpin,8,order>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 16) {
      FillV<Float,nSpin,16,order>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 24) {
      FillV<Float,nSpin,24,order>(V,B,Nvec);
    } else {
      errorQuda("Unsupported nColor %d", B[0]->Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder order>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (V.Nspin() == 4) {
      FillV<Float,4,order>(V,B,Nvec);
    } else if (V.Nspin() == 2) {
      FillV<Float,2,order>(V,B,Nvec);
#ifdef GPU_STAGGERED_DIRAC
    } else if (V.Nspin() == 1) {
      FillV<Float,1,order>(V,B,Nvec);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", V.Nspin());
    }
  }

  template <typename Float>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (V.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      FillV<Float,QUDA_FLOAT2_FIELD_ORDER>(V,B,Nvec);
    } else if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      FillV<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(V,B,Nvec);
    } else {
      errorQuda("Unsupported field type %d", V.FieldOrder());
    }
  }

  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      FillV<double>(V,B,Nvec);
    } else if (V.Precision() == QUDA_SINGLE_PRECISION) {
      FillV<float>(V,B,Nvec);
    } else {
      errorQuda("Unsupported precision %d", V.Precision());
    }
  }

  // Creates a block-ordered version of a ColorSpinorField
  // N.B.: Only works for the V field, as we need to block spin.
  template <bool toBlock, int nVec, class Complex, class FieldOrder>
  void blockOrderV(Complex *out, FieldOrder &in,
		   const int *geo_map, const int *geo_bs, int spin_bs,
		   const cpuColorSpinorField &V) {
    //printfQuda("in.Ncolor = %d\n", in.Ncolor());
    int nSpin_coarse = in.Nspin() / spin_bs; // this is number of chiral blocks

    //Compute the size of each block
    int geoBlockSize = 1;
    for (int d=0; d<in.Ndim(); d++) geoBlockSize *= geo_bs[d];
    int blockSize = geoBlockSize * in.Ncolor() * spin_bs; // blockSize includes internal dof

    int x[QUDA_MAX_DIM]; // global coordinates
    int y[QUDA_MAX_DIM]; // local coordinates within a block (full site ordering)

    int checkLength = in.Volume() * in.Ncolor() * in.Nspin() * in.Nvec();
    int *check = new int[checkLength];
    int count = 0;

    // Run through the fine grid and do the block ordering
    for (int parity = 0; parity<in.Nparity(); parity++) {
      for (int x_cb=0; x_cb<in.VolumeCB(); x_cb++) {
	int i = parity*in.VolumeCB() + x_cb;

	// Get fine grid coordinates
	V.LatticeIndex(x, i);
	
	//Compute the geometric offset within a block 
	// (x fastest direction, t is slowest direction, non-parity ordered)
	int blockOffset = 0;
	for (int d=in.Ndim()-1; d>=0; d--) {
	  y[d] = x[d]%geo_bs[d];
	  blockOffset *= geo_bs[d];
	  blockOffset += y[d];
	}
	
	//Take the block-ordered offset from the coarse grid offset (geo_map) 
	int offset = geo_map[i]*nSpin_coarse*nVec*geoBlockSize*in.Ncolor()*spin_bs;
	
	for (int v=0; v<in.Nvec(); v++) {
	  for (int s=0; s<in.Nspin(); s++) {
	    for (int c=0; c<in.Ncolor(); c++) {
	      
	      int chirality = s / spin_bs; // chirality is the coarse spin
	      int blockSpin = s % spin_bs; // the remaining spin dof left in each block
	      
	      int index = offset +                                              // geo block
		chirality * nVec * geoBlockSize * spin_bs * in.Ncolor() + // chiral block
	                       v * geoBlockSize * spin_bs * in.Ncolor() + // vector
	                            blockOffset * spin_bs * in.Ncolor() + // local geometry
	                                          blockSpin*in.Ncolor() + // block spin
	                                                                   c;   // color

	      if (toBlock) out[index] = in(parity, x_cb, s, c, v); // going to block order
	      else in(parity, x_cb, s, c, v) = out[index]; // coming from block order
	    
	      check[count++] = index;
	    }
	  }
	}
      }

      //printf("blockOrderV done %d / %d\n", i, in.Volume());
    }
    
    if (count != checkLength) {
      errorQuda("Number of elements packed %d does not match expected value %d nvec=%d nspin=%d ncolor=%d", 
		count, checkLength, in.Nvec(), in.Nspin(), in.Ncolor());
    }

    /*
    // need non-quadratic check
    for (int i=0; i<checkLength; i++) {
      for (int j=0; j<i; j++) {
      if (check[i] == check[j]) errorQuda("Collision detected in block ordering\n");
      }
    }
    */
    delete []check;
  }


  // Creates a block-ordered version of a ColorSpinorField, with parity blocking (for staggered fields)
  // N.B.: same as above but parity are separated.
  template <bool toBlock, int nVec, class Complex, class FieldOrder>
  void blockCBOrderV(Complex *out, FieldOrder &in,
		     const int *geo_map, const int *geo_bs, int spin_bs,
		     const cpuColorSpinorField &V) {
    //Compute the size of each block
    int geoBlockSize = 1;
    for (int d=0; d<in.Ndim(); d++) geoBlockSize *= geo_bs[d];
    int blockSize = geoBlockSize * in.Ncolor(); // blockSize includes internal dof

    int x[QUDA_MAX_DIM]; // global coordinates
    int y[QUDA_MAX_DIM]; // local coordinates within a block (full site ordering)

    int checkLength = in.Volume() * in.Ncolor() * in.Nvec();
    int *check = new int[checkLength];
    int count = 0;

    // Run through the fine grid and do the block ordering
    for (int parity = 0; parity<in.Nparity(); parity++) {
      for (int x_cb=0; x_cb<in.VolumeCB(); x_cb++) {
	int i = parity*in.VolumeCB() + x_cb;

	// Get fine grid coordinates
	V.LatticeIndex(x, i);

	//Compute the geometric offset within a block 
	// (x fastest direction, t is slowest direction, non-parity ordered)
	int blockOffset = 0;
	for (int d=in.Ndim()-1; d>=0; d--) {
	  y[d] = x[d]%geo_bs[d];
	  blockOffset *= geo_bs[d];
	  blockOffset += y[d];
	}

	//Take the block-ordered offset from the coarse grid offset (geo_map) 
	//A.S.: geo_map introduced for the full site ordering, so ok to use it for the offset
	int offset = geo_map[i]*nVec*geoBlockSize*in.Ncolor();

	const int s = 0;

	for (int v=0; v<in.Nvec(); v++) {
	  for (int c=0; c<in.Ncolor(); c++) {

	    int chirality = (x[0]+x[1]+x[2]+x[3])%2; // chirality is the fine-grid parity flag

	    int index = offset +                                // geo block
	      chirality * nVec * geoBlockSize * in.Ncolor() + // chiral block
	                     v * geoBlockSize * in.Ncolor() + // vector
	                          blockOffset * in.Ncolor() + // local geometry
	                                                       c;   // color

	    if (toBlock) out[index] = in(parity, x_cb, s, c, v); // going to block order
	    else in(parity, x_cb, s, c, v) = out[index]; // coming from block order

	    check[count++] = index;
	  }
	}

	//printf("blockOrderV done %d / %d\n", i, in.Volume());
      } // x_cb
    } // parity

    if (count != checkLength) {
      errorQuda("Number of elements packed %d does not match expected value %d nvec=%d ncolor=%d", 
		count, checkLength, in.Nvec(), in.Ncolor());
    }

    delete []check;
  }




  // Orthogonalise the nc vectors v[] of length n
  // this assumes the ordering v[(b * Nvec + v) * blocksize + i]

  template <typename sumFloat, typename Float, int N>
  void blockGramSchmidt(complex<Float> *v, int nBlocks, int blockSize) {
    
    for (int b=0; b<nBlocks; b++) {
      for (int jc=0; jc<N; jc++) {
      
	for (int ic=0; ic<jc; ic++) {
	  // Calculate dot product.
	  complex<Float> dot = 0.0;
	  for (int i=0; i<blockSize; i++) 
	    dot += conj(v[(b*N+ic)*blockSize+i]) * v[(b*N+jc)*blockSize+i];
	  
	  // Subtract the blocks to orthogonalise
	  for (int i=0; i<blockSize; i++) 
	    v[(b*N+jc)*blockSize+i] -= dot * v[(b*N+ic)*blockSize+i];
	}
	
	// Normalize the block
	// nrm2 is pure real, but need to use Complex because of template.
	complex<sumFloat> nrm2 = 0.0;
	for (int i=0; i<blockSize; i++) nrm2 += norm(v[(b*N+jc)*blockSize+i]);
	sumFloat scale = nrm2.real() > 0.0 ? 1.0/sqrt(nrm2.real()) : 0.0;
	for (int i=0; i<blockSize; i++) v[(b*N+jc)*blockSize+i] *= scale;
      }


      /*      
      for (int jc=0; jc<N; jc++) {
        complex<sumFloat> nrm2 = 0.0;
        for(int i=0; i<blockSize; i++) nrm2 += norm(v[(b*N+jc)*blockSize+i]);
	//printfQuda("block = %d jc = %d nrm2 = %f\n", b, jc, nrm2.real());
      }
      */

      //printf("blockGramSchmidt done %d / %d\n", b, nBlocks);
    }

  }

  template<typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order>
  void BlockOrthogonalize(ColorSpinorField &V, const int *geo_bs, const int *geo_map, int spin_bs) {
    complex<Float> *Vblock = new complex<Float>[V.Volume()*V.Nspin()*V.Ncolor()];

    FieldOrderCB<Float,nSpin,nColor,nVec,order> vOrder(const_cast<ColorSpinorField&>(V));

    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

    int blocksize = geo_blocksize * vOrder.Ncolor() * spin_bs; 
    int chiralBlocks = (V.Nspin() == 1) ? 2 : vOrder.Nspin() / spin_bs; //always 2 for staggered. 
    int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
    
    if(V.Nspin() != 1){//FIXME : this is not good, think about a separate parameter to distinguish staggered stuff!
      printfQuda("Block Orthogonalizing %d blocks of %d length and width %d\n", numblocks, blocksize, nVec);
    
      blockOrderV<true,nVec>(Vblock, vOrder, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt<double,Float,nVec>(Vblock, numblocks, blocksize);  
      blockOrderV<false,nVec>(Vblock, vOrder, geo_map, geo_bs, spin_bs, V);    
    }
    else{
      blocksize /= chiralBlocks; //for staggered chiral block size is a parity block size   

      printfQuda("Block Orthogonalizing %d blocks of %d length and width %d\n", numblocks, blocksize, nVec);

      blockCBOrderV<true,nVec>(Vblock, vOrder, geo_map, geo_bs, spin_bs, V);
      blockGramSchmidt<double,Float,nVec>(Vblock, numblocks, blocksize);  
      blockCBOrderV<false,nVec>(Vblock, vOrder, geo_map, geo_bs, spin_bs, V);    
   
    }
    delete []Vblock;
  }


  template<typename Float, int nSpin, int nColor, QudaFieldOrder order>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, const int *geo_bs, const int *geo_map, int spin_bs) {
    if (Nvec == 2) {
      BlockOrthogonalize<Float,nSpin,nColor,2,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 4) {
      BlockOrthogonalize<Float,nSpin,nColor,4,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 8) {
      BlockOrthogonalize<Float,nSpin,nColor,8,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 12) {
      BlockOrthogonalize<Float,nSpin,nColor,12,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 16) {
      BlockOrthogonalize<Float,nSpin,nColor,16,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 20) {
      BlockOrthogonalize<Float,nSpin,nColor,20,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 24) {
      BlockOrthogonalize<Float,nSpin,nColor,24,order>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 48) {
      BlockOrthogonalize<Float,nSpin,nColor,48,order>(V, geo_bs, geo_map, spin_bs);
    } else {
      errorQuda("Unsupported nVec %d\n", Nvec);
    }
  }

  template<typename Float, int nSpin, QudaFieldOrder order>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
    if (V.Ncolor()/Nvec == 3) {
      BlockOrthogonalize<Float,nSpin,3,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    }
    else if (V.Ncolor()/Nvec == 2) {
      BlockOrthogonalize<Float,nSpin,2,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    }
    else if (V.Ncolor()/Nvec == 8) {
      BlockOrthogonalize<Float,nSpin,8,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    }
    else if (V.Ncolor()/Nvec == 16) {
      BlockOrthogonalize<Float,nSpin,16,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    }
    else if (V.Ncolor()/Nvec == 24) {
      BlockOrthogonalize<Float,nSpin,24,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    }
    else if (V.Ncolor()/Nvec == 48) {
      BlockOrthogonalize<Float,nSpin,48,order>(V, Nvec, geo_bs, geo_map, spin_bs); //for staggered, even-odd blocking presumed
    }  
    else {
      errorQuda("Unsupported nColor %d\n", V.Ncolor()/Nvec);
    }
  }

  template<typename Float, QudaFieldOrder order>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
    if (V.Nspin() == 4) {
      BlockOrthogonalize<Float,4,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    }
    else if(V.Nspin() ==2) {
      BlockOrthogonalize<Float,2,order>(V, Nvec, geo_bs, geo_map, spin_bs);
    } 
    else if (V.Nspin() == 1) {
      BlockOrthogonalize<Float,1,order>(V, Nvec, geo_bs, geo_map, 1);
    }
    else {
      errorQuda("Unsupported nSpin %d\n", V.Nspin());
    }
  }

  template<typename Float>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
  if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      BlockOrthogonalize<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else {
      errorQuda("Unsupported field order %d\n", V.FieldOrder());
    }
  }

  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
      BlockOrthogonalize<double>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Precision() == QUDA_SINGLE_PRECISION) {
      BlockOrthogonalize<float>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else {
      errorQuda("Unsupported precision %d\n", V.Precision());
    }
  }

} // namespace quda
