#include <transfer.h>
#include <blas_quda.h>

namespace quda {

  /*template<class FO>
  void Transfer::testOrthogonal(const FO &in, int* geo_bs, int spin_bs) {
    int geo_blocksize = 1;
    for (int d = 0; d < V->Ndim(); d++) {
      geo_blocksize *= geo_bs[d];
    }
    int numblocks = V->Volume()*4*Nvec/(spin_bs*geo_blocksize);
    std::complex<double> * block;
    block = (std::complex<double> *)malloc(numblocks*sizeof(std::complex<double>));
    int *count = (int *) malloc(numblocks*sizeof(int));
    for(int i =0; i < numblocks; i++) {
      block[i] = 0.0;
      count[i] = 0;
    }
    for(int i = 0; i < V->Volume(); i++) {
      for(int s = 0; s < V->Nspin(); s++) {
	for(int c = 0; c < V->Ncolor(); c++) {
	  int x[QUDA_MAX_DIM];
	  V->LatticeIndex(x,i);
	  int offset = geo_map[i]*4*Nvec/spin_bs;
	  block[offset+(c/spin_bs)] += std::conj(in(i,s,c)) * (in(i,s,c));
	  count[offset+(c/spin_bs)]++;
	  
	}
      }
    }
    for(int i =0; i < numblocks; i++) {
      printfQuda("count[%d] = %d block[%d] = %e %e\n",i,count[i],i,block[i].real(),block[i].imag());
    }
    free(block);
    free(count);
    
  }
  */

  Transfer::Transfer(cpuColorSpinorField **B, int Nvec, int *geo_bs, int spin_bs)
    : B(B), Nvec(Nvec), V(0), tmp(0), geo_map(0), spin_map(0) 
  {

    // create the storage for the final block orthogonal elements
    ColorSpinorParam param(*B[0]); // takes the geometry from the null-space vectors
    param.nSpin = B[0]->Ncolor(); // the spin dimension corresponds to fine nColor
    param.nColor = B[0]->Nspin() * Nvec; // nColor = number of spin components * number of null-space vectors
    param.create = QUDA_ZERO_FIELD_CREATE;
    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }
    V = new cpuColorSpinorField(param);
    fillV(); // copy the null space vectors into V

    // create the storage for the intermediate temporary vector
    param.nSpin = B[0]->Nspin(); // tmp has same nSpin has the fine dimension
    param.nColor = Nvec; // tmp has nColor equal to the number null-space vectors
    tmp = new cpuColorSpinorField(param);

    // allocate and compute the fine-to-coarse site map
    geo_map = new int[B[0]->Volume()];
    createGeoMap(geo_bs);

    // allocate the fine-to-coarse spin map
    spin_map = new int[B[0]->Nspin()];
    createSpinMap(spin_bs);

    //printfQuda("numblocks = %d fsite_length = %d geo_blocksize = %d\n", numblocks, fsite_length, geo_blocksize);

    // orthogonalize the blocks
    BlockOrthogonalize(*V, Nvec, geo_bs, geo_map, spin_bs);
    printfQuda("V block orthonormal check %e\n", norm2(*V));
  }

  Transfer::~Transfer() {
    if (spin_map) delete [] spin_map;
    if (geo_map) delete [] geo_map;
    if (V) delete V;
    if (tmp) delete tmp;
  }

  void Transfer::fillV() {
    FillV(*V, (const ColorSpinorField**)B, Nvec);
    printfQuda("V fill check %e\n", norm2(*V));
  }

  // compute the fine-to-coarse site map
  void Transfer::createGeoMap(int *geo_bs) {

    int x[QUDA_MAX_DIM];

    // create a spinor with coarse geometry so we can use its OffsetIndex member function
    ColorSpinorParam param(*tmp);
    param.nColor = 1;
    param.nSpin = 1;
    param.create = QUDA_ZERO_FIELD_CREATE;
    for (int d=0; d<param.nDim; d++) param.x[d] /= geo_bs[d];
    cpuColorSpinorField coarse(param);

    //std::cout << coarse;

    // compute the coarse grid point for every site (assuming parity ordering currently)
    for (int i=0; i<tmp->Volume(); i++) {
      // compute the lattice-site index for this offset index
      tmp->LatticeIndex(x, i);

      //printf("fine idx %d = fine (%d,%d,%d,%d), ", i, x[0], x[1], x[2], x[3]);

      // compute the corresponding coarse-grid index given the block size
      for (int d=0; d<tmp->Ndim(); d++) x[d] /= geo_bs[d];

      // compute the coarse-offset index and store in the geo_map
      int k;
      coarse.OffsetIndex(k, x); // this index is parity ordered
      geo_map[i] = k;

      //printf("coarse (%d,%d,%d,%d), coarse idx %d\n", x[0], x[1], x[2], x[3], k);
    }

  }

  // compute the fine spin to coarse spin map
  void Transfer::createSpinMap(int spin_bs) {

    for (int s=0; s<B[0]->Nspin(); s++) {
      spin_map[s] = s / spin_bs;
    }

  }

  // apply the prolongator
  void Transfer::P(cpuColorSpinorField &out, const cpuColorSpinorField &in) {
    Prolongate(out, in, *V, *tmp, geo_map, spin_map);
  }

  // apply the restrictor
  void Transfer::R(cpuColorSpinorField &out, const cpuColorSpinorField &in) {
    Restrict(out, in, *V, *tmp, geo_map, spin_map);
  }

} // namespace quda
