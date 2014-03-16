#include <transfer.h>
#include <blas_quda.h>

#include <transfer.h>
#include <multigrid.h>

#include <iostream>
#include <algorithm>
#include <vector>

namespace quda {

  Transfer::Transfer(const std::vector<ColorSpinorField*> &B, int Nvec, int *geo_bs, int spin_bs)
    : B(B), Nvec(Nvec), V(0), tmp(0), geo_bs(0), fine_to_coarse(0), coarse_to_fine(0), spin_bs(spin_bs), spin_map(0)
  {
    int ndim = B[0]->Ndim();
    this->geo_bs = new int[ndim];
    for (int d = 0; d < ndim; d++) {
      this->geo_bs[d] = geo_bs[d];
    }

    if (B[0]->X(0) == geo_bs[0]) 
      errorQuda("X-dimension length %d cannot block length %d\n", B[0]->X(0), geo_bs[0]);

    printfQuda("Transfer: using block size %d", geo_bs[0]);
    for (int d=1; d<ndim; d++) printfQuda(" x %d", geo_bs[d]);
    printfQuda("\n");

    // create the storage for the final block orthogonal elements
    ColorSpinorParam param(*B[0]); // takes the geometry from the null-space vectors

    // the ordering of the V vector is defined by these parameters and
    // the Packed functions in ColorSpinorFieldOrder

    param.nSpin = B[0]->Nspin(); // spin has direct mapping
    param.nColor = B[0]->Ncolor()*Nvec; // nColor = number of colors * number of vectors
    param.create = QUDA_ZERO_FIELD_CREATE;
    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }

    if (typeid(*B[0]) == typeid(cpuColorSpinorField)) {
      printfQuda("Transfer: creating cpu V field with basis %d\n", param.gammaBasis);
      V = new cpuColorSpinorField(param);
    } else {
      printfQuda("Transfer: creating cuda V field with basis %d\n", param.gammaBasis);
      V = new cudaColorSpinorField(param);      
    }

    printfQuda("Transfer: filling V field with zero\n");
    fillV(); // copy the null space vectors into V

    // create the storage for the intermediate temporary vector
    param.nSpin = B[0]->Nspin(); // tmp has same nSpin has the fine dimension
    param.nColor = Nvec; // tmp has nColor equal to the number null-space vectors

    printfQuda("Transfer: creating tmp field\n");
    if (typeid(*B[0]) == typeid(cpuColorSpinorField)) 
      tmp = new cpuColorSpinorField(param);
    else 
      tmp = new cudaColorSpinorField(param);      

    // allocate and compute the fine-to-coarse and coarse-to-fine site maps
    fine_to_coarse = new int[B[0]->Volume()];
    coarse_to_fine = new int[B[0]->Volume()];
    createGeoMap(geo_bs);

    // allocate the fine-to-coarse spin map
    spin_map = new int[B[0]->Nspin()];
    createSpinMap(spin_bs);

    // orthogonalize the blocks
    printfQuda("Transfer: block orthogonalizing\n");
    BlockOrthogonalize(*V, Nvec, geo_bs, fine_to_coarse, spin_bs);
    printfQuda("Transfer: V block orthonormal check %g\n", blas::norm2(*V));
  }

  Transfer::~Transfer() {
    if (spin_map) delete [] spin_map;
    if (coarse_to_fine) delete [] coarse_to_fine;
    if (fine_to_coarse) delete [] fine_to_coarse;
    if (V) delete V;
    if (tmp) delete tmp;
  }

  void Transfer::fillV() { 
    FillV(*V, B, Nvec);  //printfQuda("V fill check %e\n", norm2(*V));
  }

  static bool operator<(const int2 &a, const int2 &b) const {
    return (a.x < b.x) ? true : (a.x==b.x && a.y<b.y) ? true : false;
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
      
      //printfQuda("fine idx %d = fine (%d,%d,%d,%d), ", i, x[0], x[1], x[2], x[3]);

      // compute the corresponding coarse-grid index given the block size
      for (int d=0; d<tmp->Ndim(); d++) x[d] /= geo_bs[d];

      // compute the coarse-offset index and store in fine_to_coarse
      int k;
      coarse.OffsetIndex(k, x); // this index is parity ordered
      fine_to_coarse[i] = k;

      //printfQuda("coarse after (%d,%d,%d,%d), coarse idx %d\n", x[0], x[1], x[2], x[3], k);
    }

    // now create an inverse-like variant of this

    std::vector<int2> geo_sort(B[0]->Volume());
    for (int i=0; i<geo_sort.size(); i++) geo_sort[i] = make_int2(fine_to_coarse[i], i);
    std::sort(geo_sort.begin(), geo_sort.end());
    for (int i=0; i<geo_sort.size(); i++) coarse_to_fine[i] = geo_sort[i].y;
  }

  // compute the fine spin to coarse spin map
  void Transfer::createSpinMap(int spin_bs) {

    for (int s=0; s<B[0]->Nspin(); s++) {
      spin_map[s] = s / spin_bs;
    }

  }

  // apply the prolongator
  void Transfer::P(ColorSpinorField &out, const ColorSpinorField &in) const {

    printf("Prolongator locations: out = %d, in = %d\n", out.Location(), in.Location());

    ColorSpinorField *output = &out;
    if (out.Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam param(out);
      param.create = QUDA_ZERO_FIELD_CREATE;
      param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      param.gammaBasis = in.GammaBasis();
      output = new cpuColorSpinorField(param);
    }

    if ((output->GammaBasis() != V->GammaBasis()) || (in.GammaBasis() != V->GammaBasis()))
      errorQuda("Cannot apply prolongator using fields in a different basis from the null space (%d,%d) != %d",
		output->GammaBasis(), in.GammaBasis(), V->GammaBasis());

    Prolongate(*output, in, *V, *tmp, Nvec, fine_to_coarse, spin_map);

    if (out.Location() == QUDA_CUDA_FIELD_LOCATION) { 
      out = *output; // copy result to cuda field
      delete output; 
    }
  }

  // apply the restrictor
  void Transfer::R(ColorSpinorField &out, const ColorSpinorField &in) const {

    printf("Restrictor locations: out = %d, in = %d\n", out.Location(), in.Location());

    ColorSpinorField *input = &const_cast<ColorSpinorField&>(in);
    if (in.Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam param(in);
      param.create = QUDA_ZERO_FIELD_CREATE;
      param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      param.gammaBasis = out.GammaBasis();
      input = new cpuColorSpinorField(param);
      *input = in; // copy input to cpu field
    }

    if ((out.GammaBasis() != V->GammaBasis()) || (input->GammaBasis() != V->GammaBasis()))
      errorQuda("Cannot apply restrictor using fields in a different basis from the null space (%d,%d) != %d",
		out.GammaBasis(), input->GammaBasis(), V->GammaBasis());

    Restrict(out, *input, *V, *tmp, Nvec, fine_to_coarse, spin_map);

    if (in.Location() == QUDA_CUDA_FIELD_LOCATION) { delete input; }
  }

} // namespace quda
