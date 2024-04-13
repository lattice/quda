
#include <transfer.h>

#include <blas_quda.h>

#include <transfer.h>
#include <multigrid.h>
#include <tune_quda.h>
#include <malloc_quda.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>

namespace quda {

  /*
  * for the staggered case, there is no spin blocking, 
  * however we do even-odd to preserve chirality (that is straightforward)
  */
  Transfer::Transfer(const std::vector<ColorSpinorField *> &B, int Nvec, int n_block_ortho, bool block_ortho_two_pass,
                     int *geo_bs, int spin_bs, QudaPrecision null_precision, const QudaTransferType transfer_type,
                     TimeProfile &profile) :
    B(B),
    Nvec(Nvec),
    NblockOrtho(n_block_ortho),
    blockOrthoTwoPass(block_ortho_two_pass),
    null_precision(null_precision),
    V_h(nullptr),
    V_d(nullptr),
    fine_tmp_h(nullptr),
    fine_tmp_d(nullptr),
    coarse_tmp_h(nullptr),
    coarse_tmp_d(nullptr),
    geo_bs(nullptr),
    fine_to_coarse_h(nullptr),
    coarse_to_fine_h(nullptr),
    fine_to_coarse_d(nullptr),
    coarse_to_fine_d(nullptr),
    spin_bs(spin_bs),
    spin_map(0),
    nspin_fine(B[0]->Nspin()),
    site_subset(QUDA_FULL_SITE_SUBSET),
    parity(QUDA_INVALID_PARITY),
    enable_gpu(false),
    enable_cpu(false),
    use_gpu(true),
    transfer_type(transfer_type),
    flops_(0),
    profile(profile)
  {
    postTrace();
    int ndim = B[0]->Ndim();

    // Only loop over four dimensions for now, we don't have
    // to worry about the fifth dimension until we hit chiral fermions.
    for (int d = 0; d < 4; d++) {
      while (geo_bs[d] > 0) {
      	if (d==0 && B[0]->X(0) == geo_bs[0])
      	  warningQuda("X-dimension length %d cannot block length %d", B[0]->X(0), geo_bs[0]);
      	else if ( (B[0]->X(d)/geo_bs[d]+1)%2 == 0)
      	  warningQuda("Indexing does not (yet) support odd coarse dimensions: X(%d) = %d", d, B[0]->X(d)/geo_bs[d]);
      	else if ( (B[0]->X(d)/geo_bs[d]) * geo_bs[d] != B[0]->X(d) )
      	  warningQuda("cannot block dim[%d]=%d with block size = %d", d, B[0]->X(d), geo_bs[d]);
      	else
      	  break; // this is a valid block size so let's use it
      	geo_bs[d] /= 2;
      }
      if (geo_bs[d] == 0) errorQuda("Unable to block dimension %d", d);
    }

    if (ndim > 4) errorQuda("Number of dimensions %d not supported", ndim);

    this->geo_bs = new int[ndim];
    int total_block_size = 1;
    for (int d = 0; d < ndim; d++) {
      this->geo_bs[d] = geo_bs[d];
      total_block_size *= geo_bs[d];
    }

    // Various consistency checks for optimized KD "transfers"
    if (transfer_type == QUDA_TRANSFER_OPTIMIZED_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {

      // Aggregation size is "technically" 1 for optimized KD
      if (total_block_size != 1)
        errorQuda("Invalid total geometric block size %d for transfer type optimized-kd, must be 1", total_block_size);

      // The number of coarse dof is technically fineColor for optimized KD
      if (Nvec != B[0]->Ncolor())
        errorQuda("Invalid Nvec %d for optimized-kd aggregation, must be fine color %d", Nvec, B[0]->Ncolor());

    } else {
      int aggregate_size = total_block_size * B[0]->Ncolor();
      if (spin_bs == 0)
        aggregate_size /= 2; // effective spin_bs of 0.5 (fine spin / coarse spin)
      else
        aggregate_size *= spin_bs;
      if (Nvec > aggregate_size)
        errorQuda("Requested coarse space %d larger than aggregate size %d", Nvec, aggregate_size);
    }

    std::string block_str = std::to_string(geo_bs[0]);
    for (int d = 1; d < ndim; d++) block_str += " x " + std::to_string(geo_bs[d]);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transfer: using block size %s\n", block_str.c_str());

    if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      for (int d = 0; d < 4; d++) {
        if (geo_bs[d] != 2) errorQuda("Invalid staggered KD block size %d for dimension %d, must be 2", geo_bs[d], d);
      }
      if (Nvec != 24) errorQuda("Invalid number of coarse vectors %d for staggered KD multigrid, must be 24", Nvec);
    }

    createV(B[0]->Location()); // allocate V field
    createTmp(QUDA_CPU_FIELD_LOCATION); // allocate temporaries

    // allocate and compute the fine-to-coarse and coarse-to-fine site maps
    fine_to_coarse_h = static_cast<int*>(pool_pinned_malloc(B[0]->Volume()*sizeof(int)));
    coarse_to_fine_h = static_cast<int*>(pool_pinned_malloc(B[0]->Volume()*sizeof(int)));

    if (enable_gpu) {
      fine_to_coarse_d = static_cast<int*>(pool_device_malloc(B[0]->Volume()*sizeof(int)));
      coarse_to_fine_d = static_cast<int*>(pool_device_malloc(B[0]->Volume()*sizeof(int)));
    }

    createGeoMap(geo_bs);

    // allocate the fine-to-coarse spin map
    spin_map = static_cast<int**>(safe_malloc(nspin_fine*sizeof(int*)));
    for (int s = 0; s < B[0]->Nspin(); s++) spin_map[s] = static_cast<int*>(safe_malloc(2*sizeof(int)));
    createSpinMap(spin_bs);

    reset();
    postTrace();
  }

  void Transfer::createV(QudaFieldLocation location) const
  {
    postTrace();

    // create the storage for the final block orthogonal elements
    ColorSpinorParam param(*B[0]); // takes the geometry from the null-space vectors

    // the ordering of the V vector is defined by these parameters and
    // the Packed functions in ColorSpinorFieldOrder

    param.nSpin = B[0]->Nspin(); // spin has direct mapping
    param.nColor = B[0]->Ncolor()*Nvec; // nColor = number of colors * number of vectors
    param.nVec = Nvec;
    param.create = QUDA_NULL_FIELD_CREATE;
    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      //keep it the same for staggered:
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }
    param.location = location;
    param.fieldOrder = location == QUDA_CUDA_FIELD_LOCATION ? colorspinor::getNative(null_precision, param.nSpin) :
                                                              QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.setPrecision(location == QUDA_CUDA_FIELD_LOCATION ? null_precision : B[0]->Precision());

    if (transfer_type == QUDA_TRANSFER_COARSE_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD
        || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      // Need to create V_d and V_h as metadata containers, but we don't
      // actually need to allocate the memory.
      param.create = QUDA_REFERENCE_FIELD_CREATE;
    }

    if (location == QUDA_CUDA_FIELD_LOCATION) {
      V_d = new ColorSpinorField(param);
      enable_gpu = true;
    } else {
      V_h = new ColorSpinorField(param);
      enable_cpu = true;
    }
    postTrace();
  }

  void Transfer::createTmp(QudaFieldLocation location) const
  {
    // The CPU temporaries are needed for creating geometry mappings.
    if ((transfer_type == QUDA_TRANSFER_COARSE_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD
         || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
        && location != QUDA_CPU_FIELD_LOCATION) {
      return;
    }

    postTrace();
    ColorSpinorParam param(*B[0]);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.location = location;
    param.fieldOrder = location == QUDA_CUDA_FIELD_LOCATION ? colorspinor::getNative(null_precision, param.nSpin) :
                                                              QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    if (param.Precision() < QUDA_SINGLE_PRECISION) param.setPrecision(QUDA_SINGLE_PRECISION);

    if (location == QUDA_CUDA_FIELD_LOCATION) {
      if (fine_tmp_d && coarse_tmp_d) return;
      fine_tmp_d = new ColorSpinorField(param);
      coarse_tmp_d = fine_tmp_d->CreateCoarse(geo_bs, spin_bs, Nvec);
    } else {
      fine_tmp_h = new ColorSpinorField(param);
      coarse_tmp_h = fine_tmp_h->CreateCoarse(geo_bs, spin_bs, Nvec);
    }
    postTrace();
  }

  void Transfer::initializeLazy(QudaFieldLocation location) const
  {
    if (!enable_cpu && !enable_gpu) errorQuda("Neither CPU or GPU coarse fields initialized");

    // delayed allocating this temporary until we need it
    if (B[0]->Location() == QUDA_CUDA_FIELD_LOCATION) createTmp(QUDA_CUDA_FIELD_LOCATION);

    switch (location) {
    case QUDA_CUDA_FIELD_LOCATION:
      if (enable_gpu) return;
      createV(location);
      if (transfer_type == QUDA_TRANSFER_AGGREGATE) *V_d = *V_h;
      createTmp(location);
      fine_to_coarse_d = static_cast<int*>(pool_device_malloc(B[0]->Volume()*sizeof(int)));
      coarse_to_fine_d = static_cast<int*>(pool_device_malloc(B[0]->Volume()*sizeof(int)));
      qudaMemcpy(fine_to_coarse_d, fine_to_coarse_h, B[0]->Volume() * sizeof(int), qudaMemcpyHostToDevice);
      qudaMemcpy(coarse_to_fine_d, coarse_to_fine_h, B[0]->Volume() * sizeof(int), qudaMemcpyHostToDevice);
      break;
    case QUDA_CPU_FIELD_LOCATION:
      if (enable_cpu) return;
      createV(location);
      if (transfer_type == QUDA_TRANSFER_AGGREGATE) *V_h = *V_d;
      break;
    default:
      errorQuda("Unknown location %d", location);
    }
  }

  void Transfer::reset()
  {
    postTrace();

    if (transfer_type == QUDA_TRANSFER_COARSE_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD
        || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      return;
    }
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transfer: block orthogonalizing\n");

    if (B[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      if (!enable_gpu) errorQuda("enable_gpu = %d so cannot reset", enable_gpu);
      BlockOrthogonalize(*V_d, B, fine_to_coarse_d, coarse_to_fine_d, geo_bs, spin_bs, NblockOrtho, blockOrthoTwoPass);
      if (enable_cpu) {
        *V_h = *V_d;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transferred prolongator back to CPU\n");
      }
    } else {
      if (!enable_cpu) errorQuda("enable_cpu = %d so cannot reset", enable_cpu);
      BlockOrthogonalize(*V_h, B, fine_to_coarse_h, coarse_to_fine_h, geo_bs, spin_bs, NblockOrtho, blockOrthoTwoPass);
      if (enable_gpu) { // if the GPU fields has been initialized then we need to update
        *V_d = *V_h;
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transferred prolongator to GPU\n");
      }
    }
    postTrace();
  }

  Transfer::~Transfer() {
    if (spin_map)
    {
      for (int s = 0; s < nspin_fine; s++) { if (spin_map[s]) host_free(spin_map[s]); } 
      host_free(spin_map);
    }
    if (coarse_to_fine_d) pool_device_free(coarse_to_fine_d);
    if (fine_to_coarse_d) pool_device_free(fine_to_coarse_d);
    if (coarse_to_fine_h) pool_pinned_free(coarse_to_fine_h);
    if (fine_to_coarse_h) pool_pinned_free(fine_to_coarse_h);
    if (V_h) delete V_h;
    if (V_d) delete V_d;

    if (fine_tmp_h) delete fine_tmp_h;
    if (fine_tmp_d) delete fine_tmp_d;

    if (coarse_tmp_h) delete coarse_tmp_h;
    if (coarse_tmp_d) delete coarse_tmp_d;

    if (geo_bs) delete []geo_bs;
  }

  void Transfer::setSiteSubset(QudaSiteSubset site_subset_, QudaParity parity_)
  {
    if (site_subset_ == QUDA_PARITY_SITE_SUBSET && parity_ != QUDA_EVEN_PARITY && parity_ != QUDA_ODD_PARITY)
      errorQuda("Undefined parity %d", parity_);
    parity = parity_;

    if (site_subset == site_subset_) return;
    site_subset = site_subset_;
  }

  struct Int2 {
    int x, y;
    Int2() : x(0), y(0) { } 
    Int2(int x, int y) : x(x), y(y) { } 
    
    bool operator<(const Int2 &a) const {
      return (x < a.x) ? true : (x==a.x && y<a.y) ? true : false;
    }
  };

  // compute the fine-to-coarse site map
  void Transfer::createGeoMap(int *geo_bs) {

    int x[QUDA_MAX_DIM];

    ColorSpinorField &fine(*fine_tmp_h);
    ColorSpinorField &coarse(*coarse_tmp_h);

    // compute the coarse grid point for every site (assuming parity ordering currently)
    for (size_t i = 0; i < fine.Volume(); i++) {
      // compute the lattice-site index for this offset index
      fine.LatticeIndex(x, i);
      
      //printfQuda("fine idx %d = fine (%d,%d,%d,%d), ", i, x[0], x[1], x[2], x[3]);

      // compute the corresponding coarse-grid index given the block size
      for (int d=0; d<fine.Ndim(); d++) x[d] /= geo_bs[d];

      // compute the coarse-offset index and store in fine_to_coarse
      int k;
      coarse.OffsetIndex(k, x); // this index is parity ordered
      fine_to_coarse_h[i] = k;

      //printfQuda("coarse after (%d,%d,%d,%d), coarse idx %d\n", x[0], x[1], x[2], x[3], k);
    }

    // now create an inverse-like variant of this

    std::vector<Int2> geo_sort(B[0]->Volume());
    for (unsigned int i=0; i<geo_sort.size(); i++) geo_sort[i] = Int2(fine_to_coarse_h[i], i);
    std::sort(geo_sort.begin(), geo_sort.end());
    for (unsigned int i=0; i<geo_sort.size(); i++) coarse_to_fine_h[i] = geo_sort[i].y;

    if (enable_gpu) {
      qudaMemcpy(fine_to_coarse_d, fine_to_coarse_h, B[0]->Volume() * sizeof(int), qudaMemcpyHostToDevice);
      qudaMemcpy(coarse_to_fine_d, coarse_to_fine_h, B[0]->Volume() * sizeof(int), qudaMemcpyHostToDevice);
    }

  }

  // compute the fine spin and checkerboard to coarse spin map
  void Transfer::createSpinMap(int spin_bs) {
    if (spin_bs == 0) // staggered
    {
      spin_map[0][0] = 0; // fine even
      spin_map[0][1] = 1; // fine odd
    }
    else
    {
      for (int s=0; s<B[0]->Nspin(); s++) {
        spin_map[s][0] = s / spin_bs; // not staggered, doesn't care about parity. 
        spin_map[s][1] = s / spin_bs;
      }
    }
  }

  // apply the prolongator
  void Transfer::P(ColorSpinorField &out, const ColorSpinorField &in) const {
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    ColorSpinorField *input = const_cast<ColorSpinorField*>(&in);
    ColorSpinorField *output = &out;
    initializeLazy(use_gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION);
    const int *fine_to_coarse = use_gpu ? fine_to_coarse_d : fine_to_coarse_h;

    if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      StaggeredProlongate(*output, *input, fine_to_coarse, spin_map, parity);
      flops_ += 0; // it's only a permutation
    } else if (transfer_type == QUDA_TRANSFER_OPTIMIZED_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {

      if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Optimized KD op only supports full-parity spinors");

      if (output->VolumeCB() != input->VolumeCB()) errorQuda("Optimized KD transfer is only between equal volumes");

      // the optimized KD op acts on fine spinors
      if (out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
        *output = input->Even();
      } else {
        *output = *input;
      }
      flops_ += 0;

    } else if (transfer_type == QUDA_TRANSFER_AGGREGATE) {

      const ColorSpinorField *V = use_gpu ? V_d : V_h;

      if (use_gpu) {
        if (in.Location() == QUDA_CPU_FIELD_LOCATION) input = coarse_tmp_d;
        if (out.Location() == QUDA_CPU_FIELD_LOCATION || out.GammaBasis() != V->GammaBasis())
          output = (out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? fine_tmp_d : &fine_tmp_d->Even();
        if (!enable_gpu) errorQuda("not created with enable_gpu set, so cannot run on GPU");
      } else {
        if (out.Location() == QUDA_CUDA_FIELD_LOCATION)
          output = (out.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? fine_tmp_h : &fine_tmp_h->Even();
      }

      *input = in; // copy result to input field (aliasing handled automatically)

      if (V->SiteSubset() == QUDA_PARITY_SITE_SUBSET && out.SiteSubset() == QUDA_FULL_SITE_SUBSET)
        errorQuda("Cannot prolongate to a full field since only have single parity null-space components");

      if ((V->Nspin() != 1) && ((output->GammaBasis() != V->GammaBasis()) || (input->GammaBasis() != V->GammaBasis()))) {
        errorQuda("Cannot apply prolongator using fields in a different basis from the null space (%d,%d) != %d",
                  output->GammaBasis(), in.GammaBasis(), V->GammaBasis());
      }

      Prolongate(*output, *input, *V, fine_to_coarse, spin_map, parity);

      flops_ += 8 * in.Ncolor() * out.Ncolor() * out.VolumeCB() * out.SiteSubset();
    } else {
      errorQuda("Invalid transfer type in prolongate");
    }

    out = *output; // copy result to out field (aliasing handled automatically)

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  // apply the restrictor
  void Transfer::R(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    ColorSpinorField *input = &const_cast<ColorSpinorField&>(in);
    ColorSpinorField *output = &out;
    initializeLazy(use_gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION);
    const int *fine_to_coarse = use_gpu ? fine_to_coarse_d : fine_to_coarse_h;
    const int *coarse_to_fine = use_gpu ? coarse_to_fine_d : coarse_to_fine_h;

    if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      StaggeredRestrict(*output, *input, fine_to_coarse, spin_map, parity);
      flops_ += 0; // it's only a permutation
    } else if (transfer_type == QUDA_TRANSFER_OPTIMIZED_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {

      if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Optimized KD op only supports full-parity spinors");

      if (output->VolumeCB() != input->VolumeCB()) errorQuda("Optimized KD transfer is only between equal volumes");

      // the optimized KD op acts on fine spinors
      if (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
        output->Even() = *input;
        blas::zero(output->Odd());
      } else {
        *output = *input;
      }
      flops_ += 0;
    } else if (transfer_type == QUDA_TRANSFER_AGGREGATE) {

      const ColorSpinorField *V = use_gpu ? V_d : V_h;

      if (use_gpu) {
        if (out.Location() == QUDA_CPU_FIELD_LOCATION) output = coarse_tmp_d;
        if (in.Location() == QUDA_CPU_FIELD_LOCATION || in.GammaBasis() != V->GammaBasis())
          input = (in.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? fine_tmp_d : &fine_tmp_d->Even();
        if (!enable_gpu) errorQuda("not created with enable_gpu set, so cannot run on GPU");
      } else {
        if (in.Location() == QUDA_CUDA_FIELD_LOCATION)
          input = (in.SiteSubset() == QUDA_FULL_SITE_SUBSET) ? fine_tmp_h : &fine_tmp_h->Even();
      }

      *input = in;

      if (V->SiteSubset() == QUDA_PARITY_SITE_SUBSET && in.SiteSubset() == QUDA_FULL_SITE_SUBSET)
        errorQuda("Cannot restrict a full field since only have single parity null-space components");

      if (V->Nspin() != 1 && (output->GammaBasis() != V->GammaBasis() || input->GammaBasis() != V->GammaBasis()))
        errorQuda("Cannot apply restrictor using fields in a different basis from the null space (%d,%d) != %d",
                  out.GammaBasis(), input->GammaBasis(), V->GammaBasis());

      Restrict(*output, *input, *V, fine_to_coarse, coarse_to_fine, spin_map, parity);

      flops_ += 8 * out.Ncolor() * in.Ncolor() * in.VolumeCB() * in.SiteSubset();
    } else {
      errorQuda("Invalid transfer type in restrict");
    }

    out = *output; // copy result to out field (aliasing handled automatically)

    // only need to synchronize if we're transferring from GPU to CPU
    if (out.Location() == QUDA_CPU_FIELD_LOCATION && in.Location() == QUDA_CUDA_FIELD_LOCATION)
      qudaDeviceSynchronize();

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  double Transfer::flops() const {
    double rtn = flops_;
    flops_ = 0;
    return rtn;
  }

} // namespace quda
