
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
  Transfer::Transfer(const std::vector<ColorSpinorField> &B, int Nvec, int n_block_ortho, bool block_ortho_two_pass,
                     int *geo_bs, int spin_bs, QudaPrecision null_precision, const QudaTransferType transfer_type) :
    B(B),
    Nvec(Nvec),
    NblockOrtho(n_block_ortho),
    blockOrthoTwoPass(block_ortho_two_pass),
    null_precision(null_precision),
    spin_bs(spin_bs),
    spin_map(nullptr),
    site_subset(QUDA_FULL_SITE_SUBSET),
    parity(QUDA_INVALID_PARITY),
    transfer_type(transfer_type)
  {
    postTrace();

    // allocate the fine-to-coarse spin map
    createSpinMap(B[0].Nspin());

    int ndim = B[0].Ndim();

    // Only loop over four dimensions for now, we don't have
    // to worry about the fifth dimension until we hit chiral fermions.
    for (int d = 0; d < 4; d++) {
      while (geo_bs[d] > 0) {
        if (d == 0 && B[0].X(0) == geo_bs[0])
          warningQuda("X-dimension length %d cannot block length %d", B[0].X(0), geo_bs[0]);
        else if ((B[0].X(d) / geo_bs[d] + 1) % 2 == 0)
          warningQuda("Indexing does not (yet) support odd coarse dimensions: X(%d) = %d", d, B[0].X(d) / geo_bs[d]);
        else if ((B[0].X(d) / geo_bs[d]) * geo_bs[d] != B[0].X(d))
          warningQuda("cannot block dim[%d]=%d with block size = %d", d, B[0].X(d), geo_bs[d]);
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
      if (Nvec != B[0].Ncolor())
        errorQuda("Invalid Nvec %d for optimized-kd aggregation, must be fine color %d", Nvec, B[0].Ncolor());

    } else {
      int aggregate_size = total_block_size * B[0].Ncolor();
      if (spin_bs == 0)
        aggregate_size /= 2; // effective spin_bs of 0.5 (fine spin / coarse spin)
      else
        aggregate_size *= spin_bs;
      if (Nvec > aggregate_size)
        errorQuda("Requested coarse space %d larger than aggregate size %d", Nvec, aggregate_size);
    }

    std::string block_str = std::to_string(geo_bs[0]);
    for (int d = 1; d < ndim; d++) block_str += " x " + std::to_string(geo_bs[d]);
    logQuda(QUDA_VERBOSE, "Transfer: using block size %s\n", block_str.c_str());

    if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      for (int d = 0; d < 4; d++) {
        if (geo_bs[d] != 2) errorQuda("Invalid staggered KD block size %d for dimension %d, must be 2", geo_bs[d], d);
      }
      if (Nvec != 24) errorQuda("Invalid number of coarse vectors %d for staggered KD multigrid, must be 24", Nvec);
    }

    createV(); // allocate V field

    // create the fine-to-coarse and coarse-to-fine site maps
    createGeoMap();

    // allocate the fine-to-coarse spin map

    createSpinMap(spin_bs);

    // create ColorSpinorParam objects for the fine and coarse fields
    createColorSpinorParams();

    reset();
    postTrace();
  }

  void Transfer::createV() const
  {
    postTrace();

    // create the storage for the final block orthogonal elements
    ColorSpinorParam param(B[0]); // takes the geometry from the null-space vectors

    // the ordering of the V vector is defined by these parameters and
    // the Packed functions in ColorSpinorFieldOrder

    param.nSpin = B[0].Nspin();          // spin has direct mapping
    param.nColor = B[0].Ncolor() * Nvec; // nColor = number of colors * number of vectors
    param.nVec = Nvec;
    param.create = QUDA_NULL_FIELD_CREATE;
    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      //keep it the same for staggered:
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }
    param.fieldOrder
      = B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.setPrecision(B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? null_precision : B[0].Precision());

    if (transfer_type == QUDA_TRANSFER_COARSE_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD
        || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      // Need to create V_d and V_h as metadata containers, but we don't
      // actually need to allocate the memory.
      param.create = QUDA_REFERENCE_FIELD_CREATE;
    }

    V = ColorSpinorField(param);
    postTrace();
  }

  void Transfer::reset()
  {
    postTrace();

    if (transfer_type == QUDA_TRANSFER_COARSE_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD
        || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      return;
    }
    logQuda(QUDA_VERBOSE, "Transfer: block orthogonalizing\n");

    BlockOrthogonalize(V, B, fine_to_coarse_d, coarse_to_fine_d, geo_bs, spin_bs, NblockOrtho, blockOrthoTwoPass);
    postTrace();
  }

  Transfer::~Transfer() {
    if (spin_map)
    {
      for (int s = 0; s < fine_param.nSpin; s++) { if (spin_map[s]) host_free(spin_map[s]); }
      host_free(spin_map);
    }
    if (coarse_to_fine_d) pool_device_free(coarse_to_fine_d);
    if (fine_to_coarse_d) pool_device_free(fine_to_coarse_d);
    if (coarse_to_fine_h) pool_pinned_free(coarse_to_fine_h);
    if (fine_to_coarse_h) pool_pinned_free(fine_to_coarse_h);

    if (geo_bs) delete []geo_bs;
  }

  void Transfer::setSiteSubset(QudaSiteSubset site_subset_, QudaParity parity_)
  {
    if (site_subset_ == QUDA_PARITY_SITE_SUBSET && parity_ != QUDA_EVEN_PARITY && parity_ != QUDA_ODD_PARITY)
      errorQuda("Undefined parity %d", parity_);
    parity = parity_;

    if (site_subset == site_subset_) return;
    site_subset = site_subset_;
    fine_param.siteSubset = site_subset;
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
  void Transfer::createGeoMap()
  {
    int x[QUDA_MAX_DIM];

    // allocate and compute the fine-to-coarse and coarse-to-fine site maps
    fine_to_coarse_h = static_cast<int *>(pool_pinned_malloc(B[0].Volume() * sizeof(int)));
    coarse_to_fine_h = static_cast<int *>(pool_pinned_malloc(B[0].Volume() * sizeof(int)));

    fine_to_coarse_d = static_cast<int *>(pool_device_malloc(B[0].Volume() * sizeof(int)));
    coarse_to_fine_d = static_cast<int *>(pool_device_malloc(B[0].Volume() * sizeof(int)));

    ColorSpinorParam param(B[0]);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    if (param.Precision() < QUDA_SINGLE_PRECISION) param.setPrecision(QUDA_SINGLE_PRECISION);

    ColorSpinorField fine = ColorSpinorParam(param);
    ColorSpinorField coarse = fine.create_coarse(geo_bs, spin_bs, Nvec);

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

    std::vector<Int2> geo_sort(B[0].Volume());
    for (unsigned int i=0; i<geo_sort.size(); i++) geo_sort[i] = Int2(fine_to_coarse_h[i], i);
    std::sort(geo_sort.begin(), geo_sort.end());
    for (unsigned int i=0; i<geo_sort.size(); i++) coarse_to_fine_h[i] = geo_sort[i].y;

    qudaMemcpy(fine_to_coarse_d, fine_to_coarse_h, B[0].Volume() * sizeof(int), qudaMemcpyHostToDevice);
    qudaMemcpy(coarse_to_fine_d, coarse_to_fine_h, B[0].Volume() * sizeof(int), qudaMemcpyHostToDevice);
  }

  // compute the fine spin and checkerboard to coarse spin map
  void Transfer::createSpinMap(int n_fine_spin) {
    if (!spin_map) {
      spin_map = static_cast<int**>(safe_malloc(n_fine_spin*sizeof(int*)));
      for (int s = 0; s < n_fine_spin; s++) spin_map[s] = static_cast<int *>(safe_malloc(2 * sizeof(int)));

      if (spin_bs == 0) // staggered
      {
        spin_map[0][0] = 0; // fine even
        spin_map[0][1] = 1; // fine odd
      }
      else
      {
        for (int s = 0; s < n_fine_spin; s++) {
          spin_map[s][0] = s / spin_bs; // not staggered, doesn't care about parity. 
          spin_map[s][1] = s / spin_bs;
        }
      }
    }
  }

  void Transfer::createColorSpinorParams() {
    // create the ColorSpinorParam objects for the fine and coarse fields
    // the precision is intentionally unset
    fine_param = ColorSpinorParam(B[0]);
    fine_param.create = QUDA_NULL_FIELD_CREATE;
    fine_param.location = B[0].Location();
    fine_param.fieldOrder = B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    fine_param.gammaBasis = (B[0].Nspin() == 4) ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    // copy the fine param and modify in place
    coarse_param = fine_param;
    for (int d = 0; d < fine_param.nDim; d++) coarse_param.x[d] = fine_param.x[d] / geo_bs[d];

    // Detect if the "coarse op" is the Kahler-Dirac op or something else that still acts on a fine staggered ColorSpinorField
    if (transfer_type == QUDA_TRANSFER_OPTIMIZED_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      coarse_param.nSpin = fine_param.nSpin;
      coarse_param.nColor = fine_param.nColor;
      coarse_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // fine staggered fields are always Degrand-Rossi
    } else if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      coarse_param.nSpin = 2;
      coarse_param.nColor = 24;
      coarse_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // coarse staggered fields are always Degrand-Rossi
    } else {
      coarse_param.nSpin = (fine_param.nSpin == 1) ? 2 : (fine_param.nSpin / spin_bs);
      coarse_param.nColor = Nvec;
      coarse_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // coarse fields are always Degrand-Rossi
    }

    coarse_param.siteSubset = QUDA_FULL_SITE_SUBSET; // coarse grid is always full
  }

  ColorSpinorParam Transfer::fineColorSpinorParam(QudaPrecision precision, QudaFieldLocation new_location, QudaMemoryType new_mem_type) const {
    auto fine_param_copy = fine_param;

    // if new location is not set, use this->location
    fine_param_copy.location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? fine_param.location : new_location;

    fine_param_copy.fieldOrder
      = (fine_param_copy.location == QUDA_CUDA_FIELD_LOCATION) ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    fine_param_copy.setPrecision(precision);

    // set where we allocate the field
    fine_param_copy.mem_type = (new_mem_type == QUDA_MEMORY_INVALID) ? fine_param.mem_type : new_mem_type;

    return fine_param_copy;
  }

  ColorSpinorParam Transfer::coarseColorSpinorParam(QudaPrecision precision, QudaFieldLocation new_location, QudaMemoryType new_mem_type) const {
    auto coarse_param_copy = coarse_param;

    // if new location is not set, use this->location
    coarse_param_copy.location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? coarse_param.location : new_location;

    coarse_param_copy.fieldOrder
      = (coarse_param_copy.location == QUDA_CUDA_FIELD_LOCATION) ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    coarse_param_copy.setPrecision(precision);

    // set where we allocate the field
    coarse_param_copy.mem_type = (new_mem_type == QUDA_MEMORY_INVALID) ? coarse_param.mem_type : new_mem_type;

    return coarse_param_copy;
  }


  void Transfer::verifyFineCompatibility(const ColorSpinorField &fine) const {
    if (fineSiteSubset() == QUDA_PARITY_SITE_SUBSET && fine.X(0) * 2 != fine_param.x[0])
      errorQuda("Mismatched fine dimension %d sizes %d != %d", 0, 2 * fine.X(0), fine_param.x[0]);
    else if (fineSiteSubset() == QUDA_FULL_SITE_SUBSET && fine.X(0) != fine_param.x[0])
      errorQuda("Mismatched fine dimension %d sizes %d != %d", 0, fine.X(0), fine_param.x[0]);
    for (int d = 1; d < 4; d++)
      if (fine.X(d) != fine_param.x[d]) errorQuda("Mismatched fine dimension %d sizes %d != %d", d, fine.X(d), fine_param.x[d]);
    if (fine.SiteSubset() != fineSiteSubset()) errorQuda("Mismatched fine site subset %d != %d", fine.SiteSubset(), fineSiteSubset());
    if (fine.Nspin() != fineNspin()) errorQuda("Mismatched fine spin sizes %d != %d", fine.Nspin(), fineNspin());
    if (fine.Ncolor() != fineNcolor()) errorQuda("Mismatched fine color sizes %d != %d", fine.Ncolor(), fineNcolor());

    // nSpin = 4 can be either UKQCD or Degrand-Rossi
    if (fine.Nspin() != 4 && fine.GammaBasis() != fineGammaBasis()) {
      errorQuda("Mismatched fine gamma basis %d != %d", fine.GammaBasis(), fineGammaBasis());
    } else if (fine.Nspin() == 4 && fine.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS && fine.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS) {
      errorQuda("Invalid fine gamma basis %d", fine.GammaBasis());
    }
  }

  void Transfer::verifyCoarseCompatibility(const ColorSpinorField &coarse) const {
    if (coarseSiteSubset() == QUDA_PARITY_SITE_SUBSET && coarse.X(0) * 2 != coarse_param.x[0])
      errorQuda("Mismatched coarse dimension %d sizes %d != %d", 0, 2 * coarse.X(0), coarse_param.x[0]);
    else if (coarseSiteSubset() == QUDA_FULL_SITE_SUBSET && coarse.X(0) != coarse_param.x[0])
      errorQuda("Mismatched coarse dimension %d sizes %d != %d", 0, coarse.X(0), coarse_param.x[0]);
    for (int d = 1; d < 4; d++)
      if (coarse.X(d) != coarse_param.x[d]) errorQuda("Mismatched coarse dimension %d sizes %d != %d", d, coarse.X(d), coarse_param.x[d]);
    if (coarse.SiteSubset() != coarseSiteSubset()) errorQuda("Mismatched coarse site subset %d != %d", coarse.SiteSubset(), coarseSiteSubset());
    if (coarse.Nspin() != coarseNspin()) errorQuda("Mismatched coarse spin sizes %d != %d", coarse.Nspin(), coarseNspin());
    if (coarse.Ncolor() != coarseNcolor()) errorQuda("Mismatched coarse color sizes %d != %d", coarse.Ncolor(), coarseNcolor());
    if (coarse.GammaBasis() != coarseGammaBasis()) errorQuda("Mismatched coarse gamma basis %d != %d", coarse.GammaBasis(), coarseGammaBasis());

    // nSpin = 4 can be either UKQCD or Degrand-Rossi... for "copy" transfer ops only, but we'll get there...
    if (coarse.Nspin() != 4 && coarse.GammaBasis() != coarseGammaBasis()) {
      errorQuda("Mismatched coarse gamma basis %d != %d", coarse.GammaBasis(), coarseGammaBasis());
    } else if (coarse.Nspin() == 4 && coarse.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS && coarse.GammaBasis() != QUDA_UKQCD_GAMMA_BASIS) {
      errorQuda("Invalid coarse gamma basis %d", coarse.GammaBasis());
    }
  }

  // apply the prolongator
  void Transfer::P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

    // verify that "out" is an appropriate fine field and "in" is an appropriate coarse field
    verifyFineCompatibility(out[0]);
    verifyCoarseCompatibility(in[0]);

    if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      StaggeredProlongate(out, in, fine_to_coarse_d, spin_map, parity);
    } else if (transfer_type == QUDA_TRANSFER_OPTIMIZED_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {

      if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Optimized KD op only supports full-parity spinors");
      if (out.VolumeCB() != in.VolumeCB()) errorQuda("Optimized KD transfer is only between equal volumes");

      // the optimized KD op acts on fine spinors
      if (out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
        for (auto i = 0u; i < out.size(); i++) out[i] = in[i].Even();
      } else {
        for (auto i = 0u; i < out.size(); i++) out[i] = in[i];
      }

    } else if (transfer_type == QUDA_TRANSFER_AGGREGATE) {

      if (V.SiteSubset() == QUDA_PARITY_SITE_SUBSET && out.SiteSubset() == QUDA_FULL_SITE_SUBSET)
        errorQuda("Cannot prolongate to a full field since only have single parity null-space components");

      Prolongate(out, in, V, fine_to_coarse_d, spin_map, _use_mma, parity);
    } else {
      errorQuda("Invalid transfer type in prolongate");
    }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  // apply the restrictor
  void Transfer::R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

    // verify that "out" is an appropriate coarse field and "in" is an appropriate fine field
    verifyCoarseCompatibility(out[0]);
    verifyFineCompatibility(in[0]);

    if (transfer_type == QUDA_TRANSFER_COARSE_KD) {
      StaggeredRestrict(out, in, fine_to_coarse_d, spin_map, parity);
    } else if (transfer_type == QUDA_TRANSFER_OPTIMIZED_KD || transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {

      if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Optimized KD op only supports full-parity spinors");
      if (out.VolumeCB() != in.VolumeCB()) errorQuda("Optimized KD transfer is only between equal volumes");

      // the optimized KD op acts on fine spinors
      if (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
        for (auto i = 0u; i < out.size(); i++) out[i].Even() = in[i];
        for (auto i = 0u; i < out.size(); i++) blas::zero(out[i].Odd());
      } else {
        for (auto i = 0u; i < out.size(); i++) out[i] = in[i];
      }

    } else if (transfer_type == QUDA_TRANSFER_AGGREGATE) {

      if (V.SiteSubset() == QUDA_PARITY_SITE_SUBSET && in.SiteSubset() == QUDA_FULL_SITE_SUBSET)
        errorQuda("Cannot restrict a full field since only have single parity null-space components");

      Restrict(out, in, V, fine_to_coarse_d, coarse_to_fine_d, spin_map, _use_mma, parity);

    } else {
      errorQuda("Invalid transfer type in restrict");
    }

    // only need to synchronize if we're transferring from GPU to CPU
    if (out[0].Location() == QUDA_CPU_FIELD_LOCATION && in[0].Location() == QUDA_CUDA_FIELD_LOCATION)
      qudaDeviceSynchronize();

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
