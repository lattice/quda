
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

  Transfer::Transfer(const std::vector<ColorSpinorField> &B, int Nvec,
                     int spin_bs, QudaPrecision null_precision) :
    B(B),
    Nvec(Nvec),
    null_precision(null_precision),
    spin_bs(spin_bs)
  {
    
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

  TransferAggregate::TransferAggregate(const std::vector<ColorSpinorField> &B, int Nvec, int n_block_ortho, bool block_ortho_two_pass,
                     int *geo_bs, int spin_bs, QudaPrecision null_precision, bool use_mma) :
    Transfer(B, Nvec, spin_bs, null_precision),
    NblockOrtho(n_block_ortho),
    blockOrthoTwoPass(block_ortho_two_pass),
    use_mma(use_mma)
  {
    postTrace();

    // initialize the block sizes
    initializeBlockSizes(geo_bs);

    // allocate the fine-to-coarse spin map
    createSpinMap(B[0].Nspin());

    // create ColorSpinorParam objects for the fine and coarse fields
    createColorSpinorParams();

    createV(); // allocate V field

    // create the fine-to-coarse and coarse-to-fine site maps
    createGeoMap();

    reset();
    postTrace();
  }

  void TransferAggregate::initializeBlockSizes(int *geo_bs) {
    int ndim = B[0].Ndim();

    // Only loop over four dimensions for now, we don't have
    // to worry about the fifth dimension until we hit chiral fermions.
    if (ndim > 4) errorQuda("Number of dimensions %d not supported", ndim);

    this->geo_bs = new int[ndim];
    int total_block_size = 1;

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
      this->geo_bs[d] = geo_bs[d];
      total_block_size *= geo_bs[d];
    }

    int aggregate_size = total_block_size * B[0].Ncolor();
    if (spin_bs == 0)
      aggregate_size /= 2; // effective spin_bs of 0.5 (fine spin / coarse spin)
    else
      aggregate_size *= spin_bs;
    if (Nvec > aggregate_size)
      errorQuda("Requested coarse space %d larger than aggregate size %d", Nvec, aggregate_size);

    std::string block_str = std::to_string(geo_bs[0]);
    for (int d = 1; d < ndim; d++) block_str += " x " + std::to_string(geo_bs[d]);
    logQuda(QUDA_VERBOSE, "Transfer: using block size %s\n", block_str.c_str());
  }

  void TransferAggregate::createV() const
  {
    postTrace();

    // create the storage for the final block orthogonal elements
    // uses the geometry from the null-space vectors
    auto param = fineColorSpinorParam(B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? null_precision : B[0].Precision(), B[0].Location());
    param.nColor *= Nvec;
    param.nVec = Nvec;

    // the prolongator/restrictor is always in the Degrand-Rossi gamma basis
    param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      //keep it the same for staggered:
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }

    V = ColorSpinorField(param);
    postTrace();
  }

  void TransferAggregate::reset()
  {
    postTrace();

    logQuda(QUDA_VERBOSE, "Transfer: block orthogonalizing\n");

    BlockOrthogonalize(V, B, fine_to_coarse_d, coarse_to_fine_d, geo_bs, spin_bs, NblockOrtho, blockOrthoTwoPass);
    postTrace();
  }

  void TransferAggregate::createColorSpinorParams() {
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

    coarse_param.nSpin = (fine_param.nSpin == 1) ? 2 : (fine_param.nSpin / spin_bs);
    coarse_param.nColor = Nvec;
    coarse_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // coarse fields are always Degrand-Rossi
    coarse_param.siteSubset = QUDA_FULL_SITE_SUBSET; // coarse grid is always full
  }

// apply the prolongator
void TransferAggregate::P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const {
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

  // verify that "out" is an appropriate fine field and "in" is an appropriate coarse field
  verifyFineCompatibility(out[0]);
  verifyCoarseCompatibility(in[0]);

  if (V.SiteSubset() == QUDA_PARITY_SITE_SUBSET && out.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    errorQuda("Cannot prolongate to a full field since only have single parity null-space components");

  Prolongate(out, in, V, fine_to_coarse_d, spin_map, use_mma, parity);

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
}

// apply the restrictor
void TransferAggregate::R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
{
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

  // verify that "out" is an appropriate coarse field and "in" is an appropriate fine field
  verifyCoarseCompatibility(out[0]);
  verifyFineCompatibility(in[0]);

  if (V.SiteSubset() == QUDA_PARITY_SITE_SUBSET && in.SiteSubset() == QUDA_FULL_SITE_SUBSET)
    errorQuda("Cannot restrict a full field since only have single parity null-space components");

  Restrict(out, in, V, fine_to_coarse_d, coarse_to_fine_d, spin_map, use_mma, parity);

  // only need to synchronize if we're transferring from GPU to CPU
  if (out[0].Location() == QUDA_CPU_FIELD_LOCATION && in[0].Location() == QUDA_CUDA_FIELD_LOCATION)
    qudaDeviceSynchronize();

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
}

  TransferCopy::TransferCopy(const std::vector<ColorSpinorField> &B, int Nvec,
                     int *geo_bs, int spin_bs, QudaPrecision null_precision, const QudaTransferType transfer_type) :
    Transfer(B, Nvec, spin_bs, null_precision),
    transfer_type(transfer_type)
  {
    postTrace();

    // initialize the block sizes
    initializeBlockSizes(geo_bs);

    // allocate the fine-to-coarse spin map
    createSpinMap(B[0].Nspin());

    // create ColorSpinorParam objects for the fine and coarse fields
    createColorSpinorParams();

    createV(); // allocate V field

    // create the fine-to-coarse and coarse-to-fine site maps
    createGeoMap();

    reset();
    postTrace();
  }


  void TransferCopy::initializeBlockSizes(int *geo_bs) {
    int ndim = B[0].Ndim();

    // Only loop over four dimensions for now, we don't have
    // to worry about the fifth dimension until we hit chiral fermions.
    if (ndim > 4) errorQuda("Number of dimensions %d not supported", ndim);

    this->geo_bs = new int[ndim];

    // the aggregation size is technically 1
    for (int d = 0; d < ndim; d++) {
      if (geo_bs[d] != 1)
        errorQuda("Invalid geometric block size %d for dimension %d for optimized KD transfer, must be 1", geo_bs[d], d);
      this->geo_bs[d] = geo_bs[d];
    }

    // The number of coarse dof is technically fineColor for optimized KD
    if (Nvec != B[0].Ncolor())
      errorQuda("Invalid Nvec %d for optimized-kd aggregation, must be fine color %d", Nvec, B[0].Ncolor());

    std::string block_str = std::to_string(geo_bs[0]);
    for (int d = 1; d < ndim; d++) block_str += " x " + std::to_string(geo_bs[d]);
    logQuda(QUDA_VERBOSE, "Transfer: using block size %s\n", block_str.c_str());
  }

  void TransferCopy::createV() const
  {
    postTrace();

    // create the storage for the final block orthogonal elements
    // uses the geometry from the null-space vectors
    auto param = fineColorSpinorParam(B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? null_precision : B[0].Precision(), B[0].Location());
    param.nColor *= Nvec;
    param.nVec = Nvec;

    // the prolongator/restrictor is always in the Degrand-Rossi gamma basis
    param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      //keep it the same for staggered:
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }

    // Need to create V_d and V_h as metadata containers, but we don't actually need to allocate the memory.
    param.create = QUDA_REFERENCE_FIELD_CREATE;

    V = ColorSpinorField(param);
    postTrace();
  }

  void TransferCopy::createColorSpinorParams() {
    // create the ColorSpinorParam objects for the fine and coarse fields
    // the precision is intentionally unset
    fine_param = ColorSpinorParam(B[0]);
    fine_param.create = QUDA_NULL_FIELD_CREATE;
    fine_param.location = B[0].Location();
    fine_param.fieldOrder = B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    fine_param.gammaBasis = (B[0].Nspin() == 4) ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    // for the copy transfer, the coarse param is the same as the fine param
    coarse_param = fine_param;

    // except maybe for this, but we'll figure it out later
    coarse_param.siteSubset = QUDA_FULL_SITE_SUBSET; // coarse grid is always full
  }

// apply the prolongator
void TransferCopy::P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const {
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

  // verify that "out" is an appropriate fine field and "in" is an appropriate coarse field
  verifyFineCompatibility(out[0]);
  verifyCoarseCompatibility(in[0]);

  if (in.SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Optimized KD op only supports full-parity spinors");
  if (out.VolumeCB() != in.VolumeCB()) errorQuda("Optimized KD transfer is only between equal volumes");

  // the optimized KD op acts on fine spinors
  if (out.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
    for (auto i = 0u; i < out.size(); i++) out[i] = in[i].Even();
  } else {
    for (auto i = 0u; i < out.size(); i++) out[i] = in[i];
  }

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
}

// apply the restrictor
void TransferCopy::R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
{
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

  // verify that "out" is an appropriate coarse field and "in" is an appropriate fine field
  verifyCoarseCompatibility(out[0]);
  verifyFineCompatibility(in[0]);

  if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET) errorQuda("Optimized KD op only supports full-parity spinors");
  if (out.VolumeCB() != in.VolumeCB()) errorQuda("Optimized KD transfer is only between equal volumes");

  // the optimized KD op acts on fine spinors
  if (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
    for (auto i = 0u; i < out.size(); i++) out[i].Even() = in[i];
    for (auto i = 0u; i < out.size(); i++) blas::zero(out[i].Odd());
  } else {
    for (auto i = 0u; i < out.size(); i++) out[i] = in[i];
  }

  // only need to synchronize if we're transferring from GPU to CPU
  if (out[0].Location() == QUDA_CPU_FIELD_LOCATION && in[0].Location() == QUDA_CUDA_FIELD_LOCATION)
    qudaDeviceSynchronize();

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
}


  TransferCoarseKD::TransferCoarseKD(const std::vector<ColorSpinorField> &B, int Nvec,
                     int *geo_bs, int spin_bs, QudaPrecision null_precision) :
    Transfer(B, Nvec, spin_bs, null_precision)
  {
    postTrace();

    // initialize the block sizes
    initializeBlockSizes(geo_bs);

    // allocate the fine-to-coarse spin map
    createSpinMap(B[0].Nspin());

    // create ColorSpinorParam objects for the fine and coarse fields
    createColorSpinorParams();

    createV(); // allocate V field

    // create the fine-to-coarse and coarse-to-fine site maps
    createGeoMap();

    reset();
    postTrace();
  }

  void TransferCoarseKD::initializeBlockSizes(int *geo_bs) {
    int ndim = B[0].Ndim();

    // Only loop over four dimensions for now, we don't have
    // to worry about the fifth dimension until we hit chiral fermions.
    if (ndim > 4) errorQuda("Number of dimensions %d not supported", ndim);

    this->geo_bs = new int[ndim];

    // the aggregation size needs to be 2^4
    for (int d = 0; d < ndim; d++) {
      if (geo_bs[d] != 2)
        errorQuda("Invalid geometric block size %d for dimension %d for coarse KD transfer, must be 2", geo_bs[d], d);
      this->geo_bs[d] = geo_bs[d];
    }

    // The number of coarse dof is 8 * fineColor / 2 for coarse KD
    if (Nvec != 8 * B[0].Ncolor())
      errorQuda("Invalid Nvec %d for coarse KD aggregation, must be %d", Nvec, 8 * B[0].Ncolor());

    std::string block_str = std::to_string(geo_bs[0]);
    for (int d = 1; d < ndim; d++) block_str += " x " + std::to_string(geo_bs[d]);
    logQuda(QUDA_VERBOSE, "Transfer: using block size %s\n", block_str.c_str());
  }

  void TransferCoarseKD::createV() const
  {
    postTrace();

    // create the storage for the final block orthogonal elements
    // uses the geometry from the null-space vectors
    auto param = fineColorSpinorParam(B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? null_precision : B[0].Precision(), B[0].Location());
    param.nColor *= Nvec;
    param.nVec = Nvec;

    // the prolongator/restrictor is always in the Degrand-Rossi gamma basis
    param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    // the V field is defined on all sites regardless of B field (maybe the B fields are always full?)
    if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      //keep it the same for staggered:
      param.siteSubset = QUDA_FULL_SITE_SUBSET;
      param.x[0] *= 2;
    }

    // Need to create V_d and V_h as metadata containers, but we don't actually need to allocate the memory.
    param.create = QUDA_REFERENCE_FIELD_CREATE;

    V = ColorSpinorField(param);
    postTrace();
  }

  void TransferCoarseKD::createColorSpinorParams() {
    // create the ColorSpinorParam objects for the fine and coarse fields
    // the precision is intentionally unset
    fine_param = ColorSpinorParam(B[0]);
    fine_param.create = QUDA_NULL_FIELD_CREATE;
    fine_param.location = B[0].Location();
    fine_param.fieldOrder = B[0].Location() == QUDA_CUDA_FIELD_LOCATION ? QUDA_NATIVE_FIELD_ORDER : QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    fine_param.gammaBasis = (B[0].Nspin() == 4) ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    // copy the fine param and modify in place
    coarse_param = fine_param;
    for (int d = 0; d < fine_param.nDim; d++) coarse_param.x[d] = fine_param.x[d] / 2;

    coarse_param.nSpin = 2;
    coarse_param.nColor *= 8; // 16 * fineColor / 2 for coarse KD
    coarse_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // coarse staggered fields are always Degrand-Rossi
    coarse_param.siteSubset = QUDA_FULL_SITE_SUBSET; // coarse grid is always full
  }

// apply the prolongator
void TransferCoarseKD::P(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const {
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

  // verify that "out" is an appropriate fine field and "in" is an appropriate coarse field
  verifyFineCompatibility(out[0]);
  verifyCoarseCompatibility(in[0]);

  StaggeredProlongate(out, in, fine_to_coarse_d, spin_map, parity);

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
}

// apply the restrictor
void TransferCoarseKD::R(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
{
  getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  if (out.size() != in.size()) errorQuda("Mismatched set sizes %lu != %lu", out.size(), in.size());

  // verify that "out" is an appropriate coarse field and "in" is an appropriate fine field
  verifyCoarseCompatibility(out[0]);
  verifyFineCompatibility(in[0]);

  StaggeredRestrict(out, in, fine_to_coarse_d, spin_map, parity);

  // only need to synchronize if we're transferring from GPU to CPU
  if (out[0].Location() == QUDA_CPU_FIELD_LOCATION && in[0].Location() == QUDA_CUDA_FIELD_LOCATION)
    qudaDeviceSynchronize();

  getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
}



} // namespace quda
