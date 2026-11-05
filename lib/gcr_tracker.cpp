#include <gcr_tracker.h>
#include <util_quda.h>

namespace quda
{

  GCRTracker *activeGCRTracker = nullptr;

  GCRTracker::GCRTracker(int maxVecs, QudaPrecision targetPrecision) :
    maxVecs_(maxVecs), targetPrecision_(targetPrecision), totalIterations_(0), active_(maxVecs > 0)
  {
  }

  void GCRTracker::reset()
  {
    residuals_.clear();
    totalIterations_ = 0;
  }

  void GCRTracker::recordIteration(const ColorSpinorField &r)
  {
    if (!active_) return;

    // Hierarchy filter. The hook in inv_gcr_quda.cpp fires for every
    // GCR instance, including the coarse-grid GCR that QUDA's MG
    // preconditioner runs at level 1 (and lower). Those coarse
    // residuals have Ns=2, Nc=nVec spinor metadata — incompatible with
    // the EigenTracker pool, which is seeded from fine-grid MG null
    // vectors at Ns=4, Nc=3 (Wilson). Without the filter the absorb
    // path faults on the multiCdot length check. We only want the
    // outer-fine-grid residuals here. (Tracking the coarse spectrum is
    // a separate problem with its own deflation manager — see the
    // CoarseDeflationManager used by NestedFGI.)
    if (r.Ncolor() != 3 || r.Nspin() != 4) return;

    totalIterations_++;

    // Cap reached: silently drop. We deliberately keep the FIRST
    // maxVecs_ residuals rather than the LAST. A FIFO-style "keep last"
    // would require std::vector::erase / pop_front, which QUDA's
    // ColorSpinorField forbids (operator= rejects assigning to an
    // already-created field as a guard against silent GPU-allocation
    // overwrite, see color_spinor_field.cpp:84). The early residuals
    // span the same Krylov subspace as the later ones; the pool's
    // Rayleigh-Ritz step downstream picks out the low-mode content
    // either way.
    if (static_cast<int>(residuals_.size()) >= maxVecs_) return;

    double rnorm = sqrt(blas::norm2(r));
    if (rnorm <= 1e-30) return;

    // Precision promotion to match the EigenTracker pool. GCR reads its
    // r_sloppy / p[k+1] argument at precision_sloppy (typically single).
    // The pool's reference vectors and absorption kernels live at the
    // target precision passed in by the caller (typically
    // inv_param.cuda_prec = double). Without the promotion the pool
    // absorb faults inside multiCdot — no instantiation exists for the
    // mixed double/single combination.
    //
    // Site-subset is *deliberately preserved*: the pool is seeded from
    // half-site (single-parity) MG null vectors (hmc.cpp's
    // seedEigenTrackingFromMG extracts B[i][QUDA_EVEN_PARITY]), so the
    // pool's reference vectors are half-site fine. Inside a PC solve
    // (solve_type=DIRECT_PC_SOLVE etc.) GCR's r_sloppy is also
    // half-site, which matches by construction. Embedding half-site
    // into full-site here would invert the match and fault in the same
    // length check.
    if (targetPrecision_ != QUDA_INVALID_PRECISION && r.Precision() != targetPrecision_) {
      ColorSpinorParam csParam(r);
      csParam.setPrecision(targetPrecision_);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      ColorSpinorField q(csParam);
      blas::copy(q, r);
      blas::ax(1.0 / rnorm, q);
      residuals_.push_back(std::move(q));
    } else {
      ColorSpinorField q(r);
      blas::ax(1.0 / rnorm, q);
      residuals_.push_back(std::move(q));
    }
  }

  std::vector<ColorSpinorField> GCRTracker::takeResiduals()
  {
    std::vector<ColorSpinorField> out = std::move(residuals_);
    residuals_.clear();
    return out;
  }

} // namespace quda
