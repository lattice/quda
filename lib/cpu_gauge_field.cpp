#include <quda_internal.h>
#include <timer.h>
#include <gauge_field.h>
#include <assert.h>
#include <string.h>
#include <typeinfo>

namespace quda {

  cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) :
    GaugeField(param)
  {
    // exchange the boundaries if a non-trivial field
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD)
      if (create == QUDA_REFERENCE_FIELD_CREATE && (geometry == QUDA_VECTOR_GEOMETRY || geometry == QUDA_COARSE_GEOMETRY)) {
        exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
      }

    // compute the fat link max now in case it is needed later (i.e., for half precision)
    if (param.compute_fat_link_max) fat_link_max = this->abs_max();
  }

  // defined in cudaGaugeField
  void *create_gauge_buffer(size_t bytes, QudaGaugeFieldOrder order, QudaFieldGeometry geometry);
  void **create_ghost_buffer(size_t bytes[], QudaGaugeFieldOrder order, QudaFieldGeometry geometry);
  void free_gauge_buffer(void *buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry);
  void free_ghost_buffer(void **buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry);

  void cpuGaugeField::copy(const GaugeField &src) {
    if (this == &src) return;

    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (fat_link_max == 0.0 && precision < QUDA_SINGLE_PRECISION) fat_link_max = src.abs_max();
    } else {
      fat_link_max = 1.0;
    }

    if (typeid(src) == typeid(cudaGaugeField)) {

      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) {

	if (!src.isNative()) errorQuda("Only native order is supported");
	void *buffer = pool_pinned_malloc(src.Bytes());
        qudaMemcpy(buffer, src.data(), src.Bytes(), qudaMemcpyDeviceToHost);

        copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, nullptr, buffer);
        pool_pinned_free(buffer);

      } else { // else on the GPU

	void *buffer = create_gauge_buffer(bytes, order, geometry);
	size_t ghost_bytes[8];
	int dstNinternal = reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : 2*nColor*nColor;
	for (int d=0; d<geometry; d++) ghost_bytes[d] = nFace * surface[d%4] * dstNinternal * precision;
	void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, order, geometry) : nullptr;

	if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED) {
          copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr, ghost_buffer, nullptr);
          if (geometry == QUDA_COARSE_GEOMETRY)
            copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr, ghost_buffer, nullptr,
                             3); // forwards links if bi-directional
        } else {
	  copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, buffer, 0);
	}

	if (order == QUDA_QDP_GAUGE_ORDER) {
	  for (int d=0; d<geometry; d++) {
            qudaMemcpy(gauge_array[d].data(), ((void **)buffer)[d], bytes / geometry, qudaMemcpyDeviceToHost);
          }
	} else {
          qudaMemcpy(gauge.data(), buffer, bytes, qudaMemcpyHostToDevice);
        }

	if (order > 4 && ghostExchange == QUDA_GHOST_EXCHANGE_PAD && src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
	  for (int d=0; d<geometry; d++)
            qudaMemcpy(Ghost()[d].data(), ghost_buffer[d], ghost_bytes[d], qudaMemcpyDeviceToHost);

        free_gauge_buffer(buffer, order, geometry);
	if (nFace > 0) free_ghost_buffer(ghost_buffer, order, geometry);
      }

    } else if (typeid(src) == typeid(cpuGaugeField)) {
      // copy field and ghost zone directly
      copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION);
    } else {
      errorQuda("Invalid gauge field type");
    }

    // if we have copied from a source without a pad then we need to exchange
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD &&
	src.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD) {
      exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
    }
  }

} // namespace quda
