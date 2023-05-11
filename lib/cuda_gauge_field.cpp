#include <cstring>
#include <typeinfo>
#include <gauge_field.h>
#include <timer.h>
#include <blas_quda.h>
#include <device.h>

namespace quda {

  cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) : GaugeField(param)
  {
    // exchange the boundaries if a non-trivial field
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD)
      if (create == QUDA_REFERENCE_FIELD_CREATE && (geometry == QUDA_VECTOR_GEOMETRY || geometry == QUDA_COARSE_GEOMETRY)) {
        exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
      }
  }

  void *create_gauge_buffer(size_t bytes, QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {
    if (order == QUDA_QDP_GAUGE_ORDER) {
      void **buffer = new void*[geometry];
      for (int d=0; d<geometry; d++) buffer[d] = pool_device_malloc(bytes/geometry);
      return ((void*)buffer);
    } else {
      return pool_device_malloc(bytes);
    }

  }

  void **create_ghost_buffer(size_t bytes[], QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {

    if (order > 4) {
      void **buffer = new void*[geometry];
      for (int d=0; d<geometry; d++) buffer[d] = pool_device_malloc(bytes[d]);
      return buffer;
    } else {
      return 0;
    }

  }

  void free_gauge_buffer(void *buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {
    if (order == QUDA_QDP_GAUGE_ORDER) {
      for (int d=0; d<geometry; d++) pool_device_free(((void**)buffer)[d]);
      delete []((void**)buffer);
    } else {
      pool_device_free(buffer);
    }
  }

  void free_ghost_buffer(void **buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {
    if (order > 4) {
      for (int d=0; d<geometry; d++) pool_device_free(buffer[d]);
      delete []buffer;
    }
  }

  void cudaGaugeField::copy(const GaugeField &src) {
    if (this == &src) return;

    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (fat_link_max == 0.0 && precision < QUDA_SINGLE_PRECISION) fat_link_max = src.abs_max();
    } else {
      fat_link_max = 1.0;
    }

    if (typeid(src) == typeid(cudaGaugeField)) {

      if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
        // copy field and ghost zone into this field
        copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION);

        if (geometry == QUDA_COARSE_GEOMETRY)
          copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, nullptr, nullptr, nullptr, 3);
      } else {
        copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, nullptr);
        if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
      }

    } else if (typeid(src) == typeid(cpuGaugeField)) {
      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do reorder on the CPU
	void *buffer = pool_pinned_malloc(bytes);

	if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	  // copy field and ghost zone into buffer
          copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, nullptr);

          if (geometry == QUDA_COARSE_GEOMETRY)
            copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, nullptr, 0, 0, 3);
        } else {
          copyExtendedGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, nullptr);
          if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
	}

        qudaMemcpy(gauge.data(), buffer, bytes, qudaMemcpyDefault);
        pool_pinned_free(buffer);
      } else { // else on the GPU

        if (src.Order() == QUDA_MILC_SITE_GAUGE_ORDER ||
            src.Order() == QUDA_BQCD_GAUGE_ORDER      ||
            src.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	  // special case where we use zero-copy memory to read/write directly from application's array
          void *src_d = get_mapped_device_pointer(src.data());

          if (src.GhostExchange() == QUDA_GHOST_EXCHANGE_NO) {
            copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, data(), src_d);
          } else {
            errorQuda("Ghost copy not supported here");
          }

        } else {
	  void *buffer = create_gauge_buffer(src.Bytes(), src.Order(), src.Geometry());
	  size_t ghost_bytes[8];
	  int srcNinternal = src.Reconstruct() != QUDA_RECONSTRUCT_NO ? src.Reconstruct() : 2*nColor*nColor;
	  for (int d=0; d<geometry; d++) ghost_bytes[d] = nFace * surface[d%4] * srcNinternal * src.Precision();
	  void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, src.Order(), geometry) : nullptr;

	  if (src.Order() == QUDA_QDP_GAUGE_ORDER) {
	    for (int d=0; d<geometry; d++) {
              qudaMemcpy(((void **)buffer)[d], src.data(d), src.Bytes() / geometry, qudaMemcpyDefault);
            }
          } else {
            qudaMemcpy(buffer, src.data(), src.Bytes(), qudaMemcpyDefault);
          }

          if (src.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD
              && src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
            for (int d = 0; d < geometry; d++)
              qudaMemcpy(ghost_buffer[d], src.Ghost()[d].data(), ghost_bytes[d], qudaMemcpyDefault);

          if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
            copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, buffer, nullptr, ghost_buffer);
            if (geometry == QUDA_COARSE_GEOMETRY)
              copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, buffer, nullptr, ghost_buffer, 3);
          } else {
            copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, buffer);
            if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
          }
          free_gauge_buffer(buffer, src.Order(), src.Geometry());
          if (nFace > 0) free_ghost_buffer(ghost_buffer, src.Order(), geometry);
        }
      } // reorder_location
    } else {
      errorQuda("Invalid gauge field type");
    }

    // if we have copied from a source without a pad then we need to exchange
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD && src.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD)
      exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);

    staggeredPhaseApplied = src.StaggeredPhaseApplied();
    staggeredPhaseType = src.StaggeredPhase();

    qudaDeviceSynchronize(); // include sync here for accurate host-device profiling
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) {
    copy(cpu);
    qudaDeviceSynchronize();
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, TimeProfile &profile) {
    profile.TPSTART(QUDA_PROFILE_H2D);
    loadCPUField(cpu);
    profile.TPSTOP(QUDA_PROFILE_H2D);
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const
  {
    cpu.checkField(*this);

    if (reorder_location() == QUDA_CUDA_FIELD_LOCATION) {

      if (cpu.Order() == QUDA_MILC_SITE_GAUGE_ORDER ||
          cpu.Order() == QUDA_BQCD_GAUGE_ORDER      ||
          cpu.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	// special case where we use zero-copy memory to read/write directly from application's array
        void *cpu_d = get_mapped_device_pointer(cpu.data());
        if (cpu.GhostExchange() == QUDA_GHOST_EXCHANGE_NO) {
          copyGenericGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, cpu_d, nullptr);
        } else {
          errorQuda("Ghost copy not supported here");
        }
      } else {
	void *buffer = create_gauge_buffer(cpu.Bytes(), cpu.Order(), cpu.Geometry());

	// Allocate space for ghost zone if required
	size_t ghost_bytes[8];
	int cpuNinternal = cpu.Reconstruct() != QUDA_RECONSTRUCT_NO ? cpu.Reconstruct() : 2*nColor*nColor;
	for (int d=0; d<geometry; d++) ghost_bytes[d] = nFace * surface[d%4] * cpuNinternal * cpu.Precision();
	void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, cpu.Order(), geometry) : nullptr;

	if (cpu.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
          copyGenericGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr, ghost_buffer, nullptr);
          if (geometry == QUDA_COARSE_GEOMETRY)
            copyGenericGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr, ghost_buffer, nullptr, 3);
        } else {
          copyExtendedGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr);
        }

        if (cpu.Order() == QUDA_QDP_GAUGE_ORDER) {
          for (int d = 0; d < geometry; d++)
            qudaMemcpy(cpu.data(d), ((void **)buffer)[d], cpu.Bytes() / geometry, qudaMemcpyDefault);
        } else {
          qudaMemcpy(cpu.data(), buffer, cpu.Bytes(), qudaMemcpyDefault);
        }

        if (cpu.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD
            && cpu.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
          for (int d = 0; d < geometry; d++)
            qudaMemcpy(cpu.Ghost()[d].data(), ghost_buffer[d], ghost_bytes[d], qudaMemcpyDefault);

        free_gauge_buffer(buffer, cpu.Order(), cpu.Geometry());
        if (nFace > 0) free_ghost_buffer(ghost_buffer, cpu.Order(), geometry);
      }
    } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do copy then host-side reorder

      void *buffer = pool_pinned_malloc(bytes);
      qudaMemcpy(buffer, gauge.data(), bytes, qudaMemcpyDefault);

      if (cpu.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
        copyGenericGauge(cpu, *this, QUDA_CPU_FIELD_LOCATION, nullptr, buffer);
      } else {
        copyExtendedGauge(cpu, *this, QUDA_CPU_FIELD_LOCATION, nullptr, buffer);
      }
      pool_pinned_free(buffer);

    } else {
      errorQuda("Invalid pack location %d", reorder_location());
    }

    cpu.staggeredPhaseApplied = staggeredPhaseApplied;
    cpu.staggeredPhaseType = staggeredPhaseType;

    qudaDeviceSynchronize();
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, TimeProfile &profile) const {
    profile.TPSTART(QUDA_PROFILE_D2H);
    saveCPUField(cpu);
    profile.TPSTOP(QUDA_PROFILE_D2H);
  }

} // namespace quda
