#include <cstring>
#include <typeinfo>
#include <gauge_field.h>
#include <timer.h>
#include <blas_quda.h>
#include <device.h>

namespace quda {

  cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) : GaugeField(param) {}

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) { copy(cpu); }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, TimeProfile &profile)
  {
    profile.TPSTART(QUDA_PROFILE_H2D);
    copy(cpu);
    profile.TPSTOP(QUDA_PROFILE_H2D);
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const { cpu.copy(*this); }
#if 0
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
#endif
  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, TimeProfile &profile) const {
    profile.TPSTART(QUDA_PROFILE_D2H);
    saveCPUField(cpu);
    profile.TPSTOP(QUDA_PROFILE_D2H);
  }

} // namespace quda
