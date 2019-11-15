#ifndef __MADWF_ML_H__
#define __MADWF_ML_H__

#include <quda_internal.h>
#include <quda.h>
#include <complex_quda.h>

#include <vector>

namespace quda
{
  namespace madwf_ml
  {

    template <class real> class TrainingParameter
    {

      real *device_data;

      size_t size;

    public:
      TrainingParameter(std::vector<real> &host_vector) : size(host_vector.size())
      {
        size_t m_size = size * sizeof(real);
        device_data = (real *)device_malloc_(__func__, __FILE__, __LINE__, m_size);
        if (!device_data) { errorQuda("Unable to allocate a device buffer of %lu bytes.\n", m_size); }
        cudaMemcpy(device_data, host_vector.data(), m_size, cudaMemcpyHostToDevice);
      }

      TrainingParameter(size_t size_) : size(size_)
      {
        size_t m_size = size * sizeof(real);
        device_data = (real *)device_malloc_(__func__, __FILE__, __LINE__, m_size);
        if (!device_data) { errorQuda("Unable to allocate a device buffer of %lu bytes.\n", m_size); }
        cudaMemset(device_data, 0, m_size);
      }

      ~TrainingParameter()
      {
        if (device_data) { device_free_(__func__, __FILE__, __LINE__, device_data); }
      }

      void copy(const TrainingParameter& other)
      {
        if(size != other.size){
          errorQuda("Copying from a vector with different size.\n");
        }
        size_t m_size = size * sizeof(real);
        if(device_data != other.device_data){
          cudaMemcpy(device_data, other.device_data, m_size, cudaMemcpyDeviceToDevice);
        }
      }

      std::vector<real> to_host() const
      {
        std::vector<real> out(size);
        cudaMemcpy(out.data(), device_data, size * sizeof(real), cudaMemcpyDeviceToHost);
        return out;
      }

      void from_host(const std::vector<real> &host_vector)
      {
        if (host_vector.size() != get_size()) { errorQuda("Size mismatch.\n"); }
        cudaMemcpy(device_data, host_vector.data(), size * sizeof(real), cudaMemcpyHostToDevice);
      }

      real *data() { return device_data; }

      const real *data() const { return device_data; }

      size_t get_size() const { return size; }
    };

    void axpby(TrainingParameter<float> &out, complex<float> a, const TrainingParameter<float> &x, complex<float> b,
               const TrainingParameter<float> &y);

    constexpr int spin_dim = 4;
    constexpr int color_dim = 3;
    constexpr int sm_dim = spin_dim * spin_dim;

    constexpr int color_spin_dim = spin_dim * color_dim;
    constexpr int wm_dim = color_spin_dim * color_spin_dim;
    // TODO: make these generic
    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, const TrainingParameter<float> &tp,
                        bool dagger);

    void tensor_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, TrainingParameter<float> &tp);

  } // namespace madwf_ml
} // namespace quda

#endif
