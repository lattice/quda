#pragma once

#include <vector>

#include <quda_api.h>
#include <malloc_quda.h>
#include <util_quda.h>
#include <complex_quda.h>

namespace quda
{
  template <class real> class device_vector
  {
    real *_device_data = nullptr;

    size_t _size;

  public:
    device_vector(std::vector<real> &host_vector) : _size(host_vector.size())
    {
      size_t bytes = _size * sizeof(real);
      if (bytes > 0) {
        _device_data = reinterpret_cast<real *>(device_malloc(bytes));
        if (!_device_data) { errorQuda("Unable to allocate a device buffer of %lu bytes.\n", bytes); }
        qudaMemcpy(_device_data, host_vector.data(), bytes, qudaMemcpyHostToDevice);
      }
    }

    device_vector(size_t size_) : _size(size_) { resize(_size); }

    device_vector() : _size(0) { }

    void resize(size_t size_)
    {
      if (_device_data) { device_free(_device_data); }
      _size = size_;
      size_t bytes = _size * sizeof(real);
      if (bytes > 0) {
        _device_data = (real *)device_malloc(bytes);
        if (!_device_data) { errorQuda("Unable to allocate a device buffer of %lu bytes.\n", bytes); }
        qudaMemset(_device_data, 0, bytes);
      }
    }

    ~device_vector()
    {
      if (_device_data) { device_free(_device_data); }
    }

    void copy(const device_vector &other)
    {
      if (_size != other.size()) { errorQuda("Copying from a vector with different size.\n"); }
      size_t bytes = _size * sizeof(real);
      if (_device_data != other.data()) {
        qudaMemcpy(_device_data, other._device_data, bytes, qudaMemcpyDeviceToDevice);
      }
    }

    std::vector<real> to_host() const
    {
      std::vector<real> out(_size);
      size_t bytes = _size * sizeof(real);
      qudaMemcpy(out.data(), _device_data, bytes, qudaMemcpyDeviceToHost);
      return out;
    }

    void from_host(const std::vector<real> &host_vector)
    {
      if (host_vector.size() != _size) { errorQuda("Size mismatch: %lu vs %lu.\n", host_vector.size(), _size); }
      qudaMemcpy(_device_data, host_vector.data(), _size * sizeof(real), qudaMemcpyHostToDevice);
    }

    real *data() { return _device_data; }

    const real *data() const { return _device_data; }

    size_t size() const { return _size; }
  };

  void axpby(device_vector<float> &out, float a, const device_vector<float> &x, float b,
      const device_vector<float> &y);

#if 0
  float inner_product(const device_vector<float> &a, const device_vector<float> &b);
#endif

}
