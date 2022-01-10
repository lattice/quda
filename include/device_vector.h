#pragma once

#include <vector>

#include <quda_api.h>
#include <malloc_quda.h>
#include <util_quda.h>
#include <complex_quda.h>

/**
   @file device_vector.h
   Helper functionality for handling a small piece of device memory.
 */

namespace quda
{
  /**
    @brief This class `device_vector` provides a light weighted wrapper for data on device.
  */

  template <class real> class device_vector
  {
    real *_device_data = nullptr;

    size_t _size;

  public:
    /**
       @brief constructor. The device memory is allocated according to the size of the host_vector,
       and the content of the host_vector is copied to the device memory
       @param[in] host_vector the host vector
    */
    device_vector(std::vector<real> &host_vector) : _size(host_vector.size())
    {
      size_t bytes = _size * sizeof(real);
      if (bytes > 0) {
        _device_data = reinterpret_cast<real *>(pool_device_malloc(bytes));
        qudaMemcpy(_device_data, host_vector.data(), bytes, qudaMemcpyHostToDevice);
      }
    }

    /**
       @brief constructor. The device memory is allocated according to the input size, and all bytes
       are set to zero
       @param[in] size_ the size used to allocate the device memory
    */
    device_vector(size_t size_) : _size(size_) { resize(_size); }

    /**
       @brief constructor. No input and no memory allocations are done.
    */
    device_vector() : _size(0) { }

    /**
       @brief resize the device memory. If previously allocated, the underlying device memory is first
       free'ed, and a new piece of device memory is allocated according to the input size
       @param[in] size_ the size used to allocate the device memory
    */
    void resize(size_t size_)
    {
      if (_device_data) { pool_device_free(_device_data); }
      _size = size_;
      size_t bytes = _size * sizeof(real);
      if (bytes > 0) {
        _device_data = reinterpret_cast<real *>(pool_device_malloc(bytes));
        qudaMemset(_device_data, 0, bytes);
      }
    }

    ~device_vector()
    {
      if (_device_data) { pool_device_free(_device_data); }
    }

    /**
       @brief Copy from another device_vector that has the same size. Error is thrown if sizes do not match
       @param[in] other the other device_vector
    */
    void copy(const device_vector &other)
    {
      if (_size != other.size()) { errorQuda("Copying from a vector with different size.\n"); }
      size_t bytes = _size * sizeof(real);
      if (_device_data != other.data()) {
        qudaMemcpy(_device_data, other._device_data, bytes, qudaMemcpyDeviceToDevice);
      }
    }

    /**
      @brief return an std::vector that contains the same data as the device memory
      @return the mentioned std::vector
    */
    std::vector<real> to_host() const
    {
      std::vector<real> out(_size);
      size_t bytes = _size * sizeof(real);
      if (bytes > 0) { qudaMemcpy(out.data(), _device_data, bytes, qudaMemcpyDeviceToHost); }
      return out;
    }

    /**
       @brief Copy from a host vector that has the same size. Error is thrown if sizes do not match
       @param[in] host_vector the host vector
    */
    void from_host(const std::vector<real> &host_vector)
    {
      if (host_vector.size() != _size) { errorQuda("Size mismatch: %lu vs %lu.\n", host_vector.size(), _size); }
      qudaMemcpy(_device_data, host_vector.data(), _size * sizeof(real), qudaMemcpyHostToDevice);
    }

    /**
       @brief Copy data from a device pointer
       @param[in] device_ptr the device pointer
    */
    void from_device(const real *device_ptr)
    {
      if (_size > 0) { qudaMemcpy(_device_data, device_ptr, _size * sizeof(real), qudaMemcpyDeviceToDevice); }
    }

    /**
       @brief Get the underlying pointer
       @return the pointer
    */
    real *data() { return _device_data; }

    /**
       @brief Get the underlying const pointer
       @return the const pointer
    */
    const real *data() const { return _device_data; }

    /**
       @brief Get the underlying size
       @return the size
    */
    size_t size() const { return _size; }
  };

  /**
     @brief Perform out += a * x + b * y for the `float` instantiation
     @param[in/out] out
     @param[in/out] a the first scalar
     @param[in/out] x the first vector
     @param[in/out] b the second scalar
     @param[in/out] y the second vector
  */
  void axpby(device_vector<float> &out, float a, const device_vector<float> &x, float b, const device_vector<float> &y);

} // namespace quda
